import os
import pytorch_lightning as pl
import torch
import matplotlib.pyplot as plt
import time


class EMACallback(pl.Callback):
    """
    实现模型权重的指数移动平均(EMA)。

    这个实现会追踪所有模型参数，包括BN层的均值和方差。
    """

    def __init__(self, decay=0.9999, ema_device=None, ema_scope=None, update_bn=True):
        """
        初始化EMA回调。
        Args:
            decay: EMA衰减率，越接近1则历史权重占比越大
            ema_device: 存储EMA权重的设备，默认与原始模型相同
            ema_scope: 模型的哪些部分应该应用EMA，如'model'表示只对生成器应用EMA
            update_bn: 是否更新BN层的running_mean和running_var
        """
        super().__init__()
        self.decay = decay
        self.ema_device = ema_device
        self.ema_scope = ema_scope  # 如果为None，则应用到整个模型
        self.update_bn = update_bn  # 是否更新BN层的统计量
        self.shadow = {}
        self.backup = {}
        self.is_ema_active = False

    def on_train_start(self, trainer, pl_module):
        """训练开始时初始化EMA模型"""
        for name, param in self._get_model_parameters(pl_module):
            device = self.ema_device or param.device
            if name in self.shadow:
                # Preserve EMA history restored from a Lightning checkpoint.
                self.shadow[name] = self.shadow[name].to(device)
            else:
                self.shadow[name] = param.data.clone().to(device)

    def _get_model_parameters(self, pl_module):
        """获取应该应用EMA的模型参数"""
        if self.ema_scope is None:
            # 应用到整个模型
            for name, param in pl_module.named_parameters():
                if param.requires_grad:
                    yield name, param

            # 如果更新BN层统计量，也获取相关缓冲区
            if self.update_bn:
                for name, buf in pl_module.named_buffers():
                    if 'running_mean' in name or 'running_var' in name:
                        yield name, buf
        else:
            # 只应用到指定子模型
            prefix = self.ema_scope + '.'
            model = getattr(pl_module, self.ema_scope)

            for name, param in model.named_parameters():
                if param.requires_grad:
                    yield prefix + name, param

            # 如果更新BN层统计量，也获取相关缓冲区
            if self.update_bn:
                for name, buf in model.named_buffers():
                    if 'running_mean' in name or 'running_var' in name:
                        yield prefix + name, buf

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """每个训练批次结束后更新EMA权重"""
        # 更新EMA权重
        for name, param in self._get_model_parameters(pl_module):
            # 使用指数移动平均更新权重
            self.shadow[name] = self.shadow[name] * self.decay + param.data * (1 - self.decay)

    def apply_ema(self, pl_module):
        """应用EMA权重到模型（用于推理或保存）"""
        if self.is_ema_active:
            return

        self.is_ema_active = True

        # 备份原始权重并应用EMA权重
        for name, param in self._get_model_parameters(pl_module):
            self.backup[name] = param.data.clone()
            param.data.copy_(self.shadow[name])

    def restore(self, pl_module):
        """恢复原始权重"""
        if not self.is_ema_active:
            return

        self.is_ema_active = False

        # 恢复原始权重
        for name, param in self._get_model_parameters(pl_module):
            param.data.copy_(self.backup[name])

        self.backup = {}

    def state_dict(self):
        """返回EMA状态字典（用于保存）"""
        return {
            'decay': self.decay,
            'shadow': self.shadow,
            'backup': self.backup,
            'is_ema_active': self.is_ema_active
        }

    def load_state_dict(self, state_dict):
        """从状态字典加载EMA状态"""
        self.decay = state_dict['decay']
        self.shadow = state_dict['shadow']
        self.backup = state_dict['backup']
        self.is_ema_active = state_dict['is_ema_active']


class EMASwitchContext:
    """上下文管理器，用于临时切换到EMA模型"""

    def __init__(self, callback, pl_module):
        self.callback = callback
        self.pl_module = pl_module
        self.training = {}  # 保存训练状态

    def __enter__(self):
        # 应用EMA权重
        self.callback.apply_ema(self.pl_module)

        # 保存训练状态并切换到eval模式
        if self.callback.ema_scope is None:
            # 作用于整个模型
            self.training['model'] = self.pl_module.training
            self.pl_module.eval()
        else:
            # 只作用于特定子模型
            model = getattr(self.pl_module, self.callback.ema_scope)
            self.training[self.callback.ema_scope] = model.training
            model.eval()

        return self.pl_module

    def __exit__(self, *args):
        # 恢复原始权重
        self.callback.restore(self.pl_module)

        # 恢复原始训练状态
        if self.callback.ema_scope is None:
            # 作用于整个模型
            if self.training.get('model', True):
                self.pl_module.train()
        else:
            # 只作用于特定子模型
            model = getattr(self.pl_module, self.callback.ema_scope)
            if self.training.get(self.callback.ema_scope, True):
                model.train()

class CustomCallback(pl.Callback):
    def __init__(self, print_every_n_steps=10, save_weights_every_n_steps=1000, save_dir="model_weights",
                 viz_every_n_steps=500, results_dir="results", viz_samples=1, avg_window=10):
        super().__init__()
        self.print_every_n_steps = print_every_n_steps
        self.save_weights_every_n_steps = save_weights_every_n_steps
        self.viz_every_n_steps = viz_every_n_steps  # 可视化间隔步数
        self.viz_samples = viz_samples  # 每次可视化保存的样本数量
        self.save_dir = save_dir
        self.results_dir = results_dir  # 结果保存目录
        self.avg_window = avg_window  # 平均损失窗口大小

        # 创建保存目录
        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)

        self.last_print_time = time.time()  # 记录初始时间

        # 存储损失历史
        self.seg_loss_history = []


    def on_fit_start(self, trainer, pl_module):
        # 禁用 Lightning 的默认进度条
        trainer.enable_progress_bar = False

    def show_results(self, tensor_dict: dict, save_path: str, sample_idx=0):
        """
        结果可视化方法
        
        Args:
            tensor_dict: 包含各种张量的字典
            save_path: 保存路径
            sample_idx: 样本索引，用于多样本保存时区分文件名
        """
        dict_len = len(tensor_dict.keys())
        plt.figure(figsize=(30, 30))
        for idx, key in enumerate(tensor_dict.keys()):
            data = tensor_dict[key][sample_idx, 0]  # 使用指定的样本索引
            t, h, w = data.shape
            plt.subplot(3, dict_len, idx + 1)
            plt.title(key)
            if key == 'seis':
                plt.imshow(data[:, int(h / 2)], cmap='gray')
            else:
                plt.imshow(data[:, int(h / 2)], cmap='jet')
            plt.subplot(3, dict_len, idx + dict_len + 1)
            plt.title(key)
            if key == 'seis':
                plt.imshow(data[:, :, int(w / 2)], cmap='gray')
            else:
                plt.imshow(data[:, :, int(w / 2)], cmap='jet')

            plt.subplot(3, dict_len, idx + dict_len * 2 + 1)
            plt.title(key)
            if key == 'seis':
                plt.imshow(data[int(t / 2), :, :], cmap='gray')
            else:
                plt.imshow(data[int(t / 2), :, :], cmap='jet')

        plt.savefig(save_path)
        plt.close()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        # 获取当前损失值并存储
        if trainer.is_global_zero:
            seg_loss = trainer.callback_metrics.get("seg_loss", torch.tensor(0.0))

            # 将张量转换为浮点数
            if isinstance(seg_loss, torch.Tensor):
                seg_loss = seg_loss.item()
            # 添加到历史记录
            self.seg_loss_history.append(seg_loss)

            # 保留最近的 avg_window 个值
            self.seg_loss_history = self.seg_loss_history[-self.avg_window:]
   
            # 每n步打印一次平均损失
            if pl_module.global_step % self.print_every_n_steps == 0 and self.seg_loss_history:
                # 计算时间差
                current_time = time.time()
                elapsed_time = current_time - self.last_print_time
                self.last_print_time = current_time  # 更新上次打印时间
                # 计算平均损失
                avg_seg_loss = sum(self.seg_loss_history) / len(self.seg_loss_history)


                # 打印平均损失和时间差
                print(f"Step {pl_module.global_step}: time taken: {elapsed_time:.2f}, "
                      f"seg_loss={avg_seg_loss:.4f}")

            # 可视化训练结果
            if pl_module.global_step % self.viz_every_n_steps == 0 and pl_module.global_step > 0:
                # 只有在主进程中执行可视化
                if trainer.is_global_zero and hasattr(pl_module, 'last_batch') and hasattr(pl_module,
                                                                                           'last_pred_target'):
                    # 从模型中获取上一批次的数据和预测结果
                    seis = pl_module.last_batch['seis']
                    target = pl_module.last_batch['target']
                    pred_target = pl_module.last_pred_target

                    # 计算实际要保存的样本数量（不超过批次大小）
                    batch_size = seis.size(0)
                    actual_samples = min(self.viz_samples, batch_size)
                    print(f"将保存 {actual_samples} 个样本的可视化结果")

                    # 确定是否需要EMA推理
                    ema_callback = None
                    for callback in trainer.callbacks:
                        if hasattr(callback, 'apply_ema') and hasattr(callback, 'restore'):
                            ema_callback = callback
                            break

                    # 对指定数量的样本进行可视化
                    for sample_idx in range(actual_samples):
                        # 为每个样本创建不同的文件名
                        if actual_samples > 1:
                            save_path = os.path.join(self.results_dir,
                                                     f"result_step_{str(pl_module.global_step).zfill(8)}_sample_{sample_idx + 1}.png")
                        else:
                            save_path = os.path.join(self.results_dir,
                                                     f"result_step_{str(pl_module.global_step).zfill(8)}.png")

                        # 获取标准预测结果
                        normal_pred = pred_target.detach().cpu().float().numpy()

                        if ema_callback:
                            # 使用EMA权重进行推理
                            with EMASwitchContext(ema_callback, pl_module):
                                with torch.no_grad():
                                    with torch.autocast('cuda'):
                                        ema_pred = pl_module.model(seis)

                                # 使用普通结果和EMA结果一起可视化
                                self.show_results(
                                    {
                                        'seis': seis.detach().cpu().float().numpy(),
                                        'target': target.detach().cpu().float().numpy(),
                                        'pred_target': normal_pred,
                                        'ema_pred': ema_pred.detach().cpu().float().numpy(),
                                    }, save_path, sample_idx)
                        else:
                            # 只保存普通预测结果
                            self.show_results(
                                {
                                    'seis': seis.detach().cpu().float().numpy(),
                                    'target': target.detach().cpu().float().numpy(),
                                    'pred_target': normal_pred,
                                }, save_path, sample_idx)

                    print(f"保存了可视化结果到: {self.results_dir}")

            # 保存模型权重
            if pl_module.global_step % self.save_weights_every_n_steps == 0 and pl_module.global_step > 0:
                if trainer.is_global_zero:  # 只在主进程保存
                    # 获取当前的优化器
                    optm = pl_module.optimizers()

                    # 标准模型参数，包含更多状态
                    param = {
                        'model': pl_module.model.state_dict(),
                        'optim': optm.state_dict(),
                        'global_step': pl_module.global_step    # 保存全局步数
                    }

                    # 保存标准模型
                    weights_path = os.path.join(self.save_dir, f"model_step_{pl_module.global_step}.pth")
                    torch.save(param, weights_path)
                    print(f"已保存标准模型参数到: {weights_path}")

                    # 检查是否存在EMA回调
                    ema_callback = None
                    for callback in trainer.callbacks:
                        if hasattr(callback, 'apply_ema') and hasattr(callback, 'restore'):
                            ema_callback = callback
                            break

                    # 如果存在EMA回调，保存EMA模型
                    if ema_callback:
                        # 切换到EMA模型
                        with EMASwitchContext(ema_callback, pl_module):
                            # 准备EMA模型参数
                            ema_param = {
                                'model': pl_module.model.state_dict(),
                                'optim': optm.state_dict(),
                                'global_step': pl_module.global_step,
                                'is_ema': True
                            }

                            # 保存EMA模型
                            ema_weights_path = os.path.join(self.save_dir,
                                                            f"model_ema_step_{pl_module.global_step}.pth")
                            torch.save(ema_param, ema_weights_path)
                            print(f"已保存EMA模型参数到: {ema_weights_path}")

                            # 如果需要，也可以保存EMA状态
                            # if hasattr(ema_callback, 'state_dict'):
                            #     ema_state = {
                            #         'ema_state': ema_callback.state_dict()
                            #     }
                            #     ema_state_path = os.path.join(self.save_dir, f"ema_state_{pl_module.global_step}.pth")
                            #     torch.save(ema_state, ema_state_path)
