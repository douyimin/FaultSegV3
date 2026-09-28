"""Local FaultSegV3 inference (t, h, w), adapted from CIG-Bench.

Inference requires NumPy and PyTorch. Interactive visualization uses cigvis.
Adapted portions are covered by LICENSE-CIG-Bench.
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import warnings
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from framework.FaultSegV3 import FaultSegV3


DEFAULT_WEIGHTS = Path(__file__).resolve().parent / "framework" / "faultsegv3.ckpt"
PROJECT_ROOT = Path(__file__).resolve().parent
MODELSCOPE_DATASET_ID = "douyimin/CIG-Bench-Dataset"
DEMO_DATASET_FILE = "FaultMetricData/F3-GT.npz"
DEMO_INPUT_KEY = "seis_polyfault"
DEFAULT_DEMO_OUTPUT = PROJECT_ROOT / "F3-GT-faultsegv3-prediction.npz"


class _InferenceMemoryExceeded(RuntimeError):
    pass


def _ceil_multiple(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple


def _tensor_z_score_clip(data: torch.Tensor, clip: float) -> torch.Tensor:
    """Match CIG-Bench V3 input normalization; handle constant volumes safely."""
    std = data.std()
    if not torch.isfinite(std) or std <= 0:
        return torch.zeros_like(data)
    z = ((data - data.mean()) / std).clamp(-clip, clip)
    low, high = torch.aminmax(z)
    return (z - low) / (high - low + 1e-6)


def _numpy_z_score_clip(data: np.ndarray, clip: float) -> np.ndarray:
    """Z-score, clip and normalize to [0, 1] for cigvis slices."""
    data = np.asarray(data, dtype=np.float32)
    std = float(data.std())
    if std == 0 or not np.isfinite(std):
        return np.zeros_like(data)
    z = np.clip((data - float(data.mean())) / std, -clip, clip)
    return (z - z.min()) / (z.max() - z.min() + 1e-6)


def _is_oom_error(error: BaseException) -> bool:
    oom_types = tuple(t for t in (
        MemoryError,
        getattr(torch, "OutOfMemoryError", None),
        getattr(torch.cuda, "OutOfMemoryError", None),
    ) if isinstance(t, type))
    if isinstance(error, oom_types):
        return True
    return any(marker in str(error).lower() for marker in (
        "out of memory", "resource_exhausted", "resource exhausted", "cudnn_status_alloc_failed"
    ))


class FaultPredictor:
    """FaultSegV3 inference using local weights; no cig_bench dependency."""

    def __init__(
        self,
        restore_path: str | Path | None = None,
        device: str = "cuda",
        align: int = 16,
        use_autocast: bool = True,
    ) -> None:
        self.restore_path = Path(restore_path) if restore_path is not None else DEFAULT_WEIGHTS
        if not self.restore_path.is_file():
            raise FileNotFoundError(f"FaultSegV3 weights not found: {self.restore_path}")
        if align <= 0 or align % 16:
            raise ValueError("align must be a positive multiple of 16")
        self.device = torch.device(device if device == "cpu" or torch.cuda.is_available() else "cpu")
        if self.device.type == "cuda" and self.device.index is not None:
            torch.cuda.get_device_properties(self.device)  # Validate CUDA index early.
        self.align = align
        self.use_autocast = bool(use_autocast and self.device.type == "cuda")
        state = torch.load(self.restore_path, map_location="cpu", weights_only=False)
        if isinstance(state, dict):
            for key in ("model", "state_dict"):
                if key in state and isinstance(state[key], dict):
                    state = state[key]
                    break
        self.model = FaultSegV3(c=8)
        self.model.load_state_dict(state)
        self.model = self.model.eval().to(self.device)

    def preprocess(
        self, seis: np.ndarray, scale_t: float = 1.0,
        scale_h: float = 1.0, scale_w: float = 1.0,
    ) -> tuple[torch.Tensor, tuple[int, int, int]]:
        array = np.asarray(seis)
        if array.ndim != 3:
            raise ValueError(f"seis must have shape (t, h, w), got {array.shape}")
        if not np.issubdtype(array.dtype, np.number) or not np.isfinite(array).all():
            raise ValueError("seis must contain finite numeric values")
        scales = (scale_t, scale_h, scale_w)
        if any(s <= 0 for s in scales):
            raise ValueError("all scale values must be > 0")
        x = torch.from_numpy(np.ascontiguousarray(array, dtype=np.float32))[None, None]
        target = tuple(max(self.align, int(round(n * s)) // self.align * self.align)
                       for n, s in zip(array.shape, scales))
        if tuple(array.shape) != target:
            x = F.interpolate(x, target, mode="trilinear", align_corners=False)
        return x, tuple(array.shape)

    @torch.no_grad()
    def _predict_once(self, x: torch.Tensor, rank: int, chunk_size: int, clp_s_in: float) -> torch.Tensor:
        model_input = _tensor_z_score_clip(x.to(self.device), clp_s_in) * 2 - 1
        autocast = torch.autocast("cuda") if self.use_autocast else contextlib.nullcontext()
        with autocast:
            # FaultSegV3 already applies sigmoid. Do not apply it twice.
            return self.model(model_input, rank=rank, chunk_size=chunk_size).float().cpu()

    def _release_memory(self) -> None:
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

    def _initial_gpu_budget(self, fraction: float) -> int | None:
        if self.device.type != "cuda":
            return None
        torch.cuda.synchronize(self.device)
        free, _ = torch.cuda.mem_get_info(self.device)
        return int(free * fraction)

    def _run_with_memory_guard(
        self, x: torch.Tensor, rank: int, chunk_size: int,
        clp_s_in: float, gpu_budget_bytes: int | None,
    ) -> torch.Tensor:
        if gpu_budget_bytes is not None:
            torch.cuda.synchronize(self.device)
            free_before, _ = torch.cuda.mem_get_info(self.device)
            base_allocated = torch.cuda.memory_allocated(self.device)
            torch.cuda.reset_peak_memory_stats(self.device)
        try:
            result = self._predict_once(x, rank, chunk_size, clp_s_in)
            if gpu_budget_bytes is not None:
                torch.cuda.synchronize(self.device)
        except Exception as error:
            if not _is_oom_error(error):
                raise
            self._release_memory()
            raise _InferenceMemoryExceeded(str(error)) from error
        if gpu_budget_bytes is not None:
            free_after, _ = torch.cuda.mem_get_info(self.device)
            peak_extra = max(0, torch.cuda.max_memory_allocated(self.device) - base_allocated)
            used = max(peak_extra, free_before - free_after)
            if used > gpu_budget_bytes:
                del result
                self._release_memory()
                raise _InferenceMemoryExceeded(
                    f"GPU memory budget exceeded: {used / 2**20:.1f} MiB used, "
                    f"{gpu_budget_bytes / 2**20:.1f} MiB allowed"
                )
        return result

    @staticmethod
    def _spatial_tile_candidates(h: int, w: int, minimum_size: int) -> Iterator[tuple[int, int]]:
        lengths = (h, w)
        axis_order = (0, 1) if h >= w else (1, 0)
        parts = [1, 1]
        limit = max(1, minimum_size)
        while parts[0] < h or parts[1] < w:
            changed = False
            for axis in axis_order:
                while parts[axis] < lengths[axis]:
                    next_parts = parts[axis] + 1
                    if (lengths[axis] + next_parts - 1) // next_parts < limit:
                        break
                    parts[axis] = next_parts
                    changed = True
                    yield tuple(parts)
            if not changed:
                if limit == 1:
                    break
                limit = max(1, limit // 2)

    def _predict_spatially_tiled(
        self, x: torch.Tensor, grid: tuple[int, int], halo: int,
        rank: int, chunk_size: int, clp_s_in: float,
        gpu_budget_bytes: int | None,
    ) -> torch.Tensor:
        t, h, w = (int(v) for v in x.shape[-3:])
        h_edges = [i * h // grid[0] for i in range(grid[0] + 1)]
        w_edges = [i * w // grid[1] for i in range(grid[1] + 1)]
        output = torch.empty((1, 1, t, h, w), dtype=torch.float32)
        for hi in range(grid[0]):
            core_h0, core_h1 = h_edges[hi:hi + 2]
            patch_h0, patch_h1 = max(0, core_h0 - halo), min(h, core_h1 + halo)
            for wi in range(grid[1]):
                core_w0, core_w1 = w_edges[wi:wi + 2]
                patch_w0, patch_w1 = max(0, core_w0 - halo), min(w, core_w1 + halo)
                patch = x[..., :, patch_h0:patch_h1, patch_w0:patch_w1]
                patch_shape = tuple(int(v) for v in patch.shape[-3:])
                aligned = tuple(_ceil_multiple(v, self.align) for v in patch_shape)
                pad = (0, aligned[2] - patch_shape[2], 0, aligned[1] - patch_shape[1],
                       0, aligned[0] - patch_shape[0])
                if any(pad):
                    patch = F.pad(patch, pad, mode="replicate")
                prediction = self._run_with_memory_guard(
                    patch, rank, chunk_size, clp_s_in, gpu_budget_bytes
                )[..., :patch_shape[0], :patch_shape[1], :patch_shape[2]]
                pred_h0, pred_h1 = core_h0 - patch_h0, core_h1 - patch_h0
                pred_w0, pred_w1 = core_w0 - patch_w0, core_w1 - patch_w0
                output[..., :, core_h0:core_h1, core_w0:core_w1] = prediction[
                    ..., :, pred_h0:pred_h1, pred_w0:pred_w1
                ]
        return output

    def predict(
        self, seis: np.ndarray, rank: int = 4, chunk_size: int = 64,
        threshold: float = 0.5, resize_back: bool = False,
        scale_t: float = 1.0, scale_h: float = 1.0, scale_w: float = 1.0,
        scale: float = 1.0, clp_s_in: float = 3.0,
        oom_fallback: bool = True, gpu_memory_fraction: float = 0.9,
        fallback_min_size: int = 256, fallback_halo: int = 16,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (fault probability, used seismic), each shaped (t, h, w).

        On OOM, split only h/w, retaining the full t axis. Tile boundaries
        may differ slightly from full-volume inference due to local z-scoring.
        """
        if rank not in (0, 1, 2, 3, 4) or chunk_size <= 0:
            raise ValueError("rank must be 0..4 and chunk_size must be > 0")
        if not 0 <= threshold <= 1 or clp_s_in <= 0:
            raise ValueError("threshold must be in [0,1] and clp_s_in > 0")
        if not 0 < gpu_memory_fraction <= 1 or fallback_min_size <= 0 or fallback_halo < 0:
            raise ValueError("invalid GPU budget or fallback tile settings")
        if scale != 1.0:
            scale_t = scale_h = scale_w = scale
        x, original_shape = self.preprocess(seis, scale_t, scale_h, scale_w)
        budget = self._initial_gpu_budget(gpu_memory_fraction)
        try:
            result = self._run_with_memory_guard(x, rank, chunk_size, clp_s_in, budget)
        except _InferenceMemoryExceeded as initial_error:
            if not oom_fallback:
                raise RuntimeError(
                    f"FaultSegV3 inference at {tuple(x.shape[-3:])} ran out of memory; "
                    "retry with oom_fallback=True"
                ) from initial_error
            warnings.warn(
                f"Full-volume inference ran out of memory ({initial_error}); "
                "retrying with spatial tiles.", RuntimeWarning, stacklevel=2
            )
            h, w = (int(v) for v in x.shape[-2:])
            last_error: BaseException = initial_error
            result = None
            for grid in self._spatial_tile_candidates(h, w, fallback_min_size):
                try:
                    result = self._predict_spatially_tiled(
                        x, grid, fallback_halo, rank, chunk_size, clp_s_in, budget
                    )
                    break
                except _InferenceMemoryExceeded as error:
                    last_error = error
                    self._release_memory()
            if result is None:
                raise RuntimeError(
                    "Spatial OOM fallback exhausted all valid h/w grids; "
                    f"last failure: {last_error}"
                ) from last_error
        result = torch.where(result > threshold, result, torch.zeros_like(result))
        if resize_back:
            result = F.interpolate(result, original_shape, mode="nearest")
            # Return the exact original input and release the potentially very
            # large upsampled CPU tensor instead of interpolating it again.
            original = np.ascontiguousarray(np.asarray(seis), dtype=np.float32)
            x = torch.from_numpy(original)[None, None]
        return result[0, 0].numpy(), x[0, 0].numpy()

    @staticmethod
    def visualize(
        seis_np: np.ndarray,
        result_np: np.ndarray,
        seis_cmap: str = "gray",
        fg_cmap_name: str = "jet",
        show_clip_s: float = 3.0,
        show_input_panel: bool = True,
    ) -> None:
        """Display the CIG-Bench seismic/segmentation overlay in cigvis."""
        import cigvis
        from cigvis import colormap

        if show_clip_s <= 0:
            raise ValueError("show_clip_s must be positive")
        seis_vol = np.asarray(seis_np, dtype=np.float32)
        mask_vol = np.asarray(result_np, dtype=np.float32)
        if seis_vol.ndim != 3 or mask_vol.shape != seis_vol.shape:
            raise ValueError("seismic input and prediction must have the same (t, h, w) shape")
        seis_vol = seis_vol.transpose(1, 2, 0)
        mask_vol = mask_vol.transpose(1, 2, 0)
        fg_cmap = colormap.set_alpha_except_min(fg_cmap_name, alpha=1)
        normalized = _numpy_z_score_clip(seis_vol, show_clip_s)
        base_node = cigvis.create_slices(normalized, cmap=seis_cmap)
        overlay = cigvis.add_mask(base_node, mask_vol, cmaps=fg_cmap, interpolation="nearest")
        if show_input_panel:
            ref_node = cigvis.create_slices(normalized, cmap=seis_cmap)
            cigvis.plot3D([ref_node, overlay], grid=[1, 2], share=1)
        else:
            cigvis.plot3D(overlay)

    @staticmethod
    def preview_input(
        seis: np.ndarray, cmap: str = "seismic", clp_s: float = 4.0,
    ) -> None:
        """Display the input seismic slices in cigvis."""
        import cigvis

        if clp_s <= 0:
            raise ValueError("clp_s must be positive")
        volume = np.asarray(seis, dtype=np.float32)
        if volume.ndim != 3:
            raise ValueError("seis must have shape (t, h, w)")
        volume = _numpy_z_score_clip(volume.transpose(1, 2, 0), clp_s)
        cigvis.plot3D(cigvis.create_slices(volume, cmap=cmap))


def _load_volume(path: Path, key: str | None) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        return np.load(path, allow_pickle=False)
    if path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as data:
            selected = key or ("seis" if "seis" in data else None)
            if selected is None:
                if len(data.files) != 1:
                    raise ValueError(f"NPZ has multiple arrays {data.files}; provide --input-key")
                selected = data.files[0]
            return np.asarray(data[selected])
    raise ValueError("input must be .npy or .npz")


def _ensure_demo_dataset(cache_dir: Path | None = None) -> Path:
    """Return cached F3-GT.npz or download it with ModelScope's official API."""
    cache_root = cache_dir or (PROJECT_ROOT / ".modelscope")
    visible_candidates = (
        PROJECT_ROOT / DEMO_DATASET_FILE,
        PROJECT_ROOT / "F3-GT.npz",
    )
    for candidate in visible_candidates:
        if candidate.is_file():
            print(f"Using cached demo dataset: {candidate}")
            return candidate
    if cache_root.is_dir():
        cached = next(
            (path for path in cache_root.rglob("F3-GT.npz") if path.is_file()),
            None,
        )
        if cached is not None:
            print(f"Using cached ModelScope dataset: {cached}")
            return cached

    try:
        from modelscope.hub.file_download import dataset_file_download
    except ImportError as error:
        raise RuntimeError(
            "The zero-argument demo requires the ModelScope SDK. "
            "Install it with: pip install modelscope"
        ) from error

    print(f"Downloading {DEMO_DATASET_FILE} with the ModelScope official API")
    downloaded = dataset_file_download(
        dataset_id=MODELSCOPE_DATASET_ID,
        file_path=DEMO_DATASET_FILE,
        revision="master",
        cache_dir=str(cache_root),
        user_agent={"invoked_by": "faultsegv3"},
    )
    path = Path(downloaded)
    if not path.is_file():
        raise RuntimeError(f"ModelScope did not return a valid dataset file: {path}")
    return path


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Local FaultSegV3 Rank4 inference")
    parser.add_argument(
        "input", nargs="?", type=Path,
        help="input seismic .npy/.npz, shape (t,h,w); omit both paths for the F3 demo",
    )
    parser.add_argument(
        "output", nargs="?", type=Path,
        help="output .npz; omit both paths for the F3 demo",
    )
    parser.add_argument("--input-key", help="array name inside input NPZ (default: seis or sole array)")
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--device", default="cuda", help="cuda, cuda:0, cpu (CUDA falls back to CPU)")
    parser.add_argument("--rank", type=int, default=4, choices=range(5))
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="probability cutoff (default: 0.5)")
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--scale-t", type=float, default=1.0)
    parser.add_argument("--scale-h", type=float, default=1.0)
    parser.add_argument("--scale-w", type=float, default=1.0)
    parser.add_argument("--resize-back", action="store_true")
    parser.add_argument("--no-autocast", action="store_true")
    parser.add_argument("--no-oom-fallback", action="store_true")
    parser.add_argument("--gpu-memory-fraction", type=float, default=0.9)
    parser.add_argument("--fallback-min-size", type=int, default=256)
    parser.add_argument("--fallback-halo", type=int, default=16)
    parser.add_argument("--visualize", action="store_true",
                        help="show the seismic/prediction overlay in cigvis")
    parser.add_argument("--seis-cmap", default="gray")
    parser.add_argument("--fg-cmap", default="jet")
    parser.add_argument("--show-clip-s", type=float, default=3.0)
    parser.add_argument("--hide-input-panel", action="store_true")
    args = parser.parse_args(argv)
    demo_mode = args.input is None and args.output is None
    if (args.input is None) != (args.output is None):
        parser.error("input and output must be provided together")

    if demo_mode:
        args.input = _ensure_demo_dataset()
        args.output = DEFAULT_DEMO_OUTPUT
        args.input_key = DEMO_INPUT_KEY
        args.scale = 2.0
        args.scale_t = args.scale_h = args.scale_w = 1.0
        args.resize_back = True
        args.visualize = True
        print(
            f"Zero-argument F3 demo: key={DEMO_INPUT_KEY}, scale=2, "
            "nearest downsampling, Rank4 inference"
        )

    if args.output.suffix.lower() != ".npz":
        parser.error("output must have .npz suffix")
    predictor = FaultPredictor(args.weights, device=args.device, use_autocast=not args.no_autocast)
    seis = _load_volume(args.input, args.input_key)
    options = dict(rank=args.rank, chunk_size=args.chunk_size, scale=args.scale,
                   scale_t=args.scale_t, scale_h=args.scale_h, scale_w=args.scale_w,
                   resize_back=args.resize_back, oom_fallback=not args.no_oom_fallback,
                   gpu_memory_fraction=args.gpu_memory_fraction,
                   fallback_min_size=args.fallback_min_size, fallback_halo=args.fallback_halo)
    prediction, used_input = predictor.predict(seis, threshold=args.threshold, **options)
    display_input = np.asarray(seis, dtype=np.float32) if demo_mode else used_input
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, prediction=prediction, input=display_input)
    print(f"Saved {args.output} (shape {prediction.shape})")
    if args.visualize:
        predictor.visualize(
            display_input, prediction, seis_cmap=args.seis_cmap,
            fg_cmap_name=args.fg_cmap, show_clip_s=args.show_clip_s,
            show_input_panel=not args.hide_input_panel,
        )


if __name__ == "__main__":
    main()
