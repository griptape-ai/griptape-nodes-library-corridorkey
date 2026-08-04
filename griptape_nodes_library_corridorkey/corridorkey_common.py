"""Shared helpers for the CorridorKey image and video inference nodes.

Houses model repo constants, device/engine/BiRefNet/GVM loading (with
module-level caches shared across both nodes), and the image
encode/decode helpers used when building per-frame inputs/outputs.
"""

import io
import logging
import os
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import torch
from griptape.artifacts import ImageArtifact, ImageUrlArtifact
from griptape_nodes.common.sequences import MissingItemPolicy, NoTokenBehavior, Sequence, SequenceEntry
from griptape_nodes.files.file import File
from griptape_nodes.files.project_file import ProjectFileDestination
from griptape_nodes.retained_mode.events.os_events import ScanSequencesRequest, ScanSequencesResultSuccess
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes
from PIL import Image

logger = logging.getLogger("corridorkey_library")

# Real EXR reads/writes go through griptape_nodes_openexr, but some vendored
# CorridorKey code (clip_manager.py, gvm_core/wrapper.py) still touches EXR via
# cv2, which silently no-ops on .exr unless this is set before cv2's first use.
os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

_IMAGE_SEQUENCE_EXTENSIONS = (".exr", ".png", ".jpg", ".jpeg", ".tif", ".tiff")

# CorridorKey screen-keying checkpoints. The two checkpoints share an identical
# GreenFormer architecture; only the trained weights differ.
CORRIDORKEY_MODEL_REPO_IDS = [
    "nikopueringer/CorridorKey_v1.0",
    "nikopueringer/CorridorKeyBlue_1.0",
]

# Map repo IDs to the screen_color argument expected by CorridorKey's backend factory.
SCREEN_COLOR_BY_REPO = {
    "nikopueringer/CorridorKey_v1.0": "green",
    "nikopueringer/CorridorKeyBlue_1.0": "blue",
}

# BiRefNet variants used to generate a coarse alpha hint when one is not provided.
# Matting variants are best for soft-edge subjects (hair/fur); the general models
# are better for hard-edged objects.
BIREFNET_MODEL_REPO_IDS = [
    "ZhengPeng7/BiRefNet-matting",
    "ZhengPeng7/BiRefNet",
]

# Map BiRefNet HF repo IDs to the `usage` keys understood by BiRefNetHandler.
# Keys must match BiRefNetModule.wrapper.usage_to_weights_file.
BIREFNET_USAGE_BY_REPO = {
    "ZhengPeng7/BiRefNet-matting": "Matting",
    "ZhengPeng7/BiRefNet": "General",
}

# GVM has a single published checkpoint. Like CorridorKey/BiRefNet it gets a
# HuggingFaceRepoParameter dropdown; the weights are resolved from the standard
# HF hub cache rather than downloaded into a custom local_dir.
GVM_MODEL_REPO_ID = "geyongtao/gvm"

# VideoMaMa needs two separate repos, each with its own HuggingFaceRepoParameter
# dropdown; both are resolved from the standard HF hub cache.
VIDEOMAMA_UNET_REPO_ID = "SammyLim/VideoMaMa"
VIDEOMAMA_BASE_REPO_ID = "stabilityai/stable-video-diffusion-img2vid-xt"

# Module-level caches, shared by both the image and video inference nodes, so
# switching between them doesn't force a reload/re-download or re-trigger
# torch.compile autotuning. Keyed the same way the per-class caches used to be.
_engine_cache: dict[tuple[str, str, int], Any] = {}
_birefnet_cache: dict[tuple[str, str], Any] = {}
_gvm_cache: dict[tuple[str, str], Any] = {}
_videomama_cache: dict[tuple[str, str, str], Any] = {}


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def artifact_to_bytes(artifact: ImageArtifact | ImageUrlArtifact) -> bytes:
    if isinstance(artifact, ImageUrlArtifact):
        return File(artifact.value).read_bytes()
    # ImageArtifact stores raw bytes in .value
    return artifact.value


def decode_rgb_image(image_bytes: bytes) -> np.ndarray:
    """Decode image bytes into a float32 [H, W, 3] sRGB array in [0, 1]."""
    pil = Image.open(io.BytesIO(image_bytes))
    return pil_to_rgb_array(pil)


def pil_to_rgb_array(pil: Image.Image) -> np.ndarray:
    """Convert an already-decoded RGB PIL image into a float32 [H, W, 3] array in [0, 1]."""
    arr = np.asarray(pil.convert("RGB"), dtype=np.float32) / 255.0
    return arr


def decode_alpha_image(image_bytes: bytes) -> np.ndarray:
    """Decode an alpha-hint image into a float32 [H, W] array in [0, 1]."""
    pil = Image.open(io.BytesIO(image_bytes)).convert("L")
    arr = np.asarray(pil, dtype=np.float32) / 255.0
    return arr


def resize_alpha_to_shape(alpha_np: np.ndarray, height: int, width: int) -> np.ndarray:
    """Resize a float32 [H, W] alpha-hint array to match a target (height, width)."""
    if alpha_np.shape[:2] == (height, width):
        return alpha_np
    hint_pil = Image.fromarray((alpha_np * 255.0).astype(np.uint8), mode="L")
    hint_pil = hint_pil.resize((width, height))
    return np.asarray(hint_pil, dtype=np.float32) / 255.0


def encode_grayscale_png(alpha: np.ndarray) -> bytes:
    """Encode a single-channel float32 [H, W] or [H, W, 1] alpha matte as 8-bit PNG."""
    if alpha.ndim == 3:
        alpha = alpha[..., 0]
    arr = np.clip(alpha * 255.0, 0, 255).astype(np.uint8)
    pil = Image.fromarray(arr, mode="L")
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return buf.getvalue()


def encode_rgb_png(rgb: np.ndarray) -> bytes:
    """Encode a float32 [H, W, 3] sRGB image as 8-bit PNG."""
    arr = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
    pil = Image.fromarray(arr, mode="RGB")
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return buf.getvalue()


def encode_rgba_png(rgba: np.ndarray) -> bytes:
    """Encode a float32 [H, W, 4] straight sRGB RGBA image as 8-bit PNG."""
    arr = np.clip(rgba * 255.0, 0, 255).astype(np.uint8)
    pil = Image.fromarray(arr, mode="RGBA")
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return buf.getvalue()


def save_image_artifact(image_bytes: bytes, suffix: str) -> ImageUrlArtifact:
    filename = f"corridorkey_{suffix}_{uuid.uuid4().hex[:8]}.png"
    url = GriptapeNodes.StaticFilesManager().save_static_file(image_bytes, filename)
    return ImageUrlArtifact(url)


def save_video_artifact(video_bytes: bytes, suffix: str):
    from griptape.artifacts import VideoUrlArtifact

    filename = f"corridorkey_{suffix}_{uuid.uuid4().hex[:8]}.mp4"
    url = GriptapeNodes.StaticFilesManager().save_static_file(video_bytes, filename)
    return VideoUrlArtifact(url)


def _patch_timm_hiera_view_bug() -> None:
    """Patch timm's Hiera MaskUnitAttention.forward to use .reshape() instead of .view().

    CorridorKeyModule's encoder is a timm "hiera_base_plus_224" model. Its
    MaskUnitAttention.forward() does `q.view(...)` on a tensor produced by a prior
    `.permute()`, which requires strides .view() can't always satisfy -- this raises
    "view size is not compatible with input tensor's size and stride" on MPS (the
    stride layout that trips it doesn't come up on CPU/CUDA). .reshape() is a
    drop-in replacement (it only copies when a view isn't possible), so patch the
    pip-installed timm module in place rather than vendoring a modified copy.

    The same non-contiguous q/k/v also trips MPS's internal
    F.scaled_dot_product_attention kernel (it does its own .view() on the inputs),
    so q/k/v are made contiguous before that call too.
    """
    from timm.models import hiera

    if getattr(hiera.MaskUnitAttention, "_corridorkey_reshape_patched", False):
        return

    def patched_forward(self, x):
        import torch.nn.functional as torch_functional

        b, n, _ = x.shape
        num_windows = (n // (self.q_stride * self.window_size)) if self.use_mask_unit_attn else 1
        qkv = self.qkv(x).reshape(b, -1, num_windows, 3, self.heads, self.head_dim).permute(3, 0, 4, 2, 1, 5)
        q, k, v = qkv.unbind(0)

        if self.q_stride > 1:
            q = q.reshape(b, self.heads, num_windows, self.q_stride, -1, self.head_dim).amax(dim=3)

        if self.fused_attn:
            x = torch_functional.scaled_dot_product_attention(q.contiguous(), k.contiguous(), v.contiguous())
        else:
            attn = (q * self.scale) @ k.transpose(-1, -2)
            attn = attn.softmax(dim=-1)
            x = attn @ v

        x = x.transpose(1, 3).reshape(b, -1, self.dim_out)
        return self.proj(x)

    hiera.MaskUnitAttention.forward = patched_forward
    setattr(hiera.MaskUnitAttention, "_corridorkey_reshape_patched", True)  # noqa: B010


def _patch_connected_components_mps_race() -> None:
    """Patch CorridorKeyModule's connected_components to fix an MPS boolean-index race.

    connected_components() (color_utils.py) evaluates `mask == 1` twice per loop
    iteration -- once to select the assignment target, once to gather from the
    max_pool2d result -- across up to `max_iterations` iterations. On MPS, the two
    evaluations occasionally disagree on their true-count (an async-execution race
    in the MPS boolean-indexing kernel), raising "shape mismatch: value tensor ...
    cannot be broadcast to indexing result ...". `mask` never changes across the
    loop, so hoisting a single `mask == 1` evaluation out of the loop and reusing it
    for both sides removes the race.
    """
    from CorridorKeyModule.core import color_utils

    if getattr(color_utils, "_corridorkey_cc_race_patched", False):
        return

    def patched_connected_components(mask, min_component_distance=1, max_iterations=100):
        bs, _, h, w = mask.shape
        comp = (torch.randperm(bs * w * h, device=mask.device, dtype=torch.float32) + 1.1).view(mask.shape)
        idx = mask == 1
        comp[~idx] = 0

        for _ in range(max_iterations):
            comp[idx] = color_utils.F.max_pool2d(
                comp, kernel_size=(2 * min_component_distance) + 1, stride=1, padding=min_component_distance
            )[idx]

        comp = comp.long()
        unique_labels = torch.unique(comp)
        if unique_labels[0] != 0:
            unique_labels = torch.cat([torch.tensor([0], device=mask.device), unique_labels])
        label_map = torch.zeros(int(unique_labels.max().item()) + 1, dtype=torch.long, device=mask.device)
        label_map[unique_labels] = torch.arange(len(unique_labels), device=mask.device)
        return label_map[comp]

    color_utils.connected_components = patched_connected_components
    color_utils._corridorkey_cc_race_patched = True


def load_engine(model_repo_id: str, device: str, img_size: int):
    """Load and cache the CorridorKey engine for (repo, device, img_size)."""
    # Deferred imports: these resolve only after the advanced library has
    # pip-installed CorridorKeyModule and added the submodule root to sys.path.
    from CorridorKeyModule import backend as ck_backend
    from CorridorKeyModule.backend import create_engine

    _patch_timm_hiera_view_bug()
    _patch_connected_components_mps_race()

    # Upstream bug: backend.py defines CHECKPOINT_DIR twice (line 22 absolute,
    # line 74 relative). The second definition wins at import time and points
    # at "CorridorKeyModule/checkpoints" relative to CWD, so checkpoint copies
    # fail with FileNotFoundError. Restore the absolute path.
    ck_backend.CHECKPOINT_DIR = os.path.join(os.path.dirname(os.path.abspath(ck_backend.__file__)), "checkpoints")
    os.makedirs(ck_backend.CHECKPOINT_DIR, exist_ok=True)

    key = (model_repo_id, device, img_size)
    if key in _engine_cache:
        return _engine_cache[key]

    screen_color = SCREEN_COLOR_BY_REPO[model_repo_id]
    logger.info(
        "Loading CorridorKey engine: repo=%s screen_color=%s device=%s img_size=%d",
        model_repo_id,
        screen_color,
        device,
        img_size,
    )
    engine = create_engine(
        backend="torch",
        device=device,
        img_size=img_size,
        screen_color=screen_color,
    )
    _engine_cache[key] = engine
    return engine


def load_birefnet(birefnet_repo_id: str, device: str):
    """Load and cache a BiRefNetHandler for (repo, device)."""
    # Deferred import: BiRefNetModule is exposed via sys.path.insert in the
    # advanced library loader, since it isn't included in the hatch wheel.
    import warnings

    from BiRefNetModule import wrapper as birefnet_wrapper
    from BiRefNetModule.wrapper import BiRefNetHandler
    from huggingface_hub.utils.tqdm import disable_progress_bars, enable_progress_bars

    key = (birefnet_repo_id, device)
    if key in _birefnet_cache:
        return _birefnet_cache[key]

    usage = BIREFNET_USAGE_BY_REPO[birefnet_repo_id]
    logger.info("Loading BiRefNet handler: repo=%s usage=%s device=%s", birefnet_repo_id, usage, device)

    # Upstream bug: BiRefNetModule.wrapper hardcodes trust_remote_code=False, but
    # BiRefNet's HF checkpoints ship custom modeling code (birefnet.py) that
    # AutoModelForImageSegmentation cannot load without it. wrapper.py is a vendored
    # git submodule, so patch its from_pretrained rather than editing it directly.
    original_from_pretrained = birefnet_wrapper.AutoModelForImageSegmentation.from_pretrained

    def _from_pretrained_trusted(*args, **kwargs):
        kwargs["trust_remote_code"] = True
        return original_from_pretrained(*args, **kwargs)

    birefnet_wrapper.AutoModelForImageSegmentation.from_pretrained = _from_pretrained_trusted
    # Also silence the deprecated local_dir_use_symlinks warning and the download
    # progress bar wrapper.py's snapshot_download call emits straight to stderr,
    # bypassing `logging` and showing up unformatted in the node's log.
    disable_progress_bars()
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*local_dir_use_symlinks.*")
            handler = BiRefNetHandler(device=device, usage=usage)
    finally:
        birefnet_wrapper.AutoModelForImageSegmentation.from_pretrained = original_from_pretrained
        enable_progress_bars()

    _birefnet_cache[key] = handler
    return handler


def run_birefnet(handler, image_rgb_float: np.ndarray) -> np.ndarray:
    """Run BiRefNet on a float32 [H, W, 3] sRGB image and return a float32 [H, W] alpha hint.

    BiRefNetHandler only ships a `process()` method that operates on file paths
    and writes outputs to disk, so we replicate its in-memory inference path
    (preprocess -> sigmoid -> resize-to-source) directly against the loaded
    model. This mirrors the public `process()` flow in BiRefNetModule.wrapper.
    """
    # Deferred imports: BiRefNetModule is exposed via sys.path.insert in the
    # advanced library loader, since it isn't included in the hatch wheel.
    # torchvision is already available because torch installs it as a sibling.
    from BiRefNetModule.wrapper import ImagePreprocessor, half_precision
    from torchvision import transforms

    h, w = image_rgb_float.shape[:2]
    rgb_uint8 = np.clip(image_rgb_float * 255.0, 0, 255).astype(np.uint8)
    pil_image = Image.fromarray(rgb_uint8, mode="RGB")

    # Mirror the dynamic-model resolution selection from BiRefNetHandler.process().
    if handler.resolution is None:
        resolution_div_by_32 = tuple(int(int(reso) // 32 * 32) for reso in pil_image.size)
        handler.resolution = resolution_div_by_32

    preprocessor = ImagePreprocessor(resolution=tuple(handler.resolution))
    image_proc = preprocessor.proc(pil_image).unsqueeze(0).to(handler.device)
    if half_precision:
        image_proc = image_proc.half()

    with torch.no_grad():
        preds = handler.birefnet(image_proc)[-1].sigmoid().cpu()

    pred = preds[0].squeeze()
    pred_pil = transforms.ToPILImage()(pred.float())
    mask_pil = pred_pil.resize((w, h))
    return np.asarray(mask_pil, dtype=np.float32) / 255.0


def load_gvm_processor(repo_id: str, device: str):
    """Load and cache a GVMProcessor for (repo, device).

    Weights are resolved from the standard HF hub cache -- the GVM
    HuggingFaceRepoParameter dropdown gates node execution until the user has
    downloaded the model via Model Manager, so by the time this runs the repo
    is already cached and `local_files_only=True` never touches the network.
    """
    from gvm_core.wrapper import GVMProcessor
    from huggingface_hub import snapshot_download

    key = (repo_id, device)
    if key in _gvm_cache:
        return _gvm_cache[key]

    logger.debug("Resolving cached GVM weights for repo=%s", repo_id)
    weights_dir = snapshot_download(repo_id=repo_id, local_files_only=True)

    logger.info("Loading GVM processor: repo=%s device=%s", repo_id, device)
    processor = GVMProcessor(model_base=str(weights_dir), device=device)
    _gvm_cache[key] = processor
    return processor


def load_videomama_pipeline(unet_repo_id: str, base_repo_id: str, device: str):
    """Load and cache a VideoMaMa VideoInferencePipeline for (unet_repo, base_repo, device).

    Both checkpoints are resolved from the standard HF hub cache -- the VideoMaMa
    HuggingFaceRepoParameter dropdowns gate node execution until the user has
    downloaded both models via Model Manager, so by the time this runs both repos
    are already cached and `local_files_only=True` never touches the network.
    """
    from huggingface_hub import snapshot_download
    from VideoMaMaInferenceModule.inference import load_videomama_model

    key = (unet_repo_id, base_repo_id, device)
    if key in _videomama_cache:
        return _videomama_cache[key]

    logger.debug("Resolving cached VideoMaMa UNet weights for repo=%s", unet_repo_id)
    unet_dir = snapshot_download(repo_id=unet_repo_id, local_files_only=True)

    logger.debug("Resolving cached Stable Video Diffusion base weights for repo=%s", base_repo_id)
    base_dir = snapshot_download(repo_id=base_repo_id, local_files_only=True)

    logger.info("Loading VideoMaMa pipeline: device=%s", device)
    pipeline = load_videomama_model(base_model_path=str(base_dir), unet_checkpoint_path=str(unet_dir), device=device)
    _videomama_cache[key] = pipeline
    return pipeline


def resolve_frame_paths(value) -> list[Path]:
    """Resolve a Sequence, list, or raw path/pattern string into an ordered list of frame paths."""
    if isinstance(value, Sequence):
        return [Path(entry.path) for entry in value.entries]

    if isinstance(value, list):
        paths: list[Path] = []
        for item in value:
            if isinstance(item, ImageUrlArtifact):
                paths.append(Path(item.value))
            elif isinstance(item, ImageArtifact):
                raise ValueError(
                    "ImageArtifact (raw bytes) cannot be resolved to a frame file path; use ImageUrlArtifact"
                )
            else:
                paths.append(Path(item))
        return sorted(paths, key=lambda p: p.name)

    if isinstance(value, str):
        path = Path(value)
        if path.is_dir():
            frame_paths = [p for p in path.iterdir() if p.suffix.lower() in _IMAGE_SEQUENCE_EXTENSIONS]
            return sorted(frame_paths, key=lambda p: p.name)

        result = GriptapeNodes.handle_request(
            ScanSequencesRequest(
                path=value,
                policy=MissingItemPolicy.SKIP,
                no_token_behavior=NoTokenBehavior.EXPLORE_SEQUENCE,
            )
        )
        if not isinstance(result, ScanSequencesResultSuccess) or not result.has_entries:
            raise ValueError(f"Could not resolve '{value}' to a frame sequence")
        entries = [entry for sequence in result.sequences for entry in sequence.entries]
        entries.sort(key=lambda entry: entry.number)
        return [Path(entry.path) for entry in entries]

    raise ValueError(f"Unsupported sequence value type: {type(value)}")


def materialize_video_as_sequence(video_path: str, out_dir: Path) -> tuple[list[Path], float]:
    """Decode a video into a PNG frame sequence on disk, returning the frame paths and fps."""
    from gvm_core.gvm.utils.inference_utils import VideoReader

    out_dir.mkdir(parents=True, exist_ok=True)
    reader = VideoReader(video_path)
    frame_paths = []
    for idx in range(len(reader)):
        frame_path = out_dir / f"frame_{idx:05d}.png"
        reader[idx].convert("RGB").save(frame_path)
        frame_paths.append(frame_path)
    return frame_paths, float(reader.frame_rate)


def materialize_frame_directory(frame_paths: list[Path], out_dir: Path) -> Path:
    """Symlink (or copy) frame files into a fresh directory with contiguous zero-padded names.

    GVM's own ImageSequenceReader lists an entire directory and sorts by filename, so it
    needs a clean directory containing exactly (and only) the frames we want it to see,
    regardless of what directory those frames originally lived in.
    """
    import shutil

    out_dir.mkdir(parents=True, exist_ok=True)
    for idx, frame_path in enumerate(frame_paths):
        dest = out_dir / f"{idx:05d}{frame_path.suffix}"
        try:
            dest.symlink_to(frame_path.resolve())
        except OSError:
            shutil.copy2(frame_path, dest)
    return out_dir


def resolve_source_to_frames(value, tmp_dir: Path) -> tuple[list[Path], float | None]:
    """Resolve a video-or-sequence parameter value into (frame_paths, fps_or_none)."""
    from griptape.artifacts import VideoUrlArtifact

    if isinstance(value, VideoUrlArtifact):
        video_path = File(value.value).resolve()
        return materialize_video_as_sequence(video_path, tmp_dir)

    return resolve_frame_paths(value), None


def read_frame_rgb(path: Path) -> np.ndarray:
    """Read a single frame file (EXR or standard image format) into float32 [H, W, 3] RGB."""
    if path.suffix.lower() == ".exr":
        from griptape_nodes_openexr.exr.exr_io import load_exr_channels

        channels = load_exr_channels(str(path), part_index=0, channel_names=["R", "G", "B"])
        return np.stack([channels["R"], channels["G"], channels["B"]], axis=-1).astype(np.float32)

    return decode_rgb_image(path.read_bytes())


def read_mask_gray(path: Path) -> np.ndarray:
    """Read a single mask/alpha frame file (EXR or standard image format) into float32 [H, W]."""
    if path.suffix.lower() == ".exr":
        from griptape_nodes_openexr.exr.exr_io import load_exr_channels

        channels = load_exr_channels(str(path), part_index=0, channel_names=None)
        channel_name = "Y" if "Y" in channels else next(iter(channels))
        return channels[channel_name].astype(np.float32)

    return decode_alpha_image(path.read_bytes())


def write_frame_rgb(path: Path, rgb: np.ndarray) -> None:
    """Write a float32 [H, W, 3] RGB frame to an EXR or standard image file, by extension."""
    if path.suffix.lower() == ".exr":
        from griptape_nodes_openexr.exr.exr_io import write_exr_channels

        write_exr_channels(str(path), {"R": rgb[..., 0], "G": rgb[..., 1], "B": rgb[..., 2]}, pixel_type="float")
    else:
        path.write_bytes(encode_rgb_png(rgb))


def write_frame_gray(path: Path, alpha: np.ndarray) -> None:
    """Write a float32 [H, W] grayscale frame to an EXR or standard image file, by extension."""
    if path.suffix.lower() == ".exr":
        from griptape_nodes_openexr.exr.exr_io import write_exr_channels

        write_exr_channels(str(path), {"Y": alpha}, pixel_type="float")
    else:
        path.write_bytes(encode_grayscale_png(alpha))


def build_output_sequence(frame_paths: list[Path], pattern_name: str) -> Sequence:
    """Upload written frame files via ProjectFileDestination and wrap them as a Sequence output."""
    entries = []
    directory = ""
    for idx, frame_path in enumerate(frame_paths):
        dest = ProjectFileDestination.from_situation(filename=frame_path.name, situation="save_node_output")
        saved = dest.write_bytes(frame_path.read_bytes())
        directory = str(Path(saved.location).parent)
        entries.append(SequenceEntry(number=idx, padded_number=f"{idx:05d}", path=saved.location))

    return Sequence(
        entries=entries,
        first=0,
        last=len(entries) - 1 if entries else 0,
        discovered_first=0,
        discovered_last=len(entries) - 1 if entries else 0,
        padding=5,
        pattern=f"{pattern_name}.#####{frame_paths[0].suffix if frame_paths else '.exr'}",
        directory=directory,
        policy=MissingItemPolicy.SKIP,
        present_numbers=set(range(len(entries))),
    )
