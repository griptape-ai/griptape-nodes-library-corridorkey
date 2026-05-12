import io
import logging
import uuid

import numpy as np
import torch
from griptape.artifacts import ImageArtifact, ImageUrlArtifact
from griptape_nodes.exe_types.core_types import Parameter, ParameterMode
from griptape_nodes.exe_types.node_types import AsyncResult, SuccessFailureNode
from griptape_nodes.exe_types.param_components.huggingface.huggingface_repo_parameter import HuggingFaceRepoParameter
from griptape_nodes.files.file import File
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes
from PIL import Image

logger = logging.getLogger("corridorkey_library")

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


class CorridorKeyInference(SuccessFailureNode):
    """Run the CorridorKey neural keying pipeline on a single image frame.

    Takes an RGB image and an optional coarse alpha-hint mask; if no hint is
    supplied, BiRefNet is used internally to generate one. Produces a clean
    straight alpha matte, a despilled sRGB foreground, a linear premultiplied
    RGBA composite, and an optional composite-on-checkerboard preview. Supports
    both green-screen and blue-screen checkpoints with auto-download from
    HuggingFace.
    """

    # Class-level caches so repeated runs don't re-load the checkpoints or
    # re-trigger torch.compile autotuning. Keyed by (repo_id, device, img_size)
    # for the keying engine and (repo_id, device) for the BiRefNet handler.
    _engine = None
    _engine_key: tuple[str, str, int] | None = None
    _birefnet_handler = None
    _birefnet_key: tuple[str, str] | None = None

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        # CorridorKey checkpoint selection (HuggingFace).
        self._model_param = HuggingFaceRepoParameter(
            self,
            repo_ids=CORRIDORKEY_MODEL_REPO_IDS,
            parameter_name="model",
        )
        self._model_param.add_input_parameters()

        # BiRefNet variant used to auto-generate the alpha hint.
        self._birefnet_param = HuggingFaceRepoParameter(
            self,
            repo_ids=BIREFNET_MODEL_REPO_IDS,
            parameter_name="birefnet_model",
        )
        self._birefnet_param.add_input_parameters()

        self.add_parameter(
            Parameter(
                name="image",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                type="ImageUrlArtifact",
                input_types=["ImageUrlArtifact", "ImageArtifact"],
                default_value=None,
                tooltip=(
                    "RGB source frame (sRGB or linear depending on `input_is_linear`). "
                    "Any resolution; internally resized to `img_size` for inference and upsampled back."
                ),
            )
        )

        self.add_parameter(
            Parameter(
                name="alpha_hint",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                type="ImageUrlArtifact",
                input_types=["ImageUrlArtifact", "ImageArtifact"],
                default_value=None,
                tooltip=(
                    "Optional coarse alpha-hint mask as a single-channel (or grayscale) image, "
                    "0=background 1=foreground. If left empty, BiRefNet is run internally to generate the hint."
                ),
            )
        )

        self.add_parameter(
            Parameter(
                name="img_size",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                type="int",
                default_value=2048,
                tooltip=(
                    "Inference resolution (square). 2048 matches the trained resolution; "
                    "lower values are faster but less accurate."
                ),
            )
        )

        self.add_parameter(
            Parameter(
                name="input_is_linear",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                type="bool",
                default_value=False,
                tooltip=(
                    "If True, the input image is treated as linear and resized in linear before "
                    "being converted to sRGB for the model. Leave False for typical sRGB inputs."
                ),
            )
        )

        self.add_parameter(
            Parameter(
                name="fg_is_straight",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                type="bool",
                default_value=True,
                tooltip=(
                    "If True, the foreground output is straight (unpremultiplied). "
                    "If False, treated as premultiplied. Leave True for the published checkpoints."
                ),
            )
        )

        self.add_parameter(
            Parameter(
                name="despill_strength",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                type="float",
                default_value=1.0,
                tooltip="0.0-1.0 multiplier for the despill pass that removes screen-color contamination from the foreground.",
            )
        )

        self.add_parameter(
            Parameter(
                name="auto_despeckle",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                type="bool",
                default_value=True,
                tooltip="If True, runs a morphological cleanup that removes small disconnected islands from the predicted matte.",
            )
        )

        self.add_parameter(
            Parameter(
                name="despeckle_size",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                type="int",
                default_value=400,
                tooltip="Minimum connected-component size (in pixels) preserved when `auto_despeckle` is True.",
            )
        )

        self.add_parameter(
            Parameter(
                name="refiner_scale",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                type="float",
                default_value=1.0,
                tooltip=(
                    "Multiplier on the refiner head's delta contribution. "
                    "1.0 = default, >1.0 sharpens edges, 0.0 disables the refiner."
                ),
            )
        )

        self.add_parameter(
            Parameter(
                name="generate_comp",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                type="bool",
                default_value=True,
                tooltip=(
                    "If True, also produces a composite-on-gray-checkerboard preview "
                    "(useful for quickly sanity-checking the matte)."
                ),
            )
        )

        self.add_parameter(
            Parameter(
                name="alpha",
                allowed_modes={ParameterMode.OUTPUT},
                output_type="ImageUrlArtifact",
                default_value=None,
                tooltip="Single-channel straight alpha matte (float 0-1) encoded as a grayscale PNG.",
            )
        )

        self.add_parameter(
            Parameter(
                name="foreground",
                allowed_modes={ParameterMode.OUTPUT},
                output_type="ImageUrlArtifact",
                default_value=None,
                tooltip="Despilled sRGB foreground, straight (unpremultiplied), encoded as PNG.",
            )
        )

        self.add_parameter(
            Parameter(
                name="composite",
                allowed_modes={ParameterMode.OUTPUT},
                output_type="ImageUrlArtifact",
                default_value=None,
                tooltip="sRGB composite of the foreground over a gray checkerboard (PNG). None when `generate_comp` is False.",
            )
        )

        self.add_parameter(
            Parameter(
                name="rgba",
                allowed_modes={ParameterMode.OUTPUT},
                output_type="ImageUrlArtifact",
                default_value=None,
                tooltip=(
                    "Linear premultiplied RGBA (foreground * alpha, plus alpha channel) encoded as an "
                    "8-bit PNG with alpha for use as a drop-in matte."
                ),
            )
        )

        self._create_status_parameters()

    def validate_before_node_run(self) -> list[Exception] | None:
        errors: list[Exception] = []

        model_errors = self._model_param.validate_before_node_run()
        if model_errors:
            errors.extend(model_errors)

        # The BiRefNet model is only required when no alpha hint is supplied.
        # We still validate it so users get a clear "Model Download Required"
        # warning if they leave the alpha hint empty without downloading BiRefNet.
        if not self.parameter_values.get("alpha_hint"):
            birefnet_errors = self._birefnet_param.validate_before_node_run()
            if birefnet_errors:
                errors.extend(birefnet_errors)

        if not self.parameter_values.get("image"):
            errors.append(ValueError("image is required"))

        return errors if errors else None

    def _get_device(self) -> str:
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _artifact_to_bytes(self, artifact: ImageArtifact | ImageUrlArtifact) -> bytes:
        if isinstance(artifact, ImageUrlArtifact):
            return File(artifact.value).read_bytes()
        # ImageArtifact stores raw bytes in .value
        return artifact.value

    def _decode_rgb_image(self, image_bytes: bytes) -> np.ndarray:
        """Decode image bytes into a float32 [H, W, 3] sRGB array in [0, 1]."""
        pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        arr = np.asarray(pil, dtype=np.float32) / 255.0
        return arr

    def _decode_alpha_image(self, image_bytes: bytes) -> np.ndarray:
        """Decode an alpha-hint image into a float32 [H, W] array in [0, 1]."""
        pil = Image.open(io.BytesIO(image_bytes)).convert("L")
        arr = np.asarray(pil, dtype=np.float32) / 255.0
        return arr

    def _encode_grayscale_png(self, alpha: np.ndarray) -> bytes:
        """Encode a single-channel float32 [H, W] or [H, W, 1] alpha matte as 8-bit PNG."""
        if alpha.ndim == 3:
            alpha = alpha[..., 0]
        arr = np.clip(alpha * 255.0, 0, 255).astype(np.uint8)
        pil = Image.fromarray(arr, mode="L")
        buf = io.BytesIO()
        pil.save(buf, format="PNG")
        return buf.getvalue()

    def _encode_rgb_png(self, rgb: np.ndarray) -> bytes:
        """Encode a float32 [H, W, 3] sRGB image as 8-bit PNG."""
        arr = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
        pil = Image.fromarray(arr, mode="RGB")
        buf = io.BytesIO()
        pil.save(buf, format="PNG")
        return buf.getvalue()

    def _encode_rgba_png(self, rgba: np.ndarray) -> bytes:
        """Encode a float32 [H, W, 4] linear premultiplied RGBA image as 8-bit PNG."""
        arr = np.clip(rgba * 255.0, 0, 255).astype(np.uint8)
        pil = Image.fromarray(arr, mode="RGBA")
        buf = io.BytesIO()
        pil.save(buf, format="PNG")
        return buf.getvalue()

    def _save_image_artifact(self, image_bytes: bytes, suffix: str) -> ImageUrlArtifact:
        filename = f"corridorkey_{suffix}_{uuid.uuid4().hex[:8]}.png"
        url = GriptapeNodes.StaticFilesManager().save_static_file(image_bytes, filename)
        return ImageUrlArtifact(url)

    def _load_engine(self, model_repo_id: str, device: str, img_size: int):
        """Load and cache the CorridorKey engine for (repo, device, img_size)."""
        # Deferred imports: these resolve only after the advanced library has
        # pip-installed CorridorKeyModule and added the submodule root to sys.path.
        import os

        from CorridorKeyModule import backend as ck_backend
        from CorridorKeyModule.backend import create_engine

        # Upstream bug: backend.py defines CHECKPOINT_DIR twice (line 22 absolute,
        # line 74 relative). The second definition wins at import time and points
        # at "CorridorKeyModule/checkpoints" relative to CWD, so checkpoint copies
        # fail with FileNotFoundError. Restore the absolute path.
        ck_backend.CHECKPOINT_DIR = os.path.join(os.path.dirname(os.path.abspath(ck_backend.__file__)), "checkpoints")
        os.makedirs(ck_backend.CHECKPOINT_DIR, exist_ok=True)

        key = (model_repo_id, device, img_size)
        if CorridorKeyInference._engine is not None and CorridorKeyInference._engine_key == key:
            return CorridorKeyInference._engine

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
        CorridorKeyInference._engine = engine
        CorridorKeyInference._engine_key = key
        return engine

    def _load_birefnet(self, birefnet_repo_id: str, device: str):
        """Load and cache a BiRefNetHandler for (repo, device)."""
        # Deferred import: BiRefNetModule is exposed via sys.path.insert in the
        # advanced library loader, since it isn't included in the hatch wheel.
        from BiRefNetModule.wrapper import BiRefNetHandler

        key = (birefnet_repo_id, device)
        if CorridorKeyInference._birefnet_handler is not None and CorridorKeyInference._birefnet_key == key:
            return CorridorKeyInference._birefnet_handler

        usage = BIREFNET_USAGE_BY_REPO[birefnet_repo_id]
        logger.info("Loading BiRefNet handler: repo=%s usage=%s device=%s", birefnet_repo_id, usage, device)
        handler = BiRefNetHandler(device=device, usage=usage)
        CorridorKeyInference._birefnet_handler = handler
        CorridorKeyInference._birefnet_key = key
        return handler

    def _run_birefnet(self, handler, image_rgb_float: np.ndarray) -> np.ndarray:
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
        alpha_np = np.asarray(mask_pil, dtype=np.float32) / 255.0
        return alpha_np

    def process(self) -> AsyncResult[None]:
        yield lambda: self._run_inference()

    def _run_inference(self) -> None:
        try:
            self._do_inference()
        except Exception as exc:
            logger.exception("CorridorKey inference failed")
            self._set_status_results(was_successful=False, result_details=f"CorridorKey inference failed: {exc}")
            self._handle_failure_exception(exc)
            return
        self._set_status_results(was_successful=True, result_details="CorridorKey inference complete")

    def _do_inference(self) -> None:
        # Read inputs.
        image_artifact = self.parameter_values.get("image")
        if not isinstance(image_artifact, (ImageArtifact, ImageUrlArtifact)):
            raise ValueError("image is required")

        alpha_hint_artifact = self.parameter_values.get("alpha_hint")

        img_size: int = int(self.parameter_values.get("img_size") or 2048)
        input_is_linear: bool = bool(self.parameter_values.get("input_is_linear"))
        fg_is_straight: bool = bool(self.parameter_values.get("fg_is_straight", True))
        despill_strength: float = float(self.parameter_values.get("despill_strength") or 1.0)
        auto_despeckle: bool = bool(self.parameter_values.get("auto_despeckle", True))
        despeckle_size: int = int(self.parameter_values.get("despeckle_size") or 400)
        refiner_scale: float = float(self.parameter_values.get("refiner_scale") or 1.0)
        generate_comp: bool = bool(self.parameter_values.get("generate_comp", True))

        device = self._get_device()
        logger.info("CorridorKey inference: device=%s", device)

        model_repo_id, _ = self._model_param.get_repo_revision()
        screen_color = SCREEN_COLOR_BY_REPO[model_repo_id]
        # screen_channel: 1=green (RGB index), 2=blue.
        screen_channel = 1 if screen_color == "green" else 2

        # Decode the RGB source image to float32 [H, W, 3].
        image_bytes = self._artifact_to_bytes(image_artifact)
        image_np = self._decode_rgb_image(image_bytes)

        # Either decode the supplied alpha hint or generate one with BiRefNet.
        if isinstance(alpha_hint_artifact, (ImageArtifact, ImageUrlArtifact)):
            alpha_hint_bytes = self._artifact_to_bytes(alpha_hint_artifact)
            alpha_hint_np = self._decode_alpha_image(alpha_hint_bytes)
            if alpha_hint_np.shape[:2] != image_np.shape[:2]:
                # Resize the hint to match the source frame.
                hint_pil = Image.fromarray((alpha_hint_np * 255.0).astype(np.uint8), mode="L")
                hint_pil = hint_pil.resize((image_np.shape[1], image_np.shape[0]))
                alpha_hint_np = np.asarray(hint_pil, dtype=np.float32) / 255.0
        else:
            birefnet_repo_id, _ = self._birefnet_param.get_repo_revision()
            handler = self._load_birefnet(birefnet_repo_id, device)
            alpha_hint_np = self._run_birefnet(handler, image_np)

        # Load the keying engine and run inference.
        engine = self._load_engine(model_repo_id, device, img_size)

        result = engine.process_frame(
            image_np,
            alpha_hint_np,
            refiner_scale=refiner_scale,
            input_is_linear=input_is_linear,
            fg_is_straight=fg_is_straight,
            despill_strength=despill_strength,
            auto_despeckle=auto_despeckle,
            despeckle_size=despeckle_size,
            generate_comp=generate_comp,
            post_process_on_gpu=(device != "cpu"),
            screen_channel=screen_channel,
        )

        # process_frame returns a single dict for batch size 1.
        if isinstance(result, list):
            result = result[0]

        alpha_out: np.ndarray = result["alpha"]
        fg_out: np.ndarray = result["fg"]
        rgba_out: np.ndarray = result["processed"]

        self.parameter_output_values["alpha"] = self._save_image_artifact(
            self._encode_grayscale_png(alpha_out), "alpha"
        )
        self.parameter_output_values["foreground"] = self._save_image_artifact(
            self._encode_rgb_png(fg_out), "foreground"
        )
        self.parameter_output_values["rgba"] = self._save_image_artifact(self._encode_rgba_png(rgba_out), "rgba")

        if generate_comp and "comp" in result and result["comp"] is not None:
            comp_out: np.ndarray = result["comp"]
            self.parameter_output_values["composite"] = self._save_image_artifact(
                self._encode_rgb_png(comp_out), "composite"
            )
        else:
            self.parameter_output_values["composite"] = None
