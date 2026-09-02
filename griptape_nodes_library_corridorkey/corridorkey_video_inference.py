import logging
import tempfile
from pathlib import Path

import numpy as np
from griptape_nodes.exe_types.core_types import Parameter, ParameterMessage, ParameterMode
from griptape_nodes.exe_types.node_types import AsyncResult, SuccessFailureNode
from griptape_nodes.exe_types.param_components.huggingface.huggingface_repo_parameter import HuggingFaceRepoParameter
from griptape_nodes.traits.file_system_picker import FileSystemPicker
from griptape_nodes.traits.options import Options
from PIL import Image

from griptape_nodes_library_corridorkey import corridorkey_colorspace as cc
from griptape_nodes_library_corridorkey import corridorkey_common as ck

logger = logging.getLogger("corridorkey_library")

_SEQUENCE_INPUT_TYPES = ["VideoUrlArtifact", "Sequence", "list", "str"]

# Parameters only relevant for a given hint_source. Each HuggingFaceRepoParameter
# contributes both its dropdown and a "..._download" button parameter.
_HINT_SOURCE_PARAMETERS = {
    "birefnet": ["birefnet_model", "birefnet_model_download"],
    "gvm": [
        "gvm_model",
        "gvm_model_download",
        "gvm_num_frames_per_batch",
        "gvm_num_overlap_frames",
        "gvm_denoise_steps",
    ],
    "videomama": [
        "mask_hint",
        "videomama_unet_model",
        "videomama_unet_model_download",
        "videomama_base_model",
        "videomama_base_model_download",
        "videomama_chunk_size",
    ],
}

# Output parameters only relevant for a given output_format.
_OUTPUT_FORMAT_PARAMETERS = {
    "video": ["alpha", "foreground", "composite"],
    "exr_sequence": ["alpha_sequence", "foreground_sequence", "composite_sequence"],
}


class CorridorKeyVideoInference(SuccessFailureNode):
    """Run the CorridorKey neural keying pipeline across every frame of a video or image sequence.

    Generates a per-frame alpha hint via BiRefNet (default, works on any GPU),
    GVM (temporally-consistent hints across the clip, but requires ~80GB VRAM on
    a single GPU -- multi-GPU setups do not pool VRAM for this, and a large
    one-time HuggingFace download), or VideoMaMa (temporally-consistent hints
    guided by a user-supplied rough per-frame mask hint, also a large one-time
    HuggingFace download). Runs the same despill/refine pass the single-image
    node uses on each frame. Outputs are either mp4 videos (alpha/foreground/
    composite) or an EXR image sequence per output (which, unlike mp4, can also
    carry a genuine alpha channel).
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        self.add_node_element(
            ParameterMessage(
                name="usage_tip",
                variant="tip",
                title="Usage Tip",
                value=(
                    "`hint_source=birefnet` is the safe default and runs on any GPU. Only switch to `gvm` or "
                    "`videomama` if you need temporally-consistent hints across the clip and have a GPU with "
                    "enough VRAM -- both require a large one-time HuggingFace download. Use `max_frames` to "
                    "preview settings on a short clip before running the full video. Leave `color_mode` as "
                    "'basic' unless you need an OCIO display-view transform on `foreground`, in which case "
                    "set it to 'ocio' and connect `color_params`."
                ),
            )
        )

        self.add_parameter(
            Parameter(
                name="video",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                type="str",
                input_types=_SEQUENCE_INPUT_TYPES,
                traits={FileSystemPicker(allow_files=True, allow_directories=True, allow_sequences=True)},
                tooltip=(
                    "Source video clip or image sequence to key. Accepts a video file, a Sequence "
                    "(e.g. from Scan Sequence), a sequence pattern (frame.####.exr), a directory, "
                    "or a list of frame paths/ImageUrlArtifacts."
                ),
            )
        )

        # CorridorKey checkpoint selection (HuggingFace).
        self._model_param = HuggingFaceRepoParameter(
            self,
            repo_ids=ck.CORRIDORKEY_MODEL_REPO_IDS,
            parameter_name="model",
        )
        self._model_param.add_input_parameters()

        self.add_parameter(
            Parameter(
                name="hint_source",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                type="str",
                default_value="birefnet",
                traits={Options(choices=["birefnet", "gvm", "videomama"])},
                tooltip=(
                    "How the per-frame alpha hint is generated. 'birefnet' runs BiRefNet independently "
                    "on each frame and works on any GPU. 'gvm' runs the GVM video model for temporally-"
                    "consistent hints across the whole clip -- requires ~80GB VRAM on a SINGLE GPU (multi-GPU "
                    "setups do not pool VRAM here) and downloads a large checkpoint from HuggingFace on first "
                    "use. 'videomama' also generates temporally-consistent hints, guided by a rough per-frame "
                    "mask you supply via `mask_hint`; also downloads large checkpoints on first use."
                ),
            )
        )

        # BiRefNet variant used to auto-generate the alpha hint. Only used when hint_source == "birefnet".
        self._birefnet_param = HuggingFaceRepoParameter(
            self,
            repo_ids=ck.BIREFNET_MODEL_REPO_IDS,
            parameter_name="birefnet_model",
        )
        self._birefnet_param.add_input_parameters()

        # GVM checkpoint (HuggingFace). Only used when hint_source == "gvm".
        self._gvm_param = HuggingFaceRepoParameter(
            self,
            repo_ids=[ck.GVM_MODEL_REPO_ID],
            parameter_name="gvm_model",
        )
        self._gvm_param.add_input_parameters()

        # VideoMaMa checkpoints (HuggingFace). Only used when hint_source == "videomama".
        self._videomama_unet_param = HuggingFaceRepoParameter(
            self,
            repo_ids=[ck.VIDEOMAMA_UNET_REPO_ID],
            parameter_name="videomama_unet_model",
        )
        self._videomama_unet_param.add_input_parameters()

        self._videomama_base_param = HuggingFaceRepoParameter(
            self,
            repo_ids=[ck.VIDEOMAMA_BASE_REPO_ID],
            parameter_name="videomama_base_model",
        )
        self._videomama_base_param.add_input_parameters()

        self.add_parameter(
            Parameter(
                name="mask_hint",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                type="str",
                input_types=_SEQUENCE_INPUT_TYPES,
                traits={FileSystemPicker(allow_files=True, allow_directories=True, allow_sequences=True)},
                default_value=None,
                tooltip=(
                    "VideoMaMa only, required when hint_source is 'videomama': a rough per-frame foreground "
                    "mask (hand-drawn or AI-generated), one mask per video frame, as a video or image sequence. "
                    "Thresholded internally to binary."
                ),
            )
        )

        self.add_parameter(
            Parameter(
                name="gvm_num_frames_per_batch",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                type="int",
                default_value=8,
                tooltip="GVM only: number of frames per inference batch when generating temporal alpha hints.",
            )
        )

        self.add_parameter(
            Parameter(
                name="gvm_num_overlap_frames",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                type="int",
                default_value=1,
                tooltip="GVM only: number of overlapping frames between consecutive batches, for temporal smoothness.",
            )
        )

        self.add_parameter(
            Parameter(
                name="gvm_denoise_steps",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                type="int",
                default_value=1,
                tooltip="GVM only: number of diffusion denoising steps used to generate each alpha-hint batch.",
            )
        )

        self.add_parameter(
            Parameter(
                name="videomama_chunk_size",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                type="int",
                default_value=24,
                tooltip="VideoMaMa only: number of frames processed per diffusion call, to bound VRAM usage.",
            )
        )

        self.add_parameter(
            Parameter(
                name="output_frame_rate",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                type="float",
                default_value=24.0,
                tooltip=(
                    "Frame rate used for video outputs when the source is an image sequence (which has no "
                    "inherent frame rate). Ignored when the source is a video file -- its own frame rate is used."
                ),
            )
        )

        self.add_parameter(
            Parameter(
                name="max_frames",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                type="int",
                default_value=0,
                tooltip="Cap on the number of frames processed, useful for previewing a long clip. 0 = process all frames.",
            )
        )

        self.add_parameter(
            Parameter(
                name="batch_size",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                type="int",
                default_value=4,
                tooltip="Number of frames sent through the CorridorKey engine per inference call.",
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
                    "If True, input frames are treated as linear and resized in linear before "
                    "being converted to sRGB for the model. Leave False for typical sRGB video."
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
                    "If True, `foreground` is straight (unpremultiplied). If False, `foreground` is "
                    "alpha-premultiplied. Leave True for the published checkpoints. Does not affect "
                    "`composite`, which is always correctly composited regardless of this setting."
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

        default_color_mode = (
            cc.COLOR_MODE_OCIO if cc.find_colorspace_transform_request_type() is not None else cc.COLOR_MODE_BASIC
        )

        self.add_parameter(
            Parameter(
                name="color_mode",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                type="str",
                default_value=default_color_mode,
                traits={Options(choices=[cc.COLOR_MODE_BASIC, cc.COLOR_MODE_OCIO])},
                tooltip=(
                    "Colour management used to convert the `foreground` output's internal linear data "
                    "to display sRGB. 'basic' uses the local sRGB transfer function (no dependencies). "
                    "'ocio' uses a connected `color_params` OCIO display-view transform, requiring the "
                    "OpenColorIO library to be loaded."
                ),
            )
        )

        self.add_parameter(
            Parameter(
                name="color_params",
                allowed_modes={ParameterMode.INPUT},
                type="OCIOColorParamsArtifact",
                input_types=["OCIOColorParamsArtifact"],
                default_value=None,
                tooltip="OCIO colour parameters (source colorspace, display, view). Required when `color_mode` is 'ocio'.",
                ui_options={"hide": default_color_mode != cc.COLOR_MODE_OCIO},
            )
        )

        self.add_parameter(
            Parameter(
                name="output_format",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                type="str",
                default_value="video",
                traits={Options(choices=["video", "exr_sequence"])},
                tooltip=(
                    "'video' writes alpha/foreground/composite as mp4 (no alpha channel possible). "
                    "'exr_sequence' writes each as a float32 EXR image sequence instead, which can carry "
                    "a genuine alpha channel."
                ),
            )
        )

        self.add_parameter(
            Parameter(
                name="alpha",
                allowed_modes={ParameterMode.OUTPUT},
                output_type="VideoUrlArtifact",
                default_value=None,
                tooltip=(
                    "Single-channel straight alpha matte video (grayscale mp4), including `auto_despeckle` "
                    "cleanup when enabled. None when `output_format` is 'exr_sequence'."
                ),
            )
        )

        self.add_parameter(
            Parameter(
                name="foreground",
                allowed_modes={ParameterMode.OUTPUT},
                output_type="VideoUrlArtifact",
                default_value=None,
                tooltip="Despilled sRGB foreground video, straight (unpremultiplied). None when `output_format` is 'exr_sequence'.",
            )
        )

        self.add_parameter(
            Parameter(
                name="composite",
                allowed_modes={ParameterMode.OUTPUT},
                output_type="VideoUrlArtifact",
                default_value=None,
                tooltip="sRGB composite of the foreground over a gray checkerboard (mp4). None when `generate_comp` is False or `output_format` is 'exr_sequence'.",
            )
        )

        self.add_parameter(
            Parameter(
                name="alpha_sequence",
                allowed_modes={ParameterMode.OUTPUT},
                output_type="Sequence",
                default_value=None,
                tooltip=(
                    "Single-channel straight alpha matte as an EXR sequence, including `auto_despeckle` "
                    "cleanup when enabled. None when `output_format` is 'video'."
                ),
            )
        )

        self.add_parameter(
            Parameter(
                name="foreground_sequence",
                allowed_modes={ParameterMode.OUTPUT},
                output_type="Sequence",
                default_value=None,
                tooltip="Despilled sRGB foreground as an EXR sequence. None when `output_format` is 'video'.",
            )
        )

        self.add_parameter(
            Parameter(
                name="composite_sequence",
                allowed_modes={ParameterMode.OUTPUT},
                output_type="Sequence",
                default_value=None,
                tooltip="Composite-on-checkerboard as an EXR sequence. None when `output_format` is 'video' or `generate_comp` is False.",
            )
        )

        self._create_status_parameters()

        # after_value_set() is skipped for values set during construction (including
        # defaults), so the initial visibility state has to be applied explicitly here.
        self._update_hint_source_visibility("birefnet")
        self._update_output_format_visibility("video")

    def after_value_set(self, parameter: Parameter, value) -> None:
        super().after_value_set(parameter, value)
        if parameter.name == "hint_source":
            self._update_hint_source_visibility(value)
        elif parameter.name == "output_format":
            self._update_output_format_visibility(value)
        elif parameter.name == "color_mode":
            if value == cc.COLOR_MODE_OCIO:
                self.show_parameter_by_name("color_params")
            else:
                self.hide_parameter_by_name("color_params")

    def _update_hint_source_visibility(self, hint_source: str) -> None:
        for source, names in _HINT_SOURCE_PARAMETERS.items():
            if source == hint_source:
                self.show_parameter_by_name(names)
            else:
                self.hide_parameter_by_name(names)

    def _update_output_format_visibility(self, output_format: str) -> None:
        for fmt, names in _OUTPUT_FORMAT_PARAMETERS.items():
            if fmt == output_format:
                self.show_parameter_by_name(names)
            else:
                self.hide_parameter_by_name(names)

    def validate_before_node_run(self) -> list[Exception] | None:
        errors: list[Exception] = []

        model_errors = self._model_param.validate_before_node_run()
        if model_errors:
            errors.extend(model_errors)

        hint_source = self.parameter_values.get("hint_source", "birefnet")
        if hint_source == "birefnet":
            birefnet_errors = self._birefnet_param.validate_before_node_run()
            if birefnet_errors:
                errors.extend(birefnet_errors)
        elif hint_source == "gvm":
            gvm_errors = self._gvm_param.validate_before_node_run()
            if gvm_errors:
                errors.extend(gvm_errors)
        elif hint_source == "videomama":
            if not self.parameter_values.get("mask_hint"):
                errors.append(ValueError("mask_hint is required when hint_source is 'videomama'"))
            unet_errors = self._videomama_unet_param.validate_before_node_run()
            if unet_errors:
                errors.extend(unet_errors)
            base_errors = self._videomama_base_param.validate_before_node_run()
            if base_errors:
                errors.extend(base_errors)

        if not self.parameter_values.get("video"):
            errors.append(ValueError("video is required"))

        return errors if errors else None

    def process(self) -> AsyncResult[None]:
        yield lambda: self._run_inference()

    def _run_inference(self) -> None:
        try:
            self._do_inference()
        except Exception as exc:
            logger.exception("CorridorKey video inference failed")
            self._set_status_results(was_successful=False, result_details=f"CorridorKey video inference failed: {exc}")
            self._handle_failure_exception(exc)
            return
        self._set_status_results(was_successful=True, result_details="CorridorKey video inference complete")

    def _load_alpha_hint_frames_gvm(self, frame_paths: list[Path], tmp_path: Path) -> list[np.ndarray]:
        """Precompute the whole clip's alpha hints with GVM (needs full-clip context for temporal consistency)."""
        device = self.execution_device
        gvm_repo_id, _ = self._gvm_param.get_repo_revision()
        processor = ck.load_gvm_processor(gvm_repo_id, device)

        gvm_num_frames_per_batch = int(self.parameter_values.get("gvm_num_frames_per_batch") or 8)
        gvm_num_overlap_frames = int(self.parameter_values.get("gvm_num_overlap_frames") or 1)
        gvm_denoise_steps = int(self.parameter_values.get("gvm_denoise_steps") or 1)

        gvm_input_dir = ck.materialize_frame_directory(frame_paths, tmp_path / "gvm_input")
        gvm_output_dir = str(tmp_path / "gvm_hints")

        logger.info("Generating GVM alpha hints for %d frames", len(frame_paths))
        processor.process_sequence(
            input_path=str(gvm_input_dir),
            output_dir=gvm_output_dir,
            num_frames_per_batch=gvm_num_frames_per_batch,
            num_overlap_frames=gvm_num_overlap_frames,
            denoise_steps=gvm_denoise_steps,
            mode="matte",
            write_video=False,
        )

        alpha_seq_dir = Path(gvm_output_dir) / gvm_input_dir.name / "alpha_seq"
        hint_files = sorted(alpha_seq_dir.glob("*"))
        if not hint_files:
            raise RuntimeError(f"GVM did not produce any alpha-hint frames in {alpha_seq_dir}")

        return [ck.decode_alpha_image(hint_file.read_bytes()) for hint_file in hint_files]

    def _load_alpha_hint_frames_videomama(
        self, frame_paths: list[Path], mask_value, tmp_path: Path
    ) -> tuple[list[Path], list[np.ndarray]]:
        """Precompute the whole clip's alpha hints with VideoMaMa, using a user-supplied mask hint.

        Returns the (possibly truncated) frame_paths list alongside the hint frames, since
        VideoMaMa's mask_hint may have a different frame count than `video`.
        """
        from VideoMaMaInferenceModule.inference import run_inference as videomama_run_inference

        mask_paths, _ = ck.resolve_source_to_frames(mask_value, tmp_path / "mask_frames")
        frame_count = min(len(frame_paths), len(mask_paths))
        if len(frame_paths) != len(mask_paths):
            logger.warning(
                "video has %d frames but mask_hint has %d; truncating to %d",
                len(frame_paths),
                len(mask_paths),
                frame_count,
            )
        frame_paths = frame_paths[:frame_count]
        mask_paths = mask_paths[:frame_count]

        chunk_size = int(self.parameter_values.get("videomama_chunk_size") or 24)
        device = self.execution_device
        unet_repo_id, _ = self._videomama_unet_param.get_repo_revision()
        base_repo_id, _ = self._videomama_base_param.get_repo_revision()
        pipeline = ck.load_videomama_pipeline(unet_repo_id, base_repo_id, device)

        # Deferred: corridorkey_common (imported at module load, above) sets
        # OPENCV_IO_ENABLE_OPENEXR before cv2 is ever imported, which only works
        # if cv2 isn't already imported at module level -- see corridorkey_common.py.
        import cv2

        input_frames = [np.clip(ck.read_frame_rgb(p) * 255.0, 0, 255).astype(np.uint8) for p in frame_paths]
        mask_frames = []
        for mask_path in mask_paths:
            mask_uint8 = np.clip(ck.read_mask_gray(mask_path) * 255.0, 0, 255).astype(np.uint8)
            _, mask_binary = cv2.threshold(mask_uint8, 10, 255, cv2.THRESH_BINARY)
            mask_frames.append(mask_binary)

        logger.info("Generating VideoMaMa alpha hints for %d frames", frame_count)
        hint_rgb_frames: list[np.ndarray] = []
        for chunk in videomama_run_inference(pipeline, input_frames, mask_frames, chunk_size=chunk_size):
            hint_rgb_frames.extend(chunk)

        hint_frames = [
            np.asarray(Image.fromarray(frame).convert("L"), dtype=np.float32) / 255.0 for frame in hint_rgb_frames
        ]
        return frame_paths, hint_frames

    def _do_inference(self) -> None:
        from gvm_core.gvm.utils.inference_utils import VideoWriter

        video_value = self.parameter_values.get("video")
        if not video_value:
            raise ValueError("video is required")

        img_size: int = int(self.parameter_values.get("img_size") or 2048)
        input_is_linear: bool = bool(self.parameter_values.get("input_is_linear"))
        fg_is_straight: bool = bool(self.parameter_values.get("fg_is_straight", True))
        despill_strength: float = float(self.parameter_values.get("despill_strength") or 1.0)
        auto_despeckle: bool = bool(self.parameter_values.get("auto_despeckle", True))
        despeckle_size: int = int(self.parameter_values.get("despeckle_size") or 400)
        refiner_scale: float = float(self.parameter_values.get("refiner_scale") or 1.0)
        generate_comp: bool = bool(self.parameter_values.get("generate_comp", True))
        color_mode = str(self.parameter_values.get("color_mode") or cc.COLOR_MODE_BASIC)
        color_params = self.parameter_values.get("color_params") if color_mode == cc.COLOR_MODE_OCIO else None
        if color_mode == cc.COLOR_MODE_OCIO and color_params is None:
            raise ValueError("color_mode is 'ocio' but no color_params input is connected.")
        hint_source: str = self.parameter_values.get("hint_source", "birefnet")
        output_format: str = self.parameter_values.get("output_format", "video")
        output_frame_rate: float = float(self.parameter_values.get("output_frame_rate") or 24.0)
        batch_size: int = max(1, int(self.parameter_values.get("batch_size") or 4))
        max_frames_raw: int = int(self.parameter_values.get("max_frames") or 0)
        max_frames: int | None = max_frames_raw if max_frames_raw > 0 else None

        device = self.execution_device
        logger.info("CorridorKey video inference: device=%s hint_source=%s", device, hint_source)
        if hint_source in ("gvm", "videomama") and device != "cuda":
            logger.warning(
                "hint_source=%s is only tested on CUDA GPUs with ~80GB VRAM; device=%s may fail deep "
                "inside vendored GVM/VideoMaMa code with a cryptic error, or may simply be too slow/memory-"
                "constrained to complete.",
                hint_source,
                device,
            )

        model_repo_id, _ = self._model_param.get_repo_revision()
        screen_color = ck.SCREEN_COLOR_BY_REPO[model_repo_id]
        # screen_channel: 1=green (RGB index), 2=blue.
        screen_channel = 1 if screen_color == "green" else 2
        engine = ck.load_engine(model_repo_id, device, img_size)

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            frame_paths, detected_fps = ck.resolve_source_to_frames(video_value, tmp_path / "video_frames")
            if max_frames is not None:
                frame_paths = frame_paths[:max_frames]
            if not frame_paths:
                raise ValueError("video contains no frames")
            frame_rate = detected_fps if detected_fps is not None else output_frame_rate

            birefnet_handler = None
            gvm_hint_frames: list[np.ndarray] | None = None
            videomama_hint_frames: list[np.ndarray] | None = None

            if hint_source == "gvm":
                gvm_hint_frames = self._load_alpha_hint_frames_gvm(frame_paths, tmp_path)
                if len(gvm_hint_frames) != len(frame_paths):
                    raise RuntimeError(
                        f"GVM produced {len(gvm_hint_frames)} alpha-hint frames but the video has {len(frame_paths)} frames"
                    )
            elif hint_source == "videomama":
                mask_value = self.parameter_values.get("mask_hint")
                if not mask_value:
                    raise ValueError("mask_hint is required when hint_source is 'videomama'")
                frame_paths, videomama_hint_frames = self._load_alpha_hint_frames_videomama(
                    frame_paths, mask_value, tmp_path
                )
            else:
                birefnet_repo_id, _ = self._birefnet_param.get_repo_revision()
                birefnet_handler = ck.load_birefnet(birefnet_repo_id, device)

            frame_count = len(frame_paths)

            video_writers = None
            video_paths: dict[str, Path] = {}
            sequence_dirs: dict[str, Path] = {}
            sequence_frame_paths: dict[str, list[Path]] = {}

            if output_format == "video":
                video_paths["alpha"] = tmp_path / "alpha.mp4"
                video_paths["foreground"] = tmp_path / "foreground.mp4"
                if generate_comp:
                    video_paths["composite"] = tmp_path / "composite.mp4"
                video_writers = {
                    name: VideoWriter(str(path), frame_rate=frame_rate) for name, path in video_paths.items()
                }
            else:
                for name in ("alpha", "foreground", *(["composite"] if generate_comp else [])):
                    sequence_dirs[name] = tmp_path / f"{name}_seq"
                    sequence_dirs[name].mkdir(parents=True, exist_ok=True)
                    sequence_frame_paths[name] = []

            try:
                for batch_start in range(0, frame_count, batch_size):
                    batch_end = min(batch_start + batch_size, frame_count)

                    batch_images = []
                    batch_hints = []
                    for idx in range(batch_start, batch_end):
                        frame_rgb = ck.read_frame_rgb(frame_paths[idx])
                        if hint_source == "gvm":
                            assert gvm_hint_frames is not None
                            hint = ck.resize_alpha_to_shape(
                                gvm_hint_frames[idx], frame_rgb.shape[0], frame_rgb.shape[1]
                            )
                        elif hint_source == "videomama":
                            assert videomama_hint_frames is not None
                            hint = ck.resize_alpha_to_shape(
                                videomama_hint_frames[idx], frame_rgb.shape[0], frame_rgb.shape[1]
                            )
                        else:
                            hint = ck.run_birefnet(birefnet_handler, frame_rgb)
                        batch_images.append(frame_rgb)
                        batch_hints.append(hint)

                    images_np = np.stack(batch_images, axis=0)
                    hints_np = np.stack(batch_hints, axis=0)

                    result = engine.process_frame(
                        images_np,
                        hints_np,
                        refiner_scale=refiner_scale,
                        input_is_linear=input_is_linear,
                        # CorridorKeyModule's fg buffer is always straight in practice regardless of
                        # this flag (its own comment: "though our pipeline forces straight") -- pass
                        # True unconditionally so `comp`'s compositing formula is always correct. Our
                        # own `fg_is_straight` parameter is instead applied to `foreground` below.
                        fg_is_straight=True,
                        despill_strength=despill_strength,
                        auto_despeckle=auto_despeckle,
                        despeckle_size=despeckle_size,
                        generate_comp=generate_comp,
                        post_process_on_gpu=(device != "cpu"),
                        screen_channel=screen_channel,
                    )
                    if isinstance(result, dict):
                        result = [result]

                    # Despilled/straight/sRGB, not result["fg"] (the model's raw undespilled
                    # prediction) -- matches CorridorKeyInference's foreground/rgba convergence
                    # fix and the tooltip's "despilled" claim (see GitHub issue #2).
                    processed_stack = np.stack([r["processed"] for r in result], axis=0)
                    # processed_stack's alpha channel is the despeckled matte (when auto_despeckle is
                    # True) -- result["alpha"] is the pre-despeckle raw prediction and was previously
                    # used here, meaning auto_despeckle/despeckle_size had no effect on this output.
                    alpha_stack = processed_stack[..., 3]
                    fg_stack, _ = cc.build_foreground_and_rgba(processed_stack, color_params, fg_is_straight)

                    comp_stack = None
                    if generate_comp:
                        # Unlike the image node (which can silently fall back to a None composite
                        # output for a single frame), silently dropping a missing "comp" here would
                        # desync the composite output's frame count from alpha/foreground mid-video.
                        # Fail loudly instead of writing a corrupted/partial composite stream.
                        if any("comp" not in r or r["comp"] is None for r in result):
                            raise RuntimeError(
                                "generate_comp is True but the CorridorKey engine did not return a "
                                "'comp' frame for every frame in this batch."
                            )
                        comp_stack = np.stack([r["comp"] for r in result], axis=0)

                    if output_format == "video":
                        # Deferred: torch is an execution-time dependency, absent from the
                        # process that only edits this node. Importing it at module scope would
                        # make the node impossible to instantiate on a machine that never runs it.
                        import torch

                        assert video_writers is not None
                        video_writers["alpha"].write(torch.from_numpy(alpha_stack).unsqueeze(1).float())
                        video_writers["foreground"].write(torch.from_numpy(fg_stack).permute(0, 3, 1, 2).float())
                        if comp_stack is not None:
                            video_writers["composite"].write(torch.from_numpy(comp_stack).permute(0, 3, 1, 2).float())
                    else:
                        for offset, idx in enumerate(range(batch_start, batch_end)):
                            alpha_path = sequence_dirs["alpha"] / f"alpha_{idx:05d}.exr"
                            ck.write_frame_gray(alpha_path, alpha_stack[offset])
                            sequence_frame_paths["alpha"].append(alpha_path)

                            fg_path = sequence_dirs["foreground"] / f"foreground_{idx:05d}.exr"
                            ck.write_frame_rgb(fg_path, fg_stack[offset])
                            sequence_frame_paths["foreground"].append(fg_path)

                            if comp_stack is not None:
                                comp_path = sequence_dirs["composite"] / f"composite_{idx:05d}.exr"
                                ck.write_frame_rgb(comp_path, comp_stack[offset])
                                sequence_frame_paths["composite"].append(comp_path)

                    logger.info(
                        "CorridorKey video inference: processed frames %d-%d/%d", batch_start, batch_end, frame_count
                    )
            finally:
                if video_writers is not None:
                    for writer in video_writers.values():
                        writer.close()

            if output_format == "video":
                self.parameter_output_values["alpha"] = ck.save_video_artifact(
                    video_paths["alpha"].read_bytes(), "alpha"
                )
                self.parameter_output_values["foreground"] = ck.save_video_artifact(
                    video_paths["foreground"].read_bytes(), "foreground"
                )
                self.parameter_output_values["composite"] = (
                    ck.save_video_artifact(video_paths["composite"].read_bytes(), "composite")
                    if "composite" in video_paths
                    else None
                )
                self.parameter_output_values["alpha_sequence"] = None
                self.parameter_output_values["foreground_sequence"] = None
                self.parameter_output_values["composite_sequence"] = None
            else:
                self.parameter_output_values["alpha_sequence"] = ck.build_output_sequence(
                    sequence_frame_paths["alpha"], "alpha"
                )
                self.parameter_output_values["foreground_sequence"] = ck.build_output_sequence(
                    sequence_frame_paths["foreground"], "foreground"
                )
                self.parameter_output_values["composite_sequence"] = (
                    ck.build_output_sequence(sequence_frame_paths["composite"], "composite")
                    if "composite" in sequence_frame_paths
                    else None
                )
                self.parameter_output_values["alpha"] = None
                self.parameter_output_values["foreground"] = None
                self.parameter_output_values["composite"] = None
