import logging

from griptape.artifacts import ImageArtifact, ImageUrlArtifact
from griptape_nodes.exe_types.core_types import Parameter, ParameterMessage, ParameterMode
from griptape_nodes.exe_types.node_types import AsyncResult, SuccessFailureNode
from griptape_nodes.exe_types.param_components.huggingface.huggingface_repo_parameter import HuggingFaceRepoParameter
from griptape_nodes.traits.options import Options

from griptape_nodes_library_corridorkey import corridorkey_colorspace as cc
from griptape_nodes_library_corridorkey import corridorkey_common as ck

logger = logging.getLogger("corridorkey_library")


class CorridorKeyInference(SuccessFailureNode):
    """Run the CorridorKey neural keying pipeline on a single image frame.

    Takes an RGB image and an optional coarse alpha-hint mask; if no hint is
    supplied, BiRefNet is used internally to generate one. Produces a clean
    straight alpha matte, a despilled sRGB foreground, a straight sRGB RGBA
    image colour-matched to the foreground, and an optional composite-on-
    checkerboard preview. Supports both green-screen and blue-screen
    checkpoints with auto-download from HuggingFace.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        self.add_node_element(
            ParameterMessage(
                name="usage_tip",
                variant="tip",
                title="Usage Tip",
                value=(
                    "Leave `alpha_hint` empty to have BiRefNet generate one automatically -- only supply "
                    "your own hint if you need tighter control over the matte's coarse shape. Use `foreground` "
                    "for a clean despilled plate, and `rgba` when you need a single drop-in matte image. Leave "
                    "`color_mode` as 'basic' unless you need an OCIO display-view transform on `rgba`, in "
                    "which case set it to 'ocio' and connect `color_params`."
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

        # BiRefNet variant used to auto-generate the alpha hint.
        self._birefnet_param = HuggingFaceRepoParameter(
            self,
            repo_ids=ck.BIREFNET_MODEL_REPO_IDS,
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
                    "Colour management used to convert the `rgba` output's internal linear data "
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
                name="alpha",
                allowed_modes={ParameterMode.OUTPUT},
                output_type="ImageUrlArtifact",
                default_value=None,
                tooltip=(
                    "Single-channel straight alpha matte (float 0-1), including `auto_despeckle` cleanup "
                    "when enabled, encoded as a grayscale PNG."
                ),
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
                    "Straight (unpremultiplied) sRGB RGBA -- RGB channels identical to `foreground`, plus "
                    "an alpha channel -- encoded as an 8-bit PNG for use as a drop-in matte. See `color_mode`/"
                    "`color_params` for how the internal linear data is converted to sRGB."
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

    def after_value_set(self, parameter: Parameter, value) -> None:
        if parameter.name == "color_mode":
            if value == cc.COLOR_MODE_OCIO:
                self.show_parameter_by_name("color_params")
            else:
                self.hide_parameter_by_name("color_params")

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

        device = ck.get_device()
        logger.info("CorridorKey inference: device=%s", device)

        model_repo_id, _ = self._model_param.get_repo_revision()
        screen_color = ck.SCREEN_COLOR_BY_REPO[model_repo_id]
        # screen_channel: 1=green (RGB index), 2=blue.
        screen_channel = 1 if screen_color == "green" else 2

        # Decode the RGB source image to float32 [H, W, 3].
        image_bytes = ck.artifact_to_bytes(image_artifact)
        image_np = ck.decode_rgb_image(image_bytes)

        # Either decode the supplied alpha hint or generate one with BiRefNet.
        if isinstance(alpha_hint_artifact, (ImageArtifact, ImageUrlArtifact)):
            alpha_hint_bytes = ck.artifact_to_bytes(alpha_hint_artifact)
            alpha_hint_np = ck.decode_alpha_image(alpha_hint_bytes)
            alpha_hint_np = ck.resize_alpha_to_shape(alpha_hint_np, image_np.shape[0], image_np.shape[1])
        else:
            birefnet_repo_id, _ = self._birefnet_param.get_repo_revision()
            handler = ck.load_birefnet(birefnet_repo_id, device)
            alpha_hint_np = ck.run_birefnet(handler, image_np)

        # Load the keying engine and run inference.
        engine = ck.load_engine(model_repo_id, device, img_size)

        result = engine.process_frame(
            image_np,
            alpha_hint_np,
            refiner_scale=refiner_scale,
            input_is_linear=input_is_linear,
            # CorridorKeyModule's fg buffer is always straight in practice regardless of this
            # flag (its own comment: "though our pipeline forces straight") -- pass True
            # unconditionally so `comp`'s compositing formula is always the correct one. Our
            # own `fg_is_straight` parameter is instead applied to `foreground` below.
            fg_is_straight=True,
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

        rgba_out = result["processed"]
        # rgba_out's alpha channel is the despeckled matte (when auto_despeckle is True) --
        # result["alpha"] is the pre-despeckle raw prediction and was previously used here,
        # meaning auto_despeckle/despeckle_size had no effect on this output at all.
        alpha_out = rgba_out[..., 3:4]

        color_mode = str(self.parameter_values.get("color_mode") or cc.COLOR_MODE_BASIC)
        color_params = self.parameter_values.get("color_params") if color_mode == cc.COLOR_MODE_OCIO else None
        if color_mode == cc.COLOR_MODE_OCIO and color_params is None:
            raise ValueError("color_mode is 'ocio' but no color_params input is connected.")
        foreground_srgb, rgba_srgb = cc.build_foreground_and_rgba(rgba_out, color_params, fg_is_straight)

        self.parameter_output_values["alpha"] = ck.save_image_artifact(ck.encode_grayscale_png(alpha_out), "alpha")
        self.parameter_output_values["foreground"] = ck.save_image_artifact(
            ck.encode_rgb_png(foreground_srgb), "foreground"
        )
        self.parameter_output_values["rgba"] = ck.save_image_artifact(ck.encode_rgba_png(rgba_srgb), "rgba")

        if generate_comp and "comp" in result and result["comp"] is not None:
            comp_out = result["comp"]
            self.parameter_output_values["composite"] = ck.save_image_artifact(ck.encode_rgb_png(comp_out), "composite")
        else:
            self.parameter_output_values["composite"] = None
