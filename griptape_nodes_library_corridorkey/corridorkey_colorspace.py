"""Colour management for CorridorKeyInference's `rgba` output.

Converts CorridorKeyModule's despilled/linear/alpha-premultiplied "processed" buffer
into straight sRGB RGBA, either via the local sRGB transfer function or, when a
color_params input is connected, via an OCIO display-view transform dispatched
through griptape-nodes-library-opencolorio's ColorspaceTransformRequest service.
"""

import logging
from typing import Any

import numpy as np
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes

logger = logging.getLogger("corridorkey_library")

# Colour management modes for the `rgba` output's linear->sRGB conversion (CorridorKeyInference's
# `color_mode` parameter). "ocio" dispatches through a connected OCIOColorParamsArtifact and the
# griptape-nodes-library-opencolorio ColorspaceTransformRequest service; "basic" uses the local
# sRGB transfer function below.
COLOR_MODE_BASIC = "basic"
COLOR_MODE_OCIO = "ocio"


def linear_to_srgb(x: np.ndarray) -> np.ndarray:
    """Encode scene-linear values to the sRGB transfer function (official piecewise curve)."""
    x = np.clip(x, 0.0, None)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * np.power(x, 1.0 / 2.4) - 0.055).astype(np.float32)


def unpremultiply_rgb(rgb: np.ndarray, alpha: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Divide alpha-premultiplied colour by alpha to recover straight (unpremultiplied) colour."""
    return (rgb / (alpha + eps)).astype(np.float32)


def find_colorspace_transform_request_type() -> type | None:
    """Discover ColorspaceTransformRequest from any loaded library -- no hard import.

    Delegates to griptape_nodes_openexr's shared implementation of this same discovery
    pattern, since this library already uses griptape_nodes_openexr for EXR sequence I/O
    (see corridorkey_common.read_frame_rgb/write_frame_rgb). griptape_nodes_openexr is
    only listed in this library's runtime pip_dependencies (installed by the advanced-
    library bootstrap), not this repo's own dev environment, so it may genuinely be
    absent -- e.g. under a plain `uv sync` -- which is equivalent to the OCIO service
    not being available.
    """
    try:
        from griptape_nodes_openexr.exr.ocio_helpers import (
            find_colorspace_transform_request_type as find_request_type,
        )
    except ImportError:
        logger.debug("griptape_nodes_openexr not installed; OCIO colour management unavailable")
        return None

    return find_request_type()


def apply_ocio_or_local_srgb(rgb: np.ndarray, color_params: Any | None) -> tuple[np.ndarray, str]:
    """Convert scene-linear RGB to display-referred sRGB.

    If `color_params` (an OCIOColorParamsArtifact-shaped object exposing source_colorspace,
    display, view, and config_path) is given, dispatches an OCIO display-view transform via
    the ColorspaceTransformRequest service discovered on LibraryRegistry -- the same zero-
    hard-import pattern griptape_nodes_openexr's DisplayEXRPart uses -- and raises ValueError
    if OCIO is requested but the library isn't loaded or the transform fails. Otherwise falls
    back to the local sRGB transfer function.
    """
    if color_params is None:
        return linear_to_srgb(rgb), COLOR_MODE_BASIC

    try:
        source_colorspace = color_params.source_colorspace
        display = color_params.display
        view = color_params.view
        config_path = color_params.config_path
    except AttributeError as e:
        msg = f"color_params is missing a required attribute (source_colorspace, display, view, config_path): {e}"
        raise ValueError(msg) from e

    if not display or not view:
        # ColorspaceTransformRequest's own handler silently passes pixels through
        # UNCHANGED when display/view are blank (e.g. an "OCIO Color Parameters" node
        # with no config connected yet -- griptape-nodes-library-opencolorio documents
        # this as intentional scaffolding behaviour). For us that would mean returning
        # still-linear data labeled as sRGB -- the exact "rgba looks linear" symptom
        # this function exists to prevent -- so treat a blank display/view the same as
        # OCIO not being configured at all, rather than trusting the passthrough.
        logger.debug("OCIO color_params has blank display/view; falling back to local sRGB transfer function")
        return linear_to_srgb(rgb), COLOR_MODE_BASIC

    req_type = find_colorspace_transform_request_type()
    if req_type is None:
        msg = (
            f"OCIO transform requested (source={source_colorspace!r}, display={display!r}, view={view!r}) "
            "but the OpenColorIO library is not loaded. Load it, or switch color_mode to 'basic'."
        )
        raise ValueError(msg)

    try:
        req = req_type(
            pixels=rgb,
            source_colorspace=source_colorspace,
            display=display,
            view=view,
            config_path=config_path,
        )
        result = GriptapeNodes.handle_request(req)
    except TypeError as e:
        msg = f"OCIO transform failed -- {e}"
        raise ValueError(msg) from e

    if not result.succeeded():
        msg = f"OCIO transform failed -- {result.result_details}"
        raise ValueError(msg)

    logger.debug("OCIO: transform succeeded (%s -> %s/%s)", source_colorspace, display, view)
    return result.pixels, f"ocio:{source_colorspace}->{display}/{view}"  # type: ignore[attr-defined]


def build_straight_srgb_rgba(premultiplied_linear_rgba: np.ndarray, color_params: Any | None) -> np.ndarray:
    """Convert CorridorKeyModule's despilled/linear/alpha-premultiplied RGBA into straight sRGB RGBA.

    CorridorKeyModule's "processed" output is alpha-premultiplied in linear light (built for
    EXR-style delivery). An 8-bit PNG has no standard convention for premultiplied alpha, and
    writing those linear values straight into an 8-bit PNG -- interpreted as sRGB by every
    consumer -- made the `rgba` output render darker/color-shifted than `foreground` for the
    same frame (GitHub issue #2). Un-premultiply, then gamma-encode back to sRGB.
    """
    premultiplied_rgb = premultiplied_linear_rgba[..., :3]
    alpha = premultiplied_linear_rgba[..., 3:4]
    straight_linear_rgb = unpremultiply_rgb(premultiplied_rgb, alpha)
    straight_srgb_rgb, _label = apply_ocio_or_local_srgb(straight_linear_rgb, color_params)
    return np.concatenate([straight_srgb_rgb, alpha], axis=-1)


def build_foreground_and_rgba(
    premultiplied_linear_rgba: np.ndarray, color_params: Any | None, fg_is_straight: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    """Build CorridorKeyInference's `foreground` and `rgba` outputs from one shared buffer.

    Both come from the same despilled buffer (rather than `foreground` using the model's raw,
    undespilled prediction), so despill state and colorspace can no longer diverge between them.

    `rgba` is always straight -- 8-bit PNG has no standard convention for premultiplied alpha,
    so a fixed convention is safer for a "drop-in matte" output regardless of `fg_is_straight`.

    `foreground` honours `fg_is_straight`: straight (default, matches the published checkpoints)
    or alpha-premultiplied. The caller is responsible for passing CorridorKeyModule's own
    `fg_is_straight` engine argument as `True` unconditionally when generating `comp` -- that
    buffer is always straight in practice (CorridorKeyModule's own comment: "though our pipeline
    forces straight"), so `comp`'s compositing formula must not be switched by this flag.
    """
    premultiplied_rgb = premultiplied_linear_rgba[..., :3]
    rgba_srgb = build_straight_srgb_rgba(premultiplied_linear_rgba, color_params)

    if fg_is_straight:
        foreground_srgb = rgba_srgb[..., :3]
    else:
        foreground_srgb, _label = apply_ocio_or_local_srgb(premultiplied_rgb, color_params)

    return foreground_srgb, rgba_srgb
