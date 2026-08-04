"""Tests for the rgba colorspace fix (GitHub issue #2).

Covers corridorkey_common's linear<->sRGB helpers, OCIO discovery/dispatch, and the
build_straight_srgb_rgba regression case, all without requiring CorridorKeyModule,
torch, or a real Griptape Nodes engine to be installed.
"""

from __future__ import annotations

import sys
import types
from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from griptape_nodes_library_corridorkey import corridorkey_colorspace as ck


def _srgb_to_linear_ref(x: np.ndarray) -> np.ndarray:
    """Reference sRGB->linear inverse, independent of the module under test."""
    x = np.clip(x, 0.0, None)
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4).astype(np.float32)


@dataclass
class FakeColorParams:
    source_colorspace: str = "scene_linear"
    display: str = "sRGB"
    view: str = "ACES"
    config_path: str | None = None


class TestLinearToSrgb:
    def test_matches_known_values(self) -> None:
        # Round-tripping a reference sRGB->linear conversion should recover the original.
        srgb_in = np.array([0.0, 0.18, 0.5, 1.0], dtype=np.float32)
        linear = _srgb_to_linear_ref(srgb_in)
        srgb_out = ck.linear_to_srgb(linear)
        np.testing.assert_allclose(srgb_out, srgb_in, atol=1e-5)

    def test_darker_than_input_before_conversion(self) -> None:
        # Sanity check on the actual bug: linear values are numerically darker than
        # their sRGB-encoded counterparts, which is why writing them unconverted into
        # an 8-bit PNG produced the reported darkening.
        srgb = np.array([0.5], dtype=np.float32)
        linear = _srgb_to_linear_ref(srgb)
        assert linear[0] < srgb[0]
        assert ck.linear_to_srgb(linear)[0] == pytest.approx(srgb[0], abs=1e-5)


class TestUnpremultiplyRgb:
    def test_recovers_straight_color(self) -> None:
        straight = np.full((2, 2, 3), 0.4, dtype=np.float32)
        alpha = np.full((2, 2, 1), 0.5, dtype=np.float32)
        premultiplied = straight * alpha
        recovered = ck.unpremultiply_rgb(premultiplied, alpha)
        np.testing.assert_allclose(recovered, straight, atol=1e-5)

    def test_zero_alpha_does_not_divide_by_zero(self) -> None:
        rgb = np.zeros((1, 1, 3), dtype=np.float32)
        alpha = np.zeros((1, 1, 1), dtype=np.float32)
        result = ck.unpremultiply_rgb(rgb, alpha)
        assert np.all(np.isfinite(result))


class TestFindColorspaceTransformRequestType:
    def test_returns_none_when_openexr_not_installed(self) -> None:
        with patch.dict(sys.modules, {"griptape_nodes_openexr.exr.ocio_helpers": None}):
            assert ck.find_colorspace_transform_request_type() is None

    def test_delegates_to_openexr_helper(self) -> None:
        sentinel_type = object()
        fake_module = types.SimpleNamespace(find_colorspace_transform_request_type=lambda: sentinel_type)
        with patch.dict(sys.modules, {"griptape_nodes_openexr.exr.ocio_helpers": fake_module}):
            assert ck.find_colorspace_transform_request_type() is sentinel_type


class TestApplyOcioOrLocalSrgb:
    def test_no_color_params_uses_local_fallback(self) -> None:
        linear = np.array([[[0.05, 0.05, 0.05]]], dtype=np.float32)
        srgb, label = ck.apply_ocio_or_local_srgb(linear, None)
        np.testing.assert_allclose(srgb, ck.linear_to_srgb(linear))
        assert label == ck.COLOR_MODE_BASIC

    def test_raises_when_ocio_library_not_loaded(self) -> None:
        linear = np.zeros((1, 1, 3), dtype=np.float32)
        with patch.object(ck, "find_colorspace_transform_request_type", return_value=None):
            with pytest.raises(ValueError, match="not loaded"):
                ck.apply_ocio_or_local_srgb(linear, FakeColorParams())

    def test_raises_on_missing_attribute(self) -> None:
        linear = np.zeros((1, 1, 3), dtype=np.float32)

        class IncompleteColorParams:
            source_colorspace = "scene_linear"
            # missing display/view/config_path

        with pytest.raises(ValueError, match="missing a required attribute"):
            ck.apply_ocio_or_local_srgb(linear, IncompleteColorParams())

    def test_dispatches_through_ocio_and_returns_pixels(self) -> None:
        linear = np.zeros((1, 1, 3), dtype=np.float32)
        expected = np.full((1, 1, 3), 0.5, dtype=np.float32)

        req_type = MagicMock()
        mock_result = MagicMock()
        mock_result.succeeded.return_value = True
        mock_result.pixels = expected

        mock_gn = MagicMock()
        mock_gn.handle_request.return_value = mock_result

        with (
            patch.object(ck, "find_colorspace_transform_request_type", return_value=req_type),
            patch.object(ck, "GriptapeNodes", mock_gn),
        ):
            pixels, label = ck.apply_ocio_or_local_srgb(linear, FakeColorParams())

        req_type.assert_called_once_with(
            pixels=linear,
            source_colorspace="scene_linear",
            display="sRGB",
            view="ACES",
            config_path=None,
        )
        assert pixels is expected
        assert label.startswith("ocio:")

    def test_raises_on_failed_ocio_result(self) -> None:
        linear = np.zeros((1, 1, 3), dtype=np.float32)
        req_type = MagicMock()
        failed = MagicMock()
        failed.succeeded.return_value = False
        failed.result_details = "unknown colorspace"

        mock_gn = MagicMock()
        mock_gn.handle_request.return_value = failed

        with (
            patch.object(ck, "find_colorspace_transform_request_type", return_value=req_type),
            patch.object(ck, "GriptapeNodes", mock_gn),
            pytest.raises(ValueError, match="unknown colorspace"),
        ):
            ck.apply_ocio_or_local_srgb(linear, FakeColorParams())

    def test_raises_on_type_error(self) -> None:
        linear = np.zeros((1, 1, 3), dtype=np.float32)
        req_type = MagicMock(side_effect=TypeError("bad constructor"))

        with (
            patch.object(ck, "find_colorspace_transform_request_type", return_value=req_type),
            pytest.raises(ValueError, match="bad constructor"),
        ):
            ck.apply_ocio_or_local_srgb(linear, FakeColorParams())


class TestBuildStraightSrgbRgba:
    def test_matches_foreground_at_full_opacity(self) -> None:
        """Regression test for issue #2: interior (alpha=1) rgba pixels must match the
        equivalent straight sRGB foreground pixel, not the raw linear premultiplied value."""
        fg_srgb = np.full((4, 4, 3), 0.5, dtype=np.float32)
        alpha = np.ones((4, 4, 1), dtype=np.float32)

        fg_linear = _srgb_to_linear_ref(fg_srgb)
        premultiplied_rgba = np.concatenate([fg_linear * alpha, alpha], axis=-1)

        rgba_srgb = ck.build_straight_srgb_rgba(premultiplied_rgba, color_params=None)

        np.testing.assert_allclose(rgba_srgb[..., :3], fg_srgb, atol=1e-5)
        np.testing.assert_allclose(rgba_srgb[..., 3:4], alpha)

    def test_partial_alpha_edge_is_unpremultiplied(self) -> None:
        fg_srgb = np.full((1, 1, 3), 0.8, dtype=np.float32)
        alpha = np.full((1, 1, 1), 0.3, dtype=np.float32)

        fg_linear = _srgb_to_linear_ref(fg_srgb)
        premultiplied_rgba = np.concatenate([fg_linear * alpha, alpha], axis=-1)

        rgba_srgb = ck.build_straight_srgb_rgba(premultiplied_rgba, color_params=None)

        # Straight colour should recover the original sRGB foreground, not the darker
        # premultiplied-and-reinterpreted-as-sRGB value the old code produced.
        np.testing.assert_allclose(rgba_srgb[..., :3], fg_srgb, atol=1e-4)


class TestBuildForegroundAndRgba:
    def test_foreground_is_identical_to_rgba_rgb_channels(self) -> None:
        """CorridorKeyInference's `foreground` and `rgba` outputs must converge: `foreground`
        is built from the same despilled/straight/sRGB buffer as `rgba`'s RGB channels, not
        the model's separate raw/undespilled prediction, so they can no longer diverge."""
        fg_srgb = np.full((3, 3, 3), 0.7, dtype=np.float32)
        alpha = np.full((3, 3, 1), 0.6, dtype=np.float32)
        premultiplied_rgba = np.concatenate([_srgb_to_linear_ref(fg_srgb) * alpha, alpha], axis=-1)

        foreground_srgb, rgba_srgb = ck.build_foreground_and_rgba(premultiplied_rgba, color_params=None)

        np.testing.assert_array_equal(foreground_srgb, rgba_srgb[..., :3])
        np.testing.assert_allclose(foreground_srgb, fg_srgb, atol=1e-4)
