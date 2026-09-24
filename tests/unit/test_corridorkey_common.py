"""Tests for the source-passthrough fix (GitHub issue #3).

Covers corridorkey_common._blend_source_passthrough directly -- the pure numpy/cv2 blend
math -- without requiring CorridorKeyModule, torch, or a real Griptape Nodes engine to be
installed. apply_source_passthrough() itself is exercised end-to-end via the gtn-verify
skill instead, since it deep-imports CorridorKeyModule.core.color_utils, which is only
available once the advanced library has been bootstrapped.
"""

from __future__ import annotations

import numpy as np

from griptape_nodes_library_corridorkey import corridorkey_common as ck


def _checkerboard_free_frame(size: int = 200) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a source/model-fg pair with a solid alpha=1 interior and alpha=0 background.

    The interior square is well away from the edge (>20px in from the transition), so it
    should end up unaffected by even the largest erosion/blur radii this function computes.
    """
    original = np.full((size, size, 3), 0.2, dtype=np.float32)
    model_fg = np.full((size, size, 3), 0.9, dtype=np.float32)
    alpha = np.zeros((size, size, 1), dtype=np.float32)
    alpha[40:160, 40:160, :] = 1.0
    return original, model_fg, alpha


class TestBlendSourcePassthrough:
    def test_far_interior_uses_original_source_pixels(self) -> None:
        original, model_fg, alpha = _checkerboard_free_frame()
        blended = ck._blend_source_passthrough(original, model_fg, alpha)

        # Deep interior, far from both the erosion and blur radii -- should be pure source.
        np.testing.assert_allclose(blended[100, 100], original[100, 100], atol=1e-5)

    def test_far_exterior_uses_model_prediction(self) -> None:
        original, model_fg, alpha = _checkerboard_free_frame()
        blended = ck._blend_source_passthrough(original, model_fg, alpha)

        # Far background, well outside the interior mask's dilation/blur reach.
        np.testing.assert_allclose(blended[5, 5], model_fg[5, 5], atol=1e-5)

    def test_edge_band_blends_both(self) -> None:
        original, model_fg, alpha = _checkerboard_free_frame()
        blended = ck._blend_source_passthrough(original, model_fg, alpha)

        # Right at the alpha edge, erosion should have already pulled the "pure source"
        # region inward, so this pixel must be a genuine mix, not equal to either input.
        edge_pixel = blended[40, 100]
        assert not np.allclose(edge_pixel, original[40, 100], atol=1e-5)
        assert not np.allclose(edge_pixel, model_fg[40, 100], atol=1e-5)

    def test_fully_opaque_alpha_returns_pure_source(self) -> None:
        size = 64
        original = np.random.default_rng(0).random((size, size, 3)).astype(np.float32)
        model_fg = np.random.default_rng(1).random((size, size, 3)).astype(np.float32)
        alpha = np.ones((size, size, 1), dtype=np.float32)

        blended = ck._blend_source_passthrough(original, model_fg, alpha)

        np.testing.assert_allclose(blended, original, atol=1e-5)

    def test_fully_transparent_alpha_returns_pure_model_fg(self) -> None:
        size = 64
        original = np.random.default_rng(0).random((size, size, 3)).astype(np.float32)
        model_fg = np.random.default_rng(1).random((size, size, 3)).astype(np.float32)
        alpha = np.zeros((size, size, 1), dtype=np.float32)

        blended = ck._blend_source_passthrough(original, model_fg, alpha)

        np.testing.assert_allclose(blended, model_fg, atol=1e-5)

    def test_accepts_2d_alpha(self) -> None:
        size = 64
        original = np.full((size, size, 3), 0.3, dtype=np.float32)
        model_fg = np.full((size, size, 3), 0.7, dtype=np.float32)
        alpha_2d = np.ones((size, size), dtype=np.float32)

        blended = ck._blend_source_passthrough(original, model_fg, alpha_2d)

        np.testing.assert_allclose(blended, original, atol=1e-5)
