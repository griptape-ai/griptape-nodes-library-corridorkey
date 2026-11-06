"""Node errors raised by the CorridorKey nodes reach the editor as NodeError with a clean message."""

from __future__ import annotations

import pytest
from griptape_nodes.exe_types.core_types import NodeError

from griptape_nodes_library_corridorkey.corridorkey_inference import CorridorKeyInference
from griptape_nodes_library_corridorkey.corridorkey_video_inference import CorridorKeyVideoInference


def _without_model_checks(node):
    """Skip the Hugging Face download checks, which are the engine's and need a real model cache."""
    for attr in ("_model_param", "_birefnet_param", "_gvm_param", "_videomama_unet_param", "_videomama_base_param"):
        param = getattr(node, attr, None)
        if param is not None:
            param.validate_before_node_run = lambda: None
    return node


@pytest.fixture
def image_node():
    return _without_model_checks(CorridorKeyInference(name="key"))


@pytest.fixture
def video_node():
    return _without_model_checks(CorridorKeyVideoInference(name="video_key"))


def _only(errors):
    assert errors is not None
    assert len(errors) == 1
    assert isinstance(errors[0], NodeError)
    return errors[0]


def test_image_validation_names_the_parameter(image_node):
    error = _only(image_node.validate_before_node_run())
    assert str(error) == "Connect an image to 'Image'."
    assert error.fields == {"parameter": "image"}


def test_video_validation_names_the_parameter(video_node):
    error = _only(video_node.validate_before_node_run())
    assert str(error) == "Connect a video to 'Video'."
    assert error.fields == {"parameter": "video"}


def test_videomama_without_mask_is_reported(video_node):
    video_node.parameter_values["hint_source"] = "videomama"
    video_node.parameter_values["video"] = "clip.mp4"
    error = _only(video_node.validate_before_node_run())
    assert "'Mask Hint' is required when 'Hint Source' is 'videomama'" in str(error)
    assert error.fields == {"parameter": "mask_hint"}


def test_failure_raises_node_error_without_node_prefix_when_failed_is_unwired(image_node):
    image_node._has_outgoing_connections = lambda _param: False
    with pytest.raises(NodeError, match="^Connect an image to 'Image'\\.$"):
        image_node._run_inference()


def test_failure_follows_failed_output_when_wired(image_node):
    image_node._has_outgoing_connections = lambda _param: True
    image_node._run_inference()
    assert image_node.parameter_output_values["was_successful"] is False
    assert image_node.parameter_output_values["result_details"] == "Connect an image to 'Image'."


@pytest.mark.parametrize(("parameter", "label"), [("video", "Video"), ("mask_hint", "Mask Hint")])
def test_unsupported_frame_source_names_the_calling_parameter(tmp_path, parameter, label):
    from griptape_nodes_library_corridorkey import corridorkey_common as ck

    with pytest.raises(NodeError) as caught:
        ck.resolve_source_to_frames(42, tmp_path, parameter, label)
    assert f"'{label}' has an unsupported input" in str(caught.value)
    assert caught.value.fields == {"parameter": parameter}


@pytest.mark.parametrize(("parameter", "label"), [("video", "Video"), ("mask_hint", "Mask Hint")])
def test_missing_frames_name_the_calling_parameter(tmp_path, parameter, label):
    from griptape_nodes_library_corridorkey import corridorkey_common as ck

    missing = str(tmp_path / "nothing.####.exr")
    with pytest.raises(NodeError) as caught:
        ck.resolve_source_to_frames(missing, tmp_path, parameter, label)
    assert f"for '{label}'" in str(caught.value)
    assert caught.value.fields == {"parameter": parameter}


def test_both_callers_pass_their_own_parameter():
    import inspect

    from griptape_nodes_library_corridorkey import corridorkey_video_inference as vi

    source = inspect.getsource(vi)
    assert '"video", "Video"' in source
    assert '"mask_hint", "Mask Hint"' in source
