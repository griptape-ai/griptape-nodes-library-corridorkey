# Griptape Nodes CorridorKey Library

A [Griptape Nodes](https://www.griptapenodes.com/) library for neural-network green/blue screen keying using [CorridorKey](https://github.com/nikopueringer/CorridorKey).

## Overview

This library wraps CorridorKey, a transformer-based chroma keying tool built by Niko Pueringer / Corridor Digital for professional VFX pipelines. Given an RGB frame and an optional coarse alpha hint, CorridorKey produces a clean straight (unpremultiplied) alpha matte, a despilled foreground with the screen color removed, a linear premultiplied RGBA matte for compositing, and an optional checkerboard composite preview.

The library exposes a single `CorridorKey Inference` node that handles the full pipeline: when no alpha hint is supplied it runs BiRefNet internally to generate one, then runs the CorridorKey GreenFormer (Hiera + refiner) keying network with the green-screen or blue-screen checkpoint. All weights auto-download from HuggingFace on first use.

## Requirements

- **GPU**: CUDA (NVIDIA), Apple Silicon MPS, or AMD ROCm (Linux). CPU fallback is supported but slow. Minimum 6-8 GB VRAM is recommended for 2048x2048 inference.
- **Griptape Nodes Engine**: Version 0.84.0 or later

## Nodes

### CorridorKey Inference

Runs the full CorridorKey neural keying pipeline on a single image frame. If `alpha_hint` is left empty, BiRefNet is used internally to generate a coarse hint from the source image.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `model` | HuggingFace repo | CorridorKey checkpoint to load (`nikopueringer/CorridorKey_v1.0` for green, `nikopueringer/CorridorKeyBlue_1.0` for blue). |
| `birefnet_model` | HuggingFace repo | BiRefNet variant used to auto-generate an alpha hint when `alpha_hint` is not provided. Matting variants are best for soft-edge subjects (hair/fur); the general model is better for hard-edged objects. Ignored if `alpha_hint` is supplied. |
| `image` | ImageUrlArtifact | RGB source frame. Any resolution; internally resized to `img_size` for inference and upsampled back. |
| `alpha_hint` | ImageUrlArtifact (optional) | Optional coarse alpha-hint mask, 0=background 1=foreground. If left empty, BiRefNet is run internally to generate the hint. |
| `img_size` | int | Inference resolution (square). 2048 matches the trained resolution; lower values are faster but less accurate. (default: 2048) |
| `input_is_linear` | bool | If True, the input image is treated as linear and resized in linear before being converted to sRGB for the model. Leave False for typical sRGB inputs. (default: false) |
| `fg_is_straight` | bool | If True, the foreground output is straight (unpremultiplied). Leave True for the published checkpoints. (default: true) |
| `despill_strength` | float | 0.0-1.0 multiplier for the despill pass that removes screen-color contamination from the foreground. (default: 1.0) |
| `auto_despeckle` | bool | If True, runs a morphological cleanup that removes small disconnected islands from the predicted matte. (default: true) |
| `despeckle_size` | int | Minimum connected-component size (in pixels) preserved when `auto_despeckle` is True. (default: 400) |
| `refiner_scale` | float | Multiplier on the refiner head's delta contribution. 1.0 = default, >1.0 sharpens edges, 0.0 disables the refiner. (default: 1.0) |
| `generate_comp` | bool | If True, also produces a composite-on-gray-checkerboard preview. (default: true) |
| `alpha` | ImageUrlArtifact (output) | Single-channel straight alpha matte (float 0-1) encoded as a grayscale PNG. |
| `foreground` | ImageUrlArtifact (output) | Despilled sRGB foreground, straight (unpremultiplied), encoded as PNG. |
| `composite` | ImageUrlArtifact (output) | sRGB composite of the foreground over a gray checkerboard (PNG). None when `generate_comp` is False. |
| `rgba` | ImageUrlArtifact (output) | Linear premultiplied RGBA (foreground * alpha, plus alpha channel) encoded as an 8-bit PNG with alpha for use as a drop-in matte. |

## Available Models

The following models are available from HuggingFace:

| Model | Description |
|-------|-------------|
| `nikopueringer/CorridorKey_v1.0` | Primary green-screen keying weights (~300 MB). GreenFormer with a Hiera-Base-Plus backbone and refiner head. |
| `nikopueringer/CorridorKeyBlue_1.0` | Blue-screen keying weights (~300 MB). Identical architecture to the green checkpoint; only the trained weights differ. |
| `ZhengPeng7/BiRefNet-matting` | BiRefNet image-matting weights, used when no alpha hint is supplied. Best for soft-edge subjects (hair, fur). |
| `ZhengPeng7/BiRefNet` | BiRefNet general-purpose segmentation weights, used when no alpha hint is supplied. Better for hard-edged objects. |

Models are downloaded automatically on first use and cached for subsequent runs. The CorridorKey checkpoints land under `<site-packages>/CorridorKeyModule/checkpoints/`; the BiRefNet weights land under `<site-packages>/BiRefNetModule/checkpoints/<repo_name>/`.

## Installation

### Prerequisites

- [Griptape Nodes](https://github.com/griptape-ai/griptape-nodes) installed and running
- A CUDA-capable NVIDIA GPU, an Apple Silicon Mac, or an AMD ROCm-capable Linux machine

### Install the Library

1. **Clone the repository** to your Griptape Nodes workspace directory:

   ```bash
   cd `gtn config show workspace_directory`
   git clone --recurse-submodules https://github.com/griptape-ai/griptape-nodes-library-corridorkey.git
   ```

2. **Add the library** in the Griptape Nodes Editor:

   - Open the Settings menu and navigate to the *Libraries* settings
   - Click on *+ Add Library* at the bottom of the settings panel
   - Enter the path to the library JSON file:
     ```
     <workspace_directory>/griptape-nodes-library-corridorkey/griptape_nodes_library_corridorkey/griptape-nodes-library.json
     ```
   - You can check your workspace directory with `gtn config show workspace_directory`
   - Close the Settings Panel
   - Click on *Refresh Libraries*

3. **Verify installation** by checking that the `CorridorKey Inference` node appears in the node palette under the "CorridorKey" category.

## Usage

### CorridorKey Inference

1. Add a **CorridorKey Inference** node to your workflow.
2. Connect your source image to `image`. The node accepts any resolution; the source is resized internally to `img_size` for inference and upsampled back to its original resolution.
3. Choose the keying checkpoint via `model`: `nikopueringer/CorridorKey_v1.0` for green screens, `nikopueringer/CorridorKeyBlue_1.0` for blue screens.
4. *(Optional)* Connect a coarse `alpha_hint` mask if you have one. If you leave it empty, the node will run BiRefNet on `image` to generate the hint automatically; pick the BiRefNet variant via `birefnet_model` (matting for soft edges, general for hard edges).
5. *(Optional)* Tune `despill_strength`, `auto_despeckle`, `despeckle_size`, and `refiner_scale` to control the despill pass, matte cleanup, and refiner head.
6. Connect any of `alpha`, `foreground`, `composite`, or `rgba` to your downstream nodes:
   - `alpha` is the cleaned matte for use in compositing
   - `foreground` is the despilled sRGB plate
   - `rgba` is an 8-bit linear premultiplied RGBA matte ready to drop into a compositor
   - `composite` is a quick checkerboard preview for sanity-checking the matte

The first run will download the selected CorridorKey checkpoint (~300 MB) and, if used, the BiRefNet weights (varies by variant). On Linux/Windows with a C compiler installed, CorridorKey also performs a one-time `torch.compile` pass that adds 30 s - 20 min to the first call; subsequent calls reuse the autotune cache under `~/.cache/corridorkey/inductor/`.

## Troubleshooting

### Library Not Loading

- Ensure the git submodule is initialized. If you cloned without `--recurse-submodules`, run:
  ```bash
  git submodule update --init --recursive
  ```

### GPU Not Available

- Verify your GPU drivers are up to date
- For NVIDIA GPUs, ensure CUDA is properly installed and `nvidia-smi` reports your GPU correctly
- For Apple Silicon, ensure you're running on macOS 12.3 or later
- For AMD ROCm, ensure ROCm 6+ is installed and `/opt/rocm` exists (Linux only)

### Out of Memory Errors

- Reduce `img_size` (e.g., from 2048 to 1024). The 2048 default matches the trained resolution; lower values run faster and use less VRAM at the cost of fine-edge accuracy.
- Close other GPU-intensive applications before running inference.

### Slow First Run

- On Linux and Windows, CorridorKey runs `torch.compile` on the first `process_frame` call to autotune kernels for the active GPU. This adds 30 s - 20 min depending on the backend. The autotune cache is persisted under `~/.cache/corridorkey/inductor/` and reused on subsequent runs.
- To skip compilation entirely (slower steady-state, but no compile delay), set the environment variable `CORRIDORKEY_SKIP_COMPILE=1` before launching the engine.

## Additional Resources

- [CorridorKey GitHub](https://github.com/nikopueringer/CorridorKey)
- [Griptape Nodes Documentation](https://docs.griptapenodes.com/)
- [Griptape Discord](https://discord.gg/griptape)

## License

This library is distributed under the **Corridor Key Licence** (Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International, with the Corridor Key Additional Terms and Conditions). This matches the license of the upstream [CorridorKey](https://github.com/nikopueringer/CorridorKey) project, as required by the share-alike clause of that licence. See the [LICENSE](./LICENSE) file in this repository for the full text.

The "CorridorKey" name and any associated trademarks remain the property of Corridor Digital.
