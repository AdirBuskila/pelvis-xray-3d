# Unit 04 · The network

**Leads:** both (swap the driver on every TODO) · **Videos:** 09, 10 · **Proposal week:** 4 · **Status:** ready

## Goals
- Compute the output size of any conv / transposed conv layer, and write a convolution by hand.
- Build the project's network in PyTorch: a 2D encoder per view, lifting along each view's rays, a 3D decoder
  with skip connections, 64³ logits out.
- Say, for any pixel of either X-ray, which voxels of the cube it reaches, and why.
- Read the network's parameter count and shapes and explain where they come from.

## Videos
- 09 · Convolution from scratch (part A), 10 · Encoder-decoder and U-Net: from 2D to 3D (part B).
- Optional: [Geiger 10.2 Semantic Segmentation](https://www.youtube.com/watch?v=NhSujgDL918),
  [U-Net in 10 min](https://www.youtube.com/watch?v=NhdzGfB1q74),
  [Transpose Conv vs Unpooling](https://www.youtube.com/watch?v=JnN_1Nh7w4M).

## Before the session
- Nothing to download. The pictures use one sample of the processed dataset v1 (unit 02) if it's on disk.
- CPU only; the whole notebook runs in about a minute.

## TODOs
| Part · video | TODO | Function / class | Moves into | Concept |
|---|---|---|---|---|
| A · 09 | A1 | `conv_out_size` | `src/xray3d/nets.py` | kernel positions; the transposed stamp |
| | A2 | `conv2d_naive` | `nets.py` | the sliding window = what `nn.Conv2d` computes |
| | A3 | `ConvBlock` | `nets.py` | conv → InstanceNorm → ReLU, 2D or 3D |
| B · 10 | B1 | `Encoder2D` | `nets.py` | downsampling stages, maps at several scales |
| | B2 | `lift_view` | `nets.py` | a pixel is a ray: back-projecting features into the cube |
| | B3 | `Decoder3D` | `nets.py` | transposed convs, skip connections, the prior bias |
| | B4 | `XrayTo3D` | `nets.py` | one encoder per view, fusion, the whole network |

## Done when
- every check is green, everything is in `src/xray3d/nets.py`, and `pytest units/04-network` passes with nothing skipped,
- `LOG.md` has the parameter counts and the CPU forward time,
- each of you can explain, without the notebook, which voxels one AP pixel lights up.
