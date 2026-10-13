# Retinal artery–vein segmentation: Attention-WNet and Swin-WNet

Research implementations for retinal fundus images, with a legacy TensorFlow/Keras notebook and a new PyTorch inference package.

[Paper: Artery-Vein Segmentation from Fundus Images using Deep Learning](https://arxiv.org/abs/2510.03717) — arXiv preprint, 2025.

## Start here

| Implementation | Framework | Output / supervision | Status |
| --- | --- | --- | --- |
| Original Attention-WNet / Swin-WNet notebook | TensorFlow/Keras | Final cascade prediction | Historical notebook; Kaggle-specific paths |
| V11 Attention-WNet | PyTorch | Vessel probabilities, then A/V probabilities; supervision of both cascade stages | Model source preserved; image-only inference added |
| Attention-U-Net with deep supervision | PyTorch | Final logits plus an intermediate decoder prediction | Separate experimental architecture; no trained checkpoint supplied |

**Supervising two cascade outputs and supervising intermediate decoder heads are different.** The deep-supervised Attention-U-Net is not a drop-in replacement for V11 Attention-WNet checkpoints.

## Run image-only inference

Use Python 3.10 or later. From the repository root:

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .\.venv\Scripts\Activate.ps1
python -m pip install -e ./pytorch
avseg-predict --checkpoint /path/to/model_best.pth --image /path/to/fundus.png --dataset RITE --output /path/to/new-output --device cpu
```

Supply a **V11-compatible checkpoint**: six input channels, two vessel classes, three A/V classes, and encoder widths 32/64/128/256/512. Weights are not included in this update. A missing or incompatible checkpoint fails rather than producing random-weight results.

Choose the dataset explicitly: historical V11 denoising applies to `RITE`, `RAV`, and `Fundus-AVSeg`; use `custom` for a new image without that denoising. This choice is recorded in the output.

The predictor:
- Builds green-CLAHE, OpenCV LAB, and full-image y/x coordinate channels.
- Blends overlapping patch **probabilities without a second softmax**.
- Uses **predicted** vessels to gate A/V labels; no ground-truth mask is required.
- Writes full-resolution vessel masks, A/V labels, RGB predictions, overlays, and a provenance JSON containing input/checkpoint hashes and settings.

This is an inference implementation, not a clinical validation claim. See [model status and evaluation limits](docs/MODEL_STATUS.md).

## Inspect decoder deep supervision

```python
import torch
from avseg.attention_unet_ds import AttentionUNet
from avseg.supervision import supervised_loss

model = AttentionUNet(in_channels=1, out_channels=2, deep_supervision=True)
image = torch.randn(2, 1, 64, 64)
target = torch.zeros(2, 64, 64, dtype=torch.long)
logits, auxiliary_logits = model(image)
loss = supervised_loss(logits, auxiliary_logits, target,
                       torch.nn.CrossEntropyLoss(), auxiliary_weight=0.4)
loss.backward()
```

The auxiliary head receives an intermediate decoder feature at quarter resolution and upsamples its prediction. At inference, use final logits. This example demonstrates the training connection; it does not establish trained accuracy or an improvement over the cascade.

## Checks

```bash
python -m unittest discover -s pytorch/tests -v
```

Checks cover patch edges, global coordinates, probability preservation, predicted-vessel gating, missing checkpoints, preprocessing channels, cascade output contracts, and auxiliary-head gradients. GitHub Actions installs PyTorch/OpenCV and runs the checks. Tests requiring an absent optional dependency report a skip.

## Results and reproducibility

The supplied local audit found compatible V11 weights, but no checkpoints or image datasets were included in the handoff to this update. Historical results used nominal test data for validation and model selection; they are not an independent held-out test benchmark.

Next release requirements:
1. Record independent training/validation/test identities, including augmentation-parent and patient/eye grouping where available.
2. Evaluate a checkpoint with locked preprocessing, label mapping, split manifest, and hashes.
3. Publish artery/vein Dice or F1, vessel detection, and qualitative overlays with explicit evaluation regions.
4. Report the decoder-supervision experiment separately from the original cascade.

The original notebook and dataset tree are retained. This update does not add the local GNN or AVR research pipeline.

## Legacy architecture notes

# Artery Vein Segmentation using Attention-WNet and SWIN-WNet model:

## Attention-WNet:
In this model, attention techniques are added to the skip connections in the WNet encoder. Attention techniques are used to selectively combine encoder features with decoder block features. This helps to improve the features that are essential for segmentation while ignoring the features that are irrelevant. The attention mechanisms can be trained to learn the weights that assign value to the various features in the input image.

The attention techniques output is then sent into the Attention-UNet blocks, which improve the features of the input image and extract more complex features. The Attention-UNet blocks are made up of a sequence of convolutional layers and attention algorithms that is capable of capturing the local and global dependencies between image features. The Attention-UNet blocks' output is then passed into the WNet decoder, where the segmentation mask is generated. The Architectural diagram of the Attention-WNet is shown below:

![Attention_WNET](https://github.com/user-attachments/assets/6557c2f6-d4b7-435e-8b1b-f86e6b95f45a)


## SWIN-WNet:
In this model, the WNet architecture is used as the base model and the UNet part of the WNet is replaced with Swin-UNet Architecture. Swin-UNet is a variation of the UNet architecture which is used to improve the segmentation results. Swin-UNet is a variation of the Swin Transformer architecture that has been adapted for image segmentation tasks. The Swin-UNet architecture is similar to the UNet architecture, which comprises of a layered method for extracting features that gradually improves the features of the input image. The Transformer blocks are used at different levels in the structure to capture long-range dependencies between image features, while UNet-style skip connections are utilized to maintain the spatial information of the image.

Swin-UNet blocks are introduced to the WNet encoder to improve feature extraction. Swin-UNet blocks can capture long-term dependencies between image features, and UNet-style skip connections can enable to retain the image's spatial features. This can help improve segmentation results, especially in the case of bifurcation or cross-over points.
The Swin-UNet blocks' output is then passed into the WNet decoder, which generates the segmentation results. The skip connections allow the WNet decoder to combine information from the encoder with features from the respective decoder block. This helps in the improvement of the input image's features and the generation of the segmentation mask. The Architectural diagram of the Swin-WNet is shown below:

![SWIN-WNET](https://github.com/user-attachments/assets/93bca168-5b48-4de8-b854-385c79187a0c)

## Datasets used: 
- DRIVE: https://drive.grand-challenge.org
- HRF: https://www5.cs.fau.de/research/data/fundus-images/




