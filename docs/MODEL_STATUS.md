# Model status and evaluation limits

## Source provenance

This update was prepared from the 14 October 2026 local source audit and
`AVsegmentation_source_bundle.zip` supplied by the repository owner.

| Published module | Handoff source | Changes |
| --- | --- | --- |
| `pytorch/avseg/attention_wnet.py` | `experiments/Att_Wnet_V11/models.py` | Model code preserved byte-for-byte for V11 key compatibility |
| `pytorch/avseg/attention_unet_ds.py` | `archived_sources/avs_ecosystem_kaggle_exp001/models/attention_unet.py` | Model code preserved; published as a separate architecture |
| Preprocessing, tiling, predictor, auxiliary loss helper | New integration code | Explicit inference inputs, strict loading, prediction-only gating, testable numerical contracts |

V12 source matches V11 in the audit; a newer run name does not define a newer
architecture. V11 Attention-WNet supervises two cascade outputs and contains no
auxiliary decoder prediction heads. The separate Attention-U-Net has a real
intermediate head; no associated trained weights were supplied.

## Checkpoint compatibility

The audit reports a successful strict load and short CPU forward pass of the
local V11 checkpoint, and exact output equivalence with a consolidated module
after key remapping. Those were auditor checks on Windows, not checkpoint
evaluations performed by this package's authors during this update.

The new predictor retains original V11 keys and strips only a leading
`module.` DataParallel prefix. It loads with `weights_only=True` and
`strict=True`. It accepts either a raw state dictionary or a checkpoint with
`state_dict`. Four-channel V10, legacy 3/3-output, and Swin checkpoints are not
compatible with its fixed six-channel 2/3-output model.

## Preprocessing and inference

For green-only patch-based V11 input, preprocessing occurs on the full RGB
image: historical conditional denoising, green CLAHE (clip 2/grid 8), OpenCV
encoded LAB, division of image channels by 255, and global coordinates y,x in
[0,1]. Coordinates are created before patches. Original image dimensions are
preserved. The historical denoiser array convention is preserved to avoid an
unannounced input change relative to V11 training.

Small images are padded with NumPy reflection for inference, then cropped back
to their original dimensions. This is a new inference padding implementation;
it is not a certification of identical historical torchvision edge handling.

V11 model outputs are already probabilities. Applying softmax again before
patch averaging changes those values. The new blending path preserves them and
requires normalized finite inputs. Final A/V output is gated by predicted
vessels. Ground-truth masks do not enter the predictor.

## Historical metrics

The audit found 305 training images and 108 images used as validation in the
V11 configuration. Nominal test folders were used for early stopping and
checkpoint selection. Such results must be labeled validation results until
an independent test evaluation is available. A lack of byte-identical image
overlap does not prove independence of patients, eyes, crops or augmentations.

Background-dominated full-image accuracy, centerline classification F1, and
end-to-end segmentation Dice measure different tasks. Do not collapse them into
an unspecified percentage improvement or a SOTA claim. Historical JSON metric
array ordering also differs between scripts; no old score table is republished
as a verified benchmark here.

## Deferred components

- Swin V6 training passes logits into probability-based losses; its PyTorch
  training/evaluation path needs separate correction before promotion.
- The archived deep-supervision training ecosystem needs independent splits,
  consistent predicted/GT vessel-prior handling, correct partial gradient
  accumulation, and resolution-aware evaluation.
- Local GNN, optic-disc, and AVR modules are not part of this release.
- Datasets, weights, historical logs, and local memory files are not added by
  this update. Existing public dataset files are untouched.

## Validation scope

Unit checks use synthetic inputs. They verify software contracts, not trained
accuracy, clinical suitability, production throughput or reproducibility of
historical scores. A real-checkpoint inference comparison remains required.
