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



