"""Reproduce the V11 six-channel inference inputs without labels."""
import numpy as np

DENOISED_DATASETS = {"RITE", "RAV", "Fundus-AVSeg"}


def coordinates(h, w):
    """Global y,x in [0,1], matching V11 linspace ordering."""
    y = np.broadcast_to(np.linspace(0, 1, h, dtype=np.float32)[:, None], (h,w))
    x = np.broadcast_to(np.linspace(0, 1, w, dtype=np.float32)[None], (h,w))
    return np.stack((y, x))


def v11_input(rgb, dataset):
    import cv2
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise ValueError("Expected uint8 RGB image")
    if dataset in DENOISED_DATASETS:
        # Preserve the original V11 code's array/channel convention.
        rgb = cv2.fastNlMeansDenoisingColored(rgb, None, 10, 10, 7, 21)
    green = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(rgb[:,:,1])
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    image = np.concatenate((green[None], lab.transpose(2,0,1))).astype(np.float32)/255
    return np.concatenate((image, coordinates(*rgb.shape[:2])))
