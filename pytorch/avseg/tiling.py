"""Blend already-normalized cascade probabilities over a full image."""
import numpy as np


def starts(length, patch_size, stride):
    if length < patch_size or stride < 1 or stride > patch_size:
        raise ValueError("Invalid patch coverage")
    positions = list(range(0, length - patch_size + 1, stride))
    if positions[-1] != length - patch_size:
        positions.append(length - patch_size)
    return positions


def sliding_window(image, predict_patch, patch_size=512, overlap=0.25):
    """CHW -> two CHW probability arrays. Callback returns [2,P,P], [3,P,P].

    Coordinate channels must be generated globally BEFORE this function.
    No labels or ground-truth vessel masks enter this prediction path.
    """
    if image.ndim != 3 or min(image.shape) < 1:
        raise ValueError("Expected nonempty CHW image")
    if patch_size < 16 or patch_size % 16 or not 0 <= overlap < 1:
        raise ValueError("Patch size must be a positive multiple of 16; overlap in [0,1)")
    stride = max(1, int(patch_size * (1 - overlap)))
    _, h, w = image.shape
    padded = np.pad(image, ((0, 0), (0, max(0, patch_size-h)),
                          (0, max(0, patch_size-w))), mode="reflect")
    ph, pw = padded.shape[-2:]
    axis = np.linspace(-1, 1, patch_size)
    yy, xx = np.meshgrid(axis, axis, indexing="ij")
    weights = np.exp(-(xx**2 + yy**2) / 0.5).astype(np.float32)[None]
    sums = [np.zeros((c, ph, pw), np.float32) for c in (2, 3)]
    counts = np.zeros((1, ph, pw), np.float32)
    for y in starts(ph, patch_size, stride):
        for x in starts(pw, patch_size, stride):
            outputs = predict_patch(padded[:, y:y+patch_size, x:x+patch_size])
            if len(outputs) != 2:
                raise ValueError("Expected both cascade stages")
            for i, (output, classes) in enumerate(zip(outputs, (2, 3))):
                output = np.asarray(output)
                if output.shape != (classes, patch_size, patch_size):
                    raise ValueError(f"Unexpected stage-{i+1} shape: {output.shape}")
                if (not np.isfinite(output).all() or (output < 0).any()
                        or (output > 1).any()
                        or not np.allclose(output.sum(0), 1, atol=1e-4)):
                    raise ValueError("Expected normalized probabilities, not logits")
                sums[i][:, y:y+patch_size, x:x+patch_size] += output * weights
            counts[:, y:y+patch_size, x:x+patch_size] += weights
    if (counts == 0).any():
        raise RuntimeError("Uncovered pixels")
    return tuple((s / counts)[:, :h, :w] for s in sums)


def decode(vessel_probs, av_probs):
    """Use the PREDICTED vessel class to gate final A/V labels."""
    if vessel_probs.shape[0] != 2 or av_probs.shape[0] != 3:
        raise ValueError("Expected binary vessels and three A/V classes")
    if vessel_probs.shape[1:] != av_probs.shape[1:]:
        raise ValueError("Spatial shapes must match")
    vessels = vessel_probs.argmax(0).astype(np.uint8)
    av = av_probs.argmax(0).astype(np.uint8) * vessels
    return vessels, av
