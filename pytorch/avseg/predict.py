"""Image-only inference for V11-compatible checkpoints."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from .preprocess import v11_input, DENOISED_DATASETS
from .tiling import sliding_window, decode


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda:f.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_model(checkpoint, device):
    import torch
    from .attention_wnet import AttentionWNet
    checkpoint = Path(checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state = saved.get("state_dict", saved)
    state = {k.removeprefix("module."):v for k,v in state.items()}
    model = AttentionWNet(in_channels=6, out_channels1=2, out_channels2=3)
    # Retain V11 module names; fail on incompatible channels, classes or widths.
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--dataset", required=True,
                        help="Dataset name controls historical V11 denoising; use custom for new images")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--patch-size", type=int, default=512)
    parser.add_argument("--overlap", type=float, default=0.25)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    if not args.checkpoint.is_file():
        parser.error(f"Checkpoint not found: {args.checkpoint}")
    if not args.image.is_file():
        parser.error(f"Image not found: {args.image}")
    import cv2
    import torch
    model = load_model(args.checkpoint, args.device)
    bgr = cv2.imread(str(args.image))
    if bgr is None:
        parser.error("Could not decode input image")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    image = v11_input(rgb, args.dataset)
    with torch.inference_mode():
        def predict_patch(patch):
            x = torch.from_numpy(np.ascontiguousarray(patch))[None].to(args.device)
            return tuple(o[0].cpu().numpy() for o in model(x))
        probs = sliding_window(image, predict_patch, args.patch_size, args.overlap)
    vessels, av = decode(*probs)
    colors = np.array([[0,0,0], [255,0,0], [0,0,255]], dtype=np.uint8)
    palette = colors[av]
    overlay = rgb.copy()
    selected = av > 0
    overlay[selected] = (rgb[selected]*0.5 + palette[selected]*0.5).astype(np.uint8)
    args.output.mkdir(parents=True, exist_ok=True)
    files = {"vessels.png":vessels*255, "av_labels.png":av,
             "av_rgb.png":cv2.cvtColor(palette,cv2.COLOR_RGB2BGR),
             "overlay.png":cv2.cvtColor(overlay,cv2.COLOR_RGB2BGR)}
    for name, array in files.items():
        if not cv2.imwrite(str(args.output/name), array):
            raise OSError(f"Could not write {name}")
    manifest = {"model":"V11 Attention-WNet", "checkpoint_sha256":file_hash(args.checkpoint),
                "image_sha256":file_hash(args.image), "dataset":args.dataset,
                "denoising":args.dataset in DENOISED_DATASETS,
                "channels":["green_clahe","LAB_L","LAB_A","LAB_B","global_y","global_x"],
                "patch_size":args.patch_size,"overlap":args.overlap,
                "torch_version":torch.__version__,"numpy_version":np.__version__,
                "opencv_version":cv2.__version__,"ground_truth_used":False,
                "labels":{"0":"background","1":"artery","2":"vein"}}
    (args.output/"inference.json").write_text(json.dumps(manifest,indent=2)+"\n")
    print(f"Saved full-resolution predictions to {args.output}")


if __name__ == "__main__":
    main()
