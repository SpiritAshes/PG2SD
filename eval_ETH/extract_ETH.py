import os
import glob
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import torchvision.transforms as tvf
from model.model import MergedNet

RGB_mean = [0.485, 0.456, 0.406]
RGB_std  = [0.229, 0.224, 0.225]
norm_RGB = tvf.Compose([tvf.ToTensor(), tvf.Normalize(mean=RGB_mean, std=RGB_std)])


# =========================================================
# Non Max Suppression
# =========================================================
class NonMaxSuppression(nn.Module):
    def __init__(self, rel_thr=0.7, rep_thr=0.7):
        super().__init__()
        self.max_filter = nn.MaxPool2d(3, 1, 1)
        self.rel_thr = rel_thr
        self.rep_thr = rep_thr

    def forward(self, reliability, repeatability):

        maxima = repeatability == self.max_filter(repeatability)
        maxima &= repeatability >= self.rep_thr
        maxima &= reliability >= self.rel_thr

        return maxima.nonzero(as_tuple=False).t()[2:4]


# =========================================================
# Multi-scale Extraction
# =========================================================
@torch.no_grad()
def extract_multiscale(
    net,
    img,
    detector,
    scale_f,
    min_scale,
    max_scale,
    min_size,
    max_size,
    verbose=False,
):

    B, C, H, W = img.shape
    assert B == 1 and C == 3

    s = 1.0
    X, Y, S, C_list, Q_list, D = [], [], [], [], [], []

    while s + 0.001 >= max(min_scale, min_size / max(H, W)):

        if s - 0.001 <= min(max_scale, max_size / max(H, W)):

            nh, nw = img.shape[2:]
            if verbose:
                print(f"extracting at scale x{s:.02f} = {nw:4d}x{nh:3d}")

            descriptors, repeatability, reliability = net(img)

            y, x = detector(reliability, repeatability)

            if len(x) > 0:
                c = reliability[0, 0, y, x]
                q = repeatability[0, 0, y, x]
                d = descriptors[0, :, y, x].t()
                n = d.shape[0]

                X.append(x.float() * W / nw)
                Y.append(y.float() * H / nh)
                S.append(torch.full((n,), 32 / s, device=d.device))
                C_list.append(c)
                Q_list.append(q)
                D.append(d)

        s /= scale_f
        img = F.interpolate(
            img,
            (round(H * s), round(W * s)),
            mode="bilinear",
            align_corners=False,
        )

    if len(X) == 0:
        return None, None, None

    X = torch.cat(X)
    Y = torch.cat(Y)
    S = torch.cat(S)
    scores = torch.cat(C_list) * torch.cat(Q_list)

    XYS = torch.stack([X, Y, S], dim=-1)
    D = torch.cat(D)

    return XYS, D, scores


# =========================================================
# Main Extraction
# =========================================================
def extract_keypoints(config):

    # ---------------- Device ----------------
    device = torch.device(
        f"cuda:{config['gpu'][0]}"
        if config["gpu"][0] != -1
        else "cpu"
    )

    torch.backends.cudnn.benchmark = False

    # ---------------- Model ----------------
    model = MergedNet().to(device)
    model.eval()

    if config["model"]["pretrained"]:
        checkpoint = torch.load(
            config["model"]["pretrained_path"],
            map_location=device,
        )
        model.load_state_dict(checkpoint["state_dict"])

    # ---------------- Detector ----------------
    detector = NonMaxSuppression(
        rel_thr=config["nms"]["reliability_thr"],
        rep_thr=config["nms"]["repeatability_thr"],
    ).to(device)

    # ---------------- Image List ----------------
    images_root = config["data"]["images"]

    if images_root.endswith(".txt"):
        with open(images_root, "r") as f:
            lines = f.read().splitlines()
        base = os.path.dirname(images_root)
        images_path = [os.path.join(base, l.strip()) for l in lines]
    else:
        images_path = []
        images_path += glob.glob(os.path.join(images_root, "*.jpg"))
        images_path += glob.glob(os.path.join(images_root, "*.png"))
        images_path += glob.glob(os.path.join(images_root, "*.JPG"))

    print("images_num_sum:", len(images_path))

    # ---------------- Extraction Loop ----------------
    for img_path in images_path:

        print(f"\nExtracting features for {img_path}")

        img = Image.open(img_path).convert("RGB")
        W, H = img.size

        img = norm_RGB(img)[None].to(device)

        xys, desc, scores = extract_multiscale(
            model,
            img,
            detector,
            scale_f=config["multiscale"]["scale_f"],
            min_scale=config["multiscale"]["min_scale"],
            max_scale=config["multiscale"]["max_scale"],
            min_size=config["multiscale"]["min_size"],
            max_size=config["multiscale"]["max_size"],
            verbose=config["multiscale"]["verbose"],
        )

        if xys is None:
            continue

        xys = xys.cpu().numpy()
        desc = desc.cpu().numpy()
        scores = scores.cpu().numpy()

        top_k = config["output"]["top_k"]
        idxs = scores.argsort()[-top_k or None:]

        outpath = img_path + "." + config["output"]["tag"]
        print(f"Saving {len(idxs)} keypoints to {outpath}")

        np.savez(
            open(outpath, "wb"),
            imsize=(W, H),
            keypoints=xys[idxs],
            descriptors=desc[idxs],
            scores=scores[idxs],
        )


# =========================================================
# Entry
# =========================================================
if __name__ == "__main__":

    with open('../config/extract_ETH.yaml', "r") as f:
        config = yaml.safe_load(f)

    extract_keypoints(config)