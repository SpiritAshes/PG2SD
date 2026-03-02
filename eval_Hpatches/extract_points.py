import os
import yaml
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as tvf
from model.model import MergedNet

RGB_mean = [0.485, 0.456, 0.406]
RGB_std  = [0.229, 0.224, 0.225]
norm_RGB = tvf.Compose([tvf.ToTensor(), tvf.Normalize(mean=RGB_mean, std=RGB_std)])

# torch.backends.cuda.matmul.allow_tf32 = False
# torch.backends.cudnn.allow_tf32 = False

class NonMaxSuppression(nn.Module):
    def __init__(self, rel_thr=0.7, rep_thr=0.7):
        super().__init__()
        self.max_filter = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.rel_thr = rel_thr
        self.rep_thr = rep_thr

    def forward(self, reliability, repeatability):
        assert len(reliability) == len(repeatability) == 1

        maxima = (repeatability == self.max_filter(repeatability))
        maxima &= (repeatability >= self.rep_thr)
        maxima &= (reliability >= self.rel_thr)

        return maxima.nonzero().t()[2:4]


def extract_multiscale(net, img, detector,
                       scale_f=2**0.25,
                       min_scale=0.0,
                       max_scale=1,
                       min_size=256,
                       max_size=1024,
                       verbose=False):

    old_bm = torch.backends.cudnn.benchmark 
    torch.backends.cudnn.benchmark = False # speedup

    B, C, H, W = img.shape
    assert B == 1 and C == 3

    assert max_scale <= 1
    s = 1.0

    X, Y, S, Cc, Q, D = [], [], [], [], [], []

    while s + 0.001 >= max(min_scale, min_size / max(H, W)):

        if s - 0.001 <= min(max_scale, max_size / max(H, W)):

            nh, nw = img.shape[2:]

            if verbose:
                print(f"Extracting at scale x{s:.02f} = {nw}x{nh}")

            with torch.no_grad():
                descriptors, repeatability, reliability = net(img)

            y, x = detector(reliability, repeatability)

            c = reliability[0, 0, y, x]
            q = repeatability[0, 0, y, x]
            d = descriptors[0, :, y, x].t()
            n = d.shape[0]

            X.append(x.float() * W / nw)
            Y.append(y.float() * H / nh)
            S.append((32 / s) * torch.ones(n, device=d.device))
            Cc.append(c)
            Q.append(q)
            D.append(d)

        s /= scale_f
        nh, nw = round(H * s), round(W * s)
        img = F.interpolate(img, (nh, nw),
                            mode='bilinear',
                            align_corners=False)

    torch.backends.cudnn.benchmark = old_bm

    Y = torch.cat(Y)
    X = torch.cat(X)
    S = torch.cat(S)
    scores = torch.cat(Cc) * torch.cat(Q)
    XYS = torch.stack([X, Y, S], dim=-1)
    D = torch.cat(D)

    return XYS, D, scores


def extract_keypoints(config):

    # ========= Device =========
    if config["gpu"][0] != -1:
        device = torch.device(f"cuda:{config['gpu'][0]}")
    else:
        device = torch.device("cpu")

    # torch.backends.cudnn.benchmark = False

    # ========= Load Model =========
    model = MergedNet().to(device)

    if len(config["gpu"]) > 1:
        model = torch.nn.DataParallel(model,
                                      device_ids=config["gpu"])

    if config["model"]["pretrained"]:
        checkpoint = torch.load(config["model"]["pretrained_path"])
        model.load_state_dict(checkpoint["state_dict"])

    model.eval()
    
    # ========= NMS =========
    detector = NonMaxSuppression(
        rel_thr=config["extract"]["reliability_thr"],
        rep_thr=config["extract"]["repeatability_thr"]
    )

    # ========= Image List =========
    images_path = []

    if config["extract"]["images"].endswith(".txt"):
        with open(config["extract"]["images"], "r") as f:
            lines = f.read().splitlines()

        for line in lines:
            image_path = os.path.join(
                os.path.dirname(config["extract"]["images"]),
                line.strip()
            )
            images_path.append(image_path)
    else:
        images_path = [config["extract"]["images"]]


    while images_path:

        img_path = images_path.pop(0)
        print(f"\nExtracting features for {img_path}")

        img = Image.open(img_path).convert("RGB")
        W, H = img.size
        # W = int(W / 224) * 224
        # H = int(H / 224) * 224
        
        # img = img.resize((W, H))

        img = norm_RGB(img)[None].to(device)

        xys, desc, scores = extract_multiscale(
            model, img, detector,
            scale_f=config["extract"]["scale_f"],
            min_scale=config["extract"]["min_scale"],
            max_scale=config["extract"]["max_scale"],
            min_size=config["extract"]["min_size"],
            max_size=config["extract"]["max_size"],
            verbose=True
        )

        xys = xys.cpu().numpy()
        desc = desc.cpu().numpy()
        scores = scores.cpu().numpy()

        top_k = config["extract"]["top_k"]
        idxs = scores.argsort()[-top_k or None:]

        outpath = img_path + "." + config["extract"]["tag"]
        print(f"Saving {len(idxs)} keypoints to {outpath}")

        np.savez(open(outpath, "wb"),
                 imsize=(W, H),
                 keypoints=xys[idxs],
                 descriptors=desc[idxs],
                 scores=scores[idxs])


if __name__ == "__main__":

    with open('../config/extract.yaml', "r") as f:
        config = yaml.safe_load(f)

    extract_keypoints(config)