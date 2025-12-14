import os, glob, random
import numpy as np
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split


# =========================
# 0) Reproducibility
# =========================
def seed_everything(seed=42):
    """
    Fix all random seeds to ensure reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

seed_everything(42)

# =========================
# 1) Pairing: intensity <-> binary mask
# =========================
def list_pairs_same_name(images_dir, masks_dir):
    """
    Pair intensity images and binary masks using identical file names.
    """

    def collect_pngs(root):
        # Recursively collect png files (case-insensitive)
        paths = glob.glob(os.path.join(root, "**", "*.png"), recursive=True)
        paths += glob.glob(os.path.join(root, "**", "*.PNG"), recursive=True)
        return sorted([p for p in paths if os.path.isfile(p)])

    img_paths = collect_pngs(images_dir)
    msk_paths = collect_pngs(masks_dir)

    # Use file name (with extension) as key, e.g. "1.png"
    img_map = {os.path.basename(p): p for p in img_paths}
    msk_map = {os.path.basename(p): p for p in msk_paths}

    common = sorted(set(img_map.keys()) & set(msk_map.keys()))
    pairs = [(img_map[name], msk_map[name]) for name in common]

    print(f"[DEBUG] images found: {len(img_paths)} | masks found: {len(msk_paths)} | paired: {len(pairs)}")

    if len(pairs) == 0:
        print("[DEBUG] example image names:", list(img_map.keys())[:10])
        print("[DEBUG] example mask  names:", list(msk_map.keys())[:10])
        print("[DEBUG] image-only examples:", list(set(img_map) - set(msk_map))[:10])
        print("[DEBUG] mask-only  examples:", list(set(msk_map) - set(img_map))[:10])

    return pairs


# =========================
# 2) Dataset
# =========================
class SpineSegDataset(Dataset):
    """
    Dataset for dendritic spine segmentation.
    Returns (image, mask) pairs.
    """

    def __init__(self, pairs, resize_to=(250, 250), augment=False):
        self.pairs = pairs
        self.resize_to = resize_to
        self.augment = augment

    def __len__(self):
        return len(self.pairs)

    def _augment(self, img, mask):
        """
        Apply simple geometric and intensity augmentations.
        """

        # Horizontal flip
        if random.random() < 0.5:
            img = np.fliplr(img).copy()
            mask = np.fliplr(mask).copy()

        # Vertical flip
        if random.random() < 0.5:
            img = np.flipud(img).copy()
            mask = np.flipud(mask).copy()

        # Random 90-degree rotations
        k = random.randint(0, 3)
        if k:
            img = np.rot90(img, k).copy()
            mask = np.rot90(mask, k).copy()

        # Mild brightness / contrast perturbation (image only)
        if random.random() < 0.5:
            alpha = 1.0 + (random.random() - 0.5) * 0.2  # contrast: 0.9 ~ 1.1
            beta  = (random.random() - 0.5) * 0.08       # brightness shift
            img = np.clip(img * alpha + beta, 0, 1)

        return img, mask

    def __getitem__(self, idx):
        img_path, mask_path = self.pairs[idx]

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        msk = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if img is None or msk is None:
            raise RuntimeError(f"Cannot read: {img_path} or {mask_path}")

        # Resize (use nearest-neighbor for mask)
        img = cv2.resize(img, self.resize_to, interpolation=cv2.INTER_AREA)
        msk = cv2.resize(msk, self.resize_to, interpolation=cv2.INTER_NEAREST)

        # Normalize
        img = img.astype(np.float32) / 255.0
        msk = (msk > 127).astype(np.float32)  # binary {0,1}

        if self.augment:
            img, msk = self._augment(img, msk)

        # Shape: (1, H, W)
        img = torch.from_numpy(img[None, ...])
        msk = torch.from_numpy(msk[None, ...])
        return img, msk


# =========================
# 3) Tiny U-Net
# =========================
class DoubleConv(nn.Module):
    """
    Two consecutive convolution layers with BatchNorm and ReLU.
    """
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class TinyUNet(nn.Module):
    """
    Lightweight U-Net architecture for binary segmentation.
    """
    def __init__(self, in_ch=1, base=16):
        super().__init__()

        self.enc1 = DoubleConv(in_ch, base)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = DoubleConv(base, base * 2)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = DoubleConv(base * 2, base * 4)

        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.dec2 = DoubleConv(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec1 = DoubleConv(base * 2, base)

        self.out = nn.Conv2d(base, 1, 1)

    def _match_size(self, src, ref):
        """
        Match spatial size of src tensor to ref tensor
        using padding or cropping if necessary.
        """
        _, _, h, w = src.shape
        _, _, H, W = ref.shape

        pad_h = max(0, H - h)
        pad_w = max(0, W - w)
        if pad_h > 0 or pad_w > 0:
            src = nn.functional.pad(src, (0, pad_w, 0, pad_h))

        src = src[:, :, :H, :W]
        return src

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))

        d2 = self.up2(e3)
        e2m = self._match_size(e2, d2)
        d2 = torch.cat([d2, e2m], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        e1m = self._match_size(e1, d1)
        d1 = torch.cat([d1, e1m], dim=1)
        d1 = self.dec1(d1)

        out = self.out(d1)
        out = self._match_size(out, x)  # restore input resolution
        return out


# =========================
# 4) Loss & Metric
# =========================
def dice_coef(prob, target, eps=1e-6):
    """
    Dice coefficient for binary segmentation.
    """
    prob = prob.contiguous()
    target = target.contiguous()
    inter = (prob * target).sum(dim=(2, 3))
    union = prob.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    return ((2 * inter + eps) / (union + eps)).mean()


class DiceLoss(nn.Module):
    def forward(self, logits, target):
        prob = torch.sigmoid(logits)
        return 1.0 - dice_coef(prob, target)


def loss_fn(logits, target):
    """
    Combined BCE + Dice loss.
    """
    bce = nn.BCEWithLogitsLoss()(logits, target)
    dice = DiceLoss()(logits, target)
    return 0.5 * bce + 0.5 * dice


# =========================
# 5) Train / Evaluation
# =========================
@torch.no_grad()
def eval_one_epoch(model, loader, device):
    """
    Evaluate model for one epoch.
    """
    model.eval()
    losses, dices = [], []

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = loss_fn(logits, y).item()
        prob = torch.sigmoid(logits)
        dice = dice_coef((prob > 0.5).float(), y).item()
        losses.append(loss)
        dices.append(dice)

    return float(np.mean(losses)), float(np.mean(dices))


def train_one_epoch(model, loader, opt, device):
    """
    Train model for one epoch.
    """
    model.train()
    losses, dices = [], []

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        logits = model(x)
        loss = loss_fn(logits, y)
        loss.backward()
        opt.step()

        with torch.no_grad():
            prob = torch.sigmoid(logits)
            dice = dice_coef((prob > 0.5).float(), y).item()

        losses.append(loss.item())
        dices.append(dice)

    return float(np.mean(losses)), float(np.mean(dices))


# =========================
# 6) Main Training Pipeline
# =========================
def main():
    """
    Main training entry.
    """

    images_dir = "DATA/Dataset_MaxIntensityProjections/Dataset Intensity"
    masks_dir  = "DATA/Dataset_Annotations/Dataset Binary"
    resize_to  = (250, 250)

    out_best = "best_spine_unet_250.pth"
    out_last = "last_spine_unet_250.pth"

    pairs = list_pairs_by_numeric_id(images_dir, masks_dir)
    if len(pairs) == 0:
        raise RuntimeError("No paired images found.")

    idx = np.arange(len(pairs))
    train_idx, temp_idx = train_test_split(idx, test_size=0.30, random_state=42)
    val_idx, test_idx   = train_test_split(temp_idx, test_size=0.50, random_state=42)

    train_pairs = [pairs[i] for i in train_idx]
    val_pairs   = [pairs[i] for i in val_idx]
    test_pairs  = [pairs[i] for i in test_idx]

    train_ds = SpineSegDataset(train_pairs, resize_to=resize_to, augment=True)
    val_ds   = SpineSegDataset(val_pairs,   resize_to=resize_to, augment=False)
    test_ds  = SpineSegDataset(test_pairs,  resize_to=resize_to, augment=False)

    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=4, shuffle=False)
    test_loader  = DataLoader(test_ds, batch_size=4, shuffle=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("[INFO] Device:", device)

    model = TinyUNet(in_ch=1, base=16).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

    history = {"train_loss": [], "train_dice": [], "val_loss": [], "val_dice": []}

    best_val_dice = -1.0
    best_state = None

    num_epochs = 40
    for epoch in range(1, num_epochs + 1):
        tr_loss, tr_dice = train_one_epoch(model, train_loader, opt, device)
        va_loss, va_dice = eval_one_epoch(model, val_loader, device)

        history["train_loss"].append(tr_loss)
        history["train_dice"].append(tr_dice)
        history["val_loss"].append(va_loss)
        history["val_dice"].append(va_dice)

        print(f"Epoch {epoch:02d}/{num_epochs} | "
              f"train loss {tr_loss:.4f} dice {tr_dice:.4f} | "
              f"val loss {va_loss:.4f} dice {va_dice:.4f}")

        if va_dice > best_val_dice:
            best_val_dice = va_dice
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            torch.save(best_state, out_best)
            print(f"  [Saved BEST] {out_best} (val_dice={best_val_dice:.4f})")

    torch.save(model.state_dict(), out_last)
    print(f"[Saved LAST] {out_last}")

    if best_state is not None:
        model.load_state_dict(best_state)

    te_loss, te_dice = eval_one_epoch(model, test_loader, device)
    print(f"[TEST] loss {te_loss:.4f} dice {te_dice:.4f}")

    np.save("train_history_unet_250.npy", history)
    print("[Saved] train_history_unet_250.npy")


if __name__ == "__main__":
    main()



# =========================
# Resume / Continue Training
# =========================
def resume_training(
    images_dir="DATA/Dataset_MaxIntensityProjections/Dataset Intensity",
    masks_dir ="DATA/Dataset_Annotations/Dataset Binary",
    resize_to=(250, 250),
    ckpt_in="best_spine_unet_250.pth",          # previously trained best checkpoint
    ckpt_best_out="best_spine_unet_250_cont.pth",
    ckpt_last_out="last_spine_unet_250_cont.pth",
    history_in="train_history_unet_250.npy",    # optional: previous training history
    extra_epochs=20,
    batch_size=4,
    lr=1e-4                                     # fine-tuning learning rate
):
    import os
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    from sklearn.model_selection import train_test_split

    # ========= 1) Sanity checks =========
    assert os.path.isfile(ckpt_in), f"Checkpoint not found: {ckpt_in}"
    print("[INFO] Resuming training from:", ckpt_in)

    # ========= 2) Pair intensity images and binary masks =========
    # This must be identical to the pairing logic used in the main training script
    pairs = list_pairs_by_numeric_id(images_dir, masks_dir)
    print("[INFO] Total paired samples:", len(pairs))
    if len(pairs) == 0:
        raise RuntimeError("No valid image-mask pairs found. Please check input directories.")

    # ========= 3) Train / validation / test split =========
    # The random_state must remain unchanged to ensure consistency
    idx = np.arange(len(pairs))
    train_idx, temp_idx = train_test_split(idx, test_size=0.30, random_state=42)
    val_idx, test_idx   = train_test_split(temp_idx, test_size=0.50, random_state=42)

    train_pairs = [pairs[i] for i in train_idx]
    val_pairs   = [pairs[i] for i in val_idx]
    test_pairs  = [pairs[i] for i in test_idx]

    # ========= 4) Dataset and DataLoader =========
    # Must be consistent with the original training configuration
    train_ds = SpineSegDataset(train_pairs, resize_to=resize_to, augment=True)
    val_ds   = SpineSegDataset(val_pairs,   resize_to=resize_to, augment=False)
    test_ds  = SpineSegDataset(test_pairs,  resize_to=resize_to, augment=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=0)

    # ========= 5) Model =========
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("[INFO] Device:", device)

    model = TinyUNet(in_ch=1, base=16).to(device)
    model.load_state_dict(torch.load(ckpt_in, map_location=device))
    print("[INFO] Checkpoint weights loaded")

    # ========= 6) Optimizer =========
    # Use a smaller learning rate for fine-tuning
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    # ========= 7) Load training history =========
    history = {
        "train_loss": [], "train_dice": [],
        "val_loss": [],   "val_dice": []
    }

    if history_in and os.path.isfile(history_in):
        old = np.load(history_in, allow_pickle=True).item()
        for k in history:
            if k in old:
                history[k] = list(old[k])
        print(f"[INFO] Loaded training history from {history_in} "
              f"(epochs completed: {len(history['train_loss'])})")

    best_val_dice = max(history["val_dice"]) if history["val_dice"] else -1.0

    # ========= 8) Continue training =========
    for epoch in range(1, extra_epochs + 1):
        tr_loss, tr_dice = train_one_epoch(model, train_loader, opt, device)
        va_loss, va_dice = eval_one_epoch(model, val_loader, device)

        history["train_loss"].append(tr_loss)
        history["train_dice"].append(tr_dice)
        history["val_loss"].append(va_loss)
        history["val_dice"].append(va_dice)

        print(f"[CONT] Epoch {epoch:02d}/{extra_epochs} | "
              f"train loss {tr_loss:.4f} dice {tr_dice:.4f} | "
              f"val loss {va_loss:.4f} dice {va_dice:.4f}")

        if va_dice > best_val_dice:
            best_val_dice = va_dice
            torch.save(model.state_dict(), ckpt_best_out)
            print(f"[INFO] Best model updated and saved to {ckpt_best_out} "
                  f"(val_dice={best_val_dice:.4f})")

    # ========= 9) Save final checkpoint =========
    torch.save(model.state_dict(), ckpt_last_out)
    print(f"[INFO] Last checkpoint saved to {ckpt_last_out}")

    # ========= 10) Test evaluation =========
    model.load_state_dict(torch.load(ckpt_best_out, map_location=device))
    te_loss, te_dice = eval_one_epoch(model, test_loader, device)
    print(f"[TEST] loss {te_loss:.4f} dice {te_dice:.4f}")

    np.save(history_in, history)
    print(f"[INFO] Training history saved to {history_in}")

    return history, best_val_dice


history_cont, best_cont = resume_training(
    ckpt_in="best_spine_unet_250.pth",
    extra_epochs=20,
    lr=1e-4
)

print("Best val dice after resume:", best_cont)

history_cont, best_cont = resume_training(
    ckpt_in="best_spine_unet_250_cont.pth",
    ckpt_best_out="best_spine_unet_250_cont_1.pth",
    ckpt_last_out="last_spine_unet_250_cont_1.pth",
    extra_epochs=60,
    lr=1e-4
)

print("Best val dice after resume:", best_cont)

# =========================
# Single Image → Binary Mask
# =========================
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt

# ====== 你改这里 ======
IMG_PATH  = "DATA/Dataset_MaxIntensityProjections/Dataset Intensity/1.png"         # 一张 intensity 图
CKPT_PATH = "best_spine_unet_250_cont_1.pth" # 你训练好的模型
OUT_MASK  = "example_binary_mask.png"
IMG_SIZE  = (250, 250)
THRESHOLD = 0.55   # 你刚刚扫描得到的 best threshold
# ======================

device = "cuda" if torch.cuda.is_available() else "cpu"

# ---------- 1. Load model ----------
model = TinyUNet(in_ch=1, base=16).to(device)
model.load_state_dict(torch.load(CKPT_PATH, map_location=device))
model.eval()
print("[INFO] Model loaded")

# ---------- 2. Load & preprocess image ----------
img = cv2.imread(IMG_PATH, cv2.IMREAD_GRAYSCALE)
if img is None:
    raise RuntimeError(f"Cannot read image: {IMG_PATH}")

img_resized = cv2.resize(img, IMG_SIZE, interpolation=cv2.INTER_LINEAR)
img_norm = img_resized.astype(np.float32) / 255.0   # 与训练一致
x = torch.from_numpy(img_norm)[None, None, ...].to(device)  # (1,1,H,W)

# ---------- 3. Forward ----------
with torch.no_grad():
    logits = model(x)
    prob = torch.sigmoid(logits)[0,0].cpu().numpy()  # (H,W)

# ---------- 4. Threshold ----------
binary = (prob >= THRESHOLD).astype(np.uint8) * 255

# ---------- 5. Save ----------
cv2.imwrite(OUT_MASK, binary)
print(f"[SAVED] binary mask -> {OUT_MASK}")

# ---------- 6. Visualization ----------
plt.figure(figsize=(12,4))

plt.subplot(1,3,1)
plt.title("Intensity")
plt.imshow(img_resized, cmap="gray")
plt.axis("off")

plt.subplot(1,3,2)
plt.title("Predicted Probability")
plt.imshow(prob, cmap="viridis")
plt.colorbar(fraction=0.046)
plt.axis("off")

plt.subplot(1,3,3)
plt.title(f"Binary Mask (t={THRESHOLD})")
plt.imshow(binary, cmap="gray")
plt.axis("off")

plt.tight_layout()
plt.show()


#     images_dir = "DATA/Dataset_MaxIntensityProjections/Dataset Intensity"
#    masks_dir  = "DATA/Dataset_Annotations/Dataset Binary"