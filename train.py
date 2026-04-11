import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from model import ShapeWaveNet

# ── Config ────────────────────────────────────────────────────────────────────
QUICKDRAW_SAMPLES  = 10000  # per class
SYNTHETIC_SAMPLES  = 5000   # per class
BATCH_SIZE         = 64
EPOCHS             = 30
LEARNING_RATE      = 0.001
VAL_SPLIT          = 0.8
SAVE_PATH          = "shapewavenet.pth"
# ──────────────────────────────────────────────────────────────────────────────

CLASSES = {
    0: "square",
    1: "triangle",
    2: "circle"
}


def generate_synthetic_data(num_samples: int = SYNTHETIC_SAMPLES):
    """
    Generates clean synthetic shapes using OpenCV.
    Includes random size, position, thickness, and rotation augmentation.
    """
    images, labels = [], []
    print(f"[SYNTHETIC] Generating {num_samples} samples per class...")

    for label in range(3):
        count = 0
        while count < num_samples:
            img = np.zeros((64, 64), dtype=np.uint8)
            center_x = np.random.randint(20, 44)
            center_y = np.random.randint(20, 44)
            size      = np.random.randint(10, 22)
            thickness = np.random.randint(1, 5)  # wider range for sketchpad robustness

            if label == 0:  # Square
                cv2.rectangle(
                    img,
                    (center_x - size, center_y - size),
                    (center_x + size, center_y + size),
                    255, thickness
                )
            elif label == 1:  # Triangle
                pts = np.array([
                    [center_x, center_y - size],
                    [center_x - size, center_y + size],
                    [center_x + size, center_y + size]
                ], np.int32)
                cv2.polylines(img, [pts], True, 255, thickness)
            else:  # Circle
                cv2.circle(img, (center_x, center_y), size, 255, thickness)

            # Random rotation (full 360 for squares/circles, limited for triangles)
            angle = np.random.randint(0, 360) if label != 1 else np.random.randint(-30, 30)
            M = cv2.getRotationMatrix2D((32, 32), angle, 1.0)
            img = cv2.warpAffine(img, M, (64, 64), borderMode=cv2.BORDER_CONSTANT, borderValue=0)

            # Random Gaussian noise
            if np.random.random() < 0.3:
                noise = np.random.normal(0, 8, img.shape).astype(np.int16)
                img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

            # Random blur to simulate sketchpad strokes
            if np.random.random() < 0.4:
                k = np.random.choice([3, 5])
                img = cv2.GaussianBlur(img, (k, k), 0)

            images.append(img)
            labels.append(label)
            count += 1

        print(f"[SYNTHETIC] Generated {count} '{CLASSES[label]}' samples.")

    X = torch.tensor(np.array(images)).float().unsqueeze(1) / 255.0
    y = torch.tensor(labels).long()
    print(f"[SYNTHETIC] Shape: {X.shape} | Min: {X.min():.2f} | Max: {X.max():.2f}")
    return X, y


def generate_quickdraw_data(num_samples: int = QUICKDRAW_SAMPLES):
    """
    Downloads hand-drawn shape images from the Quick Draw dataset.
    Applies random rotation and stroke thickness augmentation.
    """
    try:
        from quickdraw import QuickDrawDataGroup
    except ImportError:
        raise ImportError("Install with: pip install quickdraw")

    images, labels = [], []

    for label, name in CLASSES.items():
        print(f"[QUICKDRAW] Downloading '{name}' ({num_samples} samples)...")
        group = QuickDrawDataGroup(name, max_drawings=num_samples, recognized=True)

        count = 0
        for drawing in group.drawings:
            img = np.array(drawing.image.convert("L").resize((64, 64)))
            img = 255 - img  # invert to white-on-black

            # Random rotation
            angle = np.random.randint(0, 360)
            M = cv2.getRotationMatrix2D((32, 32), angle, 1.0)
            img = cv2.warpAffine(img, M, (64, 64), borderMode=cv2.BORDER_CONSTANT, borderValue=0)

            # Random stroke thickness
            thickness_op  = np.random.choice(['thin', 'normal', 'thick'])
            kernel_size   = np.random.randint(2, 5)
            kernel        = np.ones((kernel_size, kernel_size), np.uint8)
            if thickness_op == 'thin':
                img = cv2.erode(img, kernel, iterations=1)
            elif thickness_op == 'thick':
                img = cv2.dilate(img, kernel, iterations=2)

            images.append(img)
            labels.append(label)
            count += 1
            if count >= num_samples:
                break

        print(f"[QUICKDRAW] Loaded {count} '{name}' drawings.")

    X = torch.tensor(np.array(images)).float().unsqueeze(1) / 255.0
    y = torch.tensor(labels).long()
    print(f"[QUICKDRAW] Shape: {X.shape} | Min: {X.min():.2f} | Max: {X.max():.2f}")
    return X, y


def get_loaders(X, y):
    perm = torch.randperm(len(X))
    X, y = X[perm], y[perm]

    split        = int(VAL_SPLIT * len(X))
    train_ds     = TensorDataset(X[:split], y[:split])
    val_ds       = TensorDataset(X[split:], y[split:])
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)

    print(f"[DATA] Train: {len(train_ds)} | Val: {len(val_ds)}")
    return train_loader, val_loader, val_ds


def evaluate(model, val_loader, val_ds, device):
    model.eval()
    correct       = 0
    class_correct = [0, 0, 0]
    class_total   = [0, 0, 0]

    with torch.no_grad():
        for vx, vy in val_loader:
            preds = torch.max(model(vx.to(device)), 1)[1]
            correct += (preds == vy.to(device)).sum().item()
            for t, p in zip(vy, preds.cpu()):
                class_correct[t] += (t == p).item()
                class_total[t]   += 1

    names     = [CLASSES[i] for i in range(3)]
    per_class = " | ".join(
        f"{names[i]}: {100 * class_correct[i] / class_total[i]:.1f}%"
        for i in range(3)
    )
    print(f"         [{per_class}]")
    return correct / len(val_ds) * 100

""" For saving model to Google Drive
def save_model(model, path):
    torch.save(model.state_dict(), path)
    print(f"[SAVED] Model saved to {path}")

    try:
        import google.colab
        from google.colab import files
        files.download(path)
        print(f"[COLAB] Download triggered for {path}")

        try:
            import shutil
            drive_path = f"/content/drive/MyDrive/{path}"
            shutil.copy(path, drive_path)
            print(f"[COLAB] Copied to Google Drive: {drive_path}")
        except Exception:
            print("[COLAB] Google Drive not mounted — skipping Drive copy")

    except ImportError:
        pass
"""

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[DEVICE] Running on: {device}")

    # Generate and combine both datasets
    X_synth, y_synth       = generate_synthetic_data()
    X_qd, y_qd             = generate_quickdraw_data()

    X_all = torch.cat([X_synth, X_qd], dim=0)
    y_all = torch.cat([y_synth, y_qd], dim=0)
    print(f"\n[DATA] Combined dataset: {X_all.shape[0]} samples total")

    train_loader, val_loader, val_ds = get_loaders(X_all, y_all)

    model     = ShapeWaveNet().to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print(f"\n[TRAIN] Starting {EPOCHS} epochs...\n")

    best_val_acc = 0

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0

        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            loss = criterion(model(bx), by)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        if (epoch + 1) % 5 == 0:
            val_acc  = evaluate(model, val_loader, val_ds, device)
            avg_loss = epoch_loss / len(train_loader)
            print(f"  > Epoch {epoch+1:02d} | Avg Loss: {avg_loss:.4f} | Val Acc: {val_acc:.2f}%")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                save_model(model, SAVE_PATH)
                print(f"  [BEST] New best: {val_acc:.2f}%")

    print(f"\n[DONE] Best val accuracy: {best_val_acc:.2f}%")
    print(f"[DONE] Final model saved to: {SAVE_PATH}")


if __name__ == "__main__":
    train()
