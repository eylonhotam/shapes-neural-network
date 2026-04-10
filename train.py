import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

# ── Config ────────────────────────────────────────────────────────────────────
NUM_SAMPLES   = 15000   # per class
BATCH_SIZE    = 64
EPOCHS        = 30
LEARNING_RATE = 0.001
VAL_SPLIT     = 0.8
SAVE_PATH     = "shapewavenet.pth"
# ──────────────────────────────────────────────────────────────────────────────

# Maps class index to label and Quick Draw name
CLASSES = {
    0: "square",
    1: "triangle",
    2: "circle"
}


def generate_shape_data(num_samples: int = NUM_SAMPLES):
    """
    Downloads hand-drawn shape images from the Quick Draw dataset.
    Each class gets num_samples drawings — 15000 total for 3 classes.
    Images are resized to 64x64 grayscale, white shape on black background.
    Requires: pip install quickdraw
    """
    try:
        from quickdraw import QuickDrawDataGroup
    except ImportError:
        raise ImportError(
            "quickdraw package not found. Install it with: pip install quickdraw"
        )

    images, labels = [], []

    for label, name in CLASSES.items():
        print(f"[DATA] Downloading '{name}' drawings ({num_samples} samples)...")
        group = QuickDrawDataGroup(name, max_drawings=num_samples, recognized=True)

        count = 0
        for drawing in group.drawings:
            # Convert PIL image to grayscale numpy array at 64x64
            img = np.array(drawing.image.convert("L").resize((64, 64)))

            # Quick Draw is black-on-white — invert to white-on-black
      
            img = 255 - img

            # Random rotation so model handles all orientations
            angle = np.random.randint(0, 360)
            center = (32, 32)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            img = cv2.warpAffine(img, M, (64, 64), borderMode=cv2.BORDER_CONSTANT, borderValue=0)

            images.append(img)
            labels.append(label)
            count += 1
            if count >= num_samples:
                break

        print(f"[DATA] Loaded {count} '{name}' drawings.")

    X = torch.tensor(np.array(images)).float().unsqueeze(1) / 255.0
    y = torch.tensor(labels).long()

    print(f"\n[DATA] Total shape: {X.shape} | Min: {X.min():.2f} | Max: {X.max():.2f}")
    return X, y


def get_loaders(X, y):
    # Shuffle before splitting to prevent class skew in val set
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


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[DEVICE] Running on: {device}")

    X_all, y_all                     = generate_shape_data()
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

            # Save best model only
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), SAVE_PATH)
                print(f"  [SAVED] New best model: {val_acc:.2f}% -> {SAVE_PATH}")

    print(f"\n[DONE] Best val accuracy: {best_val_acc:.2f}%")


if __name__ == "__main__":
    train()
