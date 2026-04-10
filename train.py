import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import cv2
from torch.utils.data import DataLoader, TensorDataset
from model import ShapeWaveNet

# ── Config ────────────────────────────────────────────────────────────────────
NUM_SAMPLES   = 10000
BATCH_SIZE    = 64
EPOCHS        = 50
LEARNING_RATE = 0.001
VAL_SPLIT     = 0.8
SAVE_PATH     = "shapewavenet.pth"
# ──────────────────────────────────────────────────────────────────────────────


def generate_shape_data(num_samples: int = NUM_SAMPLES):
    """
    Generates synthetic 64x64 grayscale images of squares, triangles, and circles.
    Shapes are randomized in position, size, stroke thickness, and rotation angle.
    No external dataset required.
    """
    images, labels = [], []
    print(f"[DATA] Generating {num_samples} synthetic shapes...")

    for _ in range(num_samples):
        img = np.zeros((64, 64), dtype=np.uint8)
        shape_type = np.random.randint(0, 3)
        cx, cy     = np.random.randint(20, 44), np.random.randint(20, 44)
        size       = np.random.randint(10, 20)
        thickness  = np.random.randint(1, 3)

        if shape_type == 0:   # Square
            cv2.rectangle(img, (cx - size, cy - size), (cx + size, cy + size), 255, thickness)

        elif shape_type == 1: # Triangle
            pts = np.array([[cx, cy - size], [cx - size, cy + size], [cx + size, cy + size]], np.int32)
            cv2.polylines(img, [pts], True, 255, thickness)

        else:                 # Circle
            cv2.circle(img, (cx, cy), size, 255, thickness)

        # Random rotation for augmentation
        angle = np.random.randint(-180, 180)
        M = cv2.getRotationMatrix2D((32, 32), angle, 1.0)
        img = cv2.warpAffine(img, M, (64, 64), borderMode=cv2.BORDER_REPLICATE)

        images.append(img)
        labels.append(shape_type)

    X = torch.tensor(np.array(images)).float().unsqueeze(1) / 255.0
    y = torch.tensor(labels).long()

    print(f"[DATA] Shape: {X.shape} | Min: {X.min():.2f} | Max: {X.max():.2f}")
    return X, y


def get_loaders(X, y):
    split      = int(VAL_SPLIT * len(X))
    train_ds   = TensorDataset(X[:split], y[:split])
    val_ds     = TensorDataset(X[split:], y[split:])
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)

    print(f"[DATA] Train: {len(train_ds)} | Val: {len(val_ds)}")
    return train_loader, val_loader, val_ds


def evaluate(model, val_loader, val_ds, device):
    model.eval()
    correct = 0
    with torch.no_grad():
        for vx, vy in val_loader:
            preds = torch.max(model(vx.to(device)), 1)[1]
            correct += (preds == vy.to(device)).sum().item()
    return correct / len(val_ds) * 100


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[DEVICE] Running on: {device}")

    X_all, y_all             = generate_shape_data()
    train_loader, val_loader, val_ds = get_loaders(X_all, y_all)

    model     = ShapeWaveNet().to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print(f"\n[TRAIN] Starting {EPOCHS} epochs...\n")

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

        if (epoch + 1) % 10 == 0:
            val_acc = evaluate(model, val_loader, val_ds, device)
            avg_loss = epoch_loss / len(train_loader)
            print(f"  > Epoch {epoch+1:02d} | Avg Loss: {avg_loss:.4f} | Val Acc: {val_acc:.2f}%")

    torch.save(model.state_dict(), SAVE_PATH)
    print(f"\n[SAVED] Model weights saved to {SAVE_PATH}")


if __name__ == "__main__":
    train()
