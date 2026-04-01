import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import cv2
import gradio as gr
from torch.utils.data import DataLoader, TensorDataset

print(f"Using PyTorch version: {torch.__version__}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on: {device}")

def generate_shape_data(num_samples=3000):
    # Generates shapes for training, background 0 and shape 1 for max difference.
    
    images = []
    labels = []
    print(f"[PROCESS] Generating {num_samples} synthetic shapes...")
    for _ in range(num_samples):
        img = np.zeros((64, 64), dtype=np.uint8)
        shape_type = np.random.randint(0, 3)

        
        center_x = np.random.randint(20, 44)
        center_y = np.random.randint(20, 44)
        size = np.random.randint(10, 20)
        thickness = np.random.randint(1, 3)

        if shape_type == 0: # Square
            cv2.rectangle(img, (center_x-size, center_y-size), (center_x+size, center_y+size), 255, thickness)
        elif shape_type == 1: # Triangle
            pts = np.array([[center_x, center_y-size], [center_x-size, center_y+size], [center_x+size, center_y+size]], np.int32)
            cv2.polylines(img, [pts], True, 255, thickness)
        else: # Circle
            cv2.circle(img, (center_x, center_y), size, 255, thickness)

        # Slight rotation in data
        angle = np.random.randint(-30, 30)
        M = cv2.getRotationMatrix2D((32, 32), angle, 1.0)
        img = cv2.warpAffine(img, M, (64, 64))

        images.append(img)
        labels.append(shape_type)

    # Convert to Torch Tensors [N, C, H, W] and normalize 0.0 - 1.0
    X = torch.tensor(np.array(images)).float().unsqueeze(1) / 255.0
    y = torch.tensor(labels).long()
    print(f"[DEBUG] Data shape: {X.shape} | Max value: {X.max().item()} | Min: {X.min().item()}")
    return X, y

# 3. Split dataset into training and validation
X_all, y_all = generate_shape_data(4000)
split = int(0.8 * len(X_all))

train_ds = TensorDataset(X_all[:split], y_all[:split])
val_ds = TensorDataset(X_all[split:], y_all[split:])
train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)

print(f"[DEBUG] Total Validation Samples (val_ds): {len(val_ds)}")
print(f"[DEBUG] Total Validation Batches (val_loader): {len(val_loader)}")
print(f"[DEBUG] Training samples: {len(train_loader.dataset)}")
print(f"[DEBUG] Validation samples: {len(val_loader.dataset)}")
print(f"[LOG] Shuffling enabled for training to prevent class bias.")


class ShapeWaveNet(nn.Module):
    def __init__(self):
        super(ShapeWaveNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25)
        )
        self.classifier = nn.Sequential(
            nn.Linear(32 * 16 * 16, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 3)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

model = ShapeWaveNet().to(device)
print(f"[DEBUG] Model initialized on {device}. Architecture ready.")

# 5. Training
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = optim.Adam(model.parameters(), lr=0.001) 
epochs = 50
print("Starting Training...")

for epoch in range(epochs):
    model.train()
    total_loss = 0
    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)

        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    if (epoch + 1) % 5 == 0:
        model.eval()
        correct = 0
        with torch.no_grad():
            for vX, vy in val_loader:
                vX, vy = vX.to(device), vy.to(device)
                _, pred = torch.max(model(vX), 1)
                correct += (pred == vy).sum().item()

        val_acc = (correct / len(val_ds)) * 100
        print(f"Epoch {epoch+1} | Loss: {total_loss/len(train_loader):.4f} | Val Acc: {val_acc:.2f}%")

# 6. Drawing and Prediction Interface
def predict_shape(data):
    # Process drawing, make sure it's white on black
    img_array = np.array(data['composite'][:, :, :3])
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, (64, 64))

    if resized.mean() > 127:
        resized = 255 - resized

    # Scale 0-1
    tensor_img = torch.tensor(resized).float().unsqueeze(0).unsqueeze(0) / 255.0

    model.eval()
    with torch.no_grad():
        output = model(tensor_img.to(device))
        scaled = output / temperature          # temperature scaling
        probs = torch.nn.functional.softmax(scaled, dim=1)[0]

    top_prob = float(probs.max())
    sorted_probs, _ = torch.sort(probs, descending=True)
    margin = float(sorted_probs[0] - sorted_probs[1])

    print(f"[INFERENCE] Temp: {temperature} | Confidence -> Sq: {probs[0]:.2f}, Tri: {probs[1]:.2f}, Cir: {probs[2]:.2f}")
    print(f"[INFERENCE] Top prob: {top_prob:.2f} | Margin: {margin:.2f}")

    if top_prob > 0.95:
        print("[WARNING] High confidence — possible overconfidence")
    elif margin < 0.2:
        print("[WARNING] Low margin — drawing may be ambiguous")

    return {"Square": float(probs[0]), "Triangle": float(probs[1]), "Circle": float(probs[2])}
# Launch UI
interface = gr.Interface(
    fn=predict_shape,
    inputs=gr.Sketchpad(canvas_size=(300, 300), type="numpy"),
    outputs=gr.Label(num_top_classes=3),
    title="Shape Classifier",
    description="Draw a shape! If it's still guessing Square, try drawing thicker lines."
)

interface.launch(share=True)
