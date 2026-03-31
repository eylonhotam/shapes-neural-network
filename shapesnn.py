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
    return X, y

# 3. Split dataset into training and validation
X_all, y_all = generate_shape_data(4000)
split = int(0.8 * len(X_all))

train_ds = TensorDataset(X_all[:split], y_all[:split])
val_ds = TensorDataset(X_all[split:], y_all[split:])

# Use DataLoader to prevent the model from getting stuck on one class
train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)

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
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 5. Training 
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
        probs = torch.nn.functional.softmax(output, dim=1)[0]

    labels = ["Square", "Triangle", "Circle"]
    return {labels[i]: float(probs[i]) for i in range(3)}

# Launch UI
interface = gr.Interface(
    fn=predict_shape,
    inputs=gr.Sketchpad(canvas_size=(300, 300), type="numpy"),
    outputs=gr.Label(num_top_classes=3),
    title="Technion AI Lab: Shape Classifier",
    description="Draw a shape! If it's still guessing Square, try drawing thicker lines."
)

interface.launch(share=True)
