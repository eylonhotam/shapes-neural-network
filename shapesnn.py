import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import cv2
import gradio as gr
import matplotlib.pyplot as plt

print(f"Using PyTorch version: {torch.__version__}")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on: {device}")

def generate_inverted_data(num_samples=2000):
    images = []
    labels = []
    for _ in range(num_samples):
        img = np.ones((64, 64), dtype=np.uint8) * 255
        shape_type = np.random.randint(0, 3)

        center_x, center_y = np.random.randint(25, 39), np.random.randint(25, 39)
        size = np.random.randint(12, 18)
        thickness = np.random.randint(1, 3)

        if shape_type == 0:
            cv2.rectangle(img, (center_x-size, center_y-size), (center_x+size, center_y+size), 0, thickness)
        elif shape_type == 1:
            pts = np.array([[center_x, center_y-size], [center_x-size, center_y+size], [center_x+size, center_y+size]], np.int32)
            cv2.polylines(img, [pts], True, 0, thickness)
        else:
            cv2.circle(img, (center_x, center_y), size, 0, thickness)

        angle = np.random.randint(-20, 20)
        M = cv2.getRotationMatrix2D((32, 32), angle, 1.0)
        img = cv2.warpAffine(img, M, (64, 64), borderValue=255)

        images.append(img)
        labels.append(shape_type)

    return torch.tensor(np.array(images)).float().unsqueeze(1) / 255.0, torch.tensor(labels)

X_all, y_all = generate_inverted_data(2500)

# Split into train and validation sets
split = int(0.8 * len(X_all))
X_train, y_train = X_all[:split], y_all[:split]
X_val, y_val = X_all[split:], y_all[split:]

print(f"Train: {len(X_train)} samples | Val: {len(X_val)} samples")

class ShapeWaveNet(nn.Module):
    def __init__(self):
        super(ShapeWaveNet, self).__init__()

        self.wave1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.wave2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.dropout = nn.Dropout(0.25)
        self.fc1 = nn.Linear(32 * 16 * 16, 128)
        self.output_layer = nn.Linear(128, 3)

    def forward(self, x):
        x = self.pool(torch.relu(self.wave1(x)))
        x = self.pool(torch.relu(self.wave2(x)))
        x = x.view(-1, 32 * 16 * 16)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)  
        x = self.output_layer(x)
        return x

model = ShapeWaveNet().to(device)
print("Model ready.")

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 100
print(f"Training for {epochs} rounds...")

for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()

    outputs = model(X_train.to(device))
    loss = criterion(outputs, y_train.to(device))
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        _, predicted = torch.max(outputs, 1)
        train_acc = (predicted == y_train.to(device)).sum().item() / y_train.size(0) * 100

        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val.to(device))
            _, val_predicted = torch.max(val_outputs, 1)
            val_acc = (val_predicted == y_val.to(device)).sum().item() / y_val.size(0) * 100
        model.train()

        print(f"Round {epoch+1:3d} | Loss: {loss.item():.4f} | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")

print("Training Finished!")

model.eval()
with torch.no_grad():
    outputs = model(X_train.to(device))
    _, predicted = torch.max(outputs, 1)
    acc = (predicted == y_train.to(device)).sum().item() / y_train.size(0) * 100
    print(f"Train accuracy: {acc:.2f}%")

    # Check distribution of predictions
    for i, name in enumerate(["Square", "Triangle", "Circle"]):
        count = (predicted == i).sum().item()
        print(f"  Predicted as {name}: {count} times")

def debug_predict(data):
    img = data['composite'][:, :, 3]
    img = cv2.resize(img, (64, 64))
    img_inverted = 255 - img

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(data['composite'])
    axes[0].set_title("Raw Sketchpad (RGBA)")
    axes[1].imshow(img, cmap='gray')
    axes[1].set_title(f"Alpha channel\nmin={img.min()} max={img.max()}")
    axes[2].imshow(img_inverted, cmap='gray')
    axes[2].set_title(f"After inversion (fed to model)\nmin={img_inverted.min()} max={img_inverted.max()}")
    plt.show()

    img_tensor = torch.tensor(img_inverted).float().unsqueeze(0).unsqueeze(0) / 255.0
    print(f"Tensor min: {img_tensor.min():.3f}, max: {img_tensor.max():.3f}, mean: {img_tensor.mean():.3f}")

    model.eval()
    with torch.no_grad():
        output = model(img_tensor.to(device))
        probs = torch.nn.functional.softmax(output, dim=1)[0]
        print(f"Raw logits: {output}")
        print(f"Square: {probs[0]:.2%} | Triangle: {probs[1]:.2%} | Circle: {probs[2]:.2%}")

# debug interface
def debug_predict(data):
    if isinstance(data, dict):
        composite = data['composite'] if data.get('composite') is not None else data['image']
    else:
        composite = data

    composite = np.array(composite)
    print(f"Composite shape: {composite.shape}, dtype: {composite.dtype}")

    if composite.ndim == 3 and composite.shape[2] == 4:
        img = composite[:, :, 3]
        print("Using alpha channel")
    elif composite.ndim == 3 and composite.shape[2] == 3:
        img = cv2.cvtColor(composite, cv2.COLOR_RGB2GRAY)
        print("Using grayscale from RGB")
    else:
        img = composite

    img_resized = cv2.resize(img, (64, 64))
    img_inverted = 255 - img_resized

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(composite)
    axes[0].set_title(f"Raw composite\nshape={composite.shape}")
    axes[1].imshow(img_resized, cmap='gray')
    axes[1].set_title(f"Extracted channel\nmin={img_resized.min()} max={img_resized.max()}")
    axes[2].imshow(img_inverted, cmap='gray')
    axes[2].set_title(f"After inversion\nmean={img_inverted.mean():.1f}")
    plt.tight_layout()
    plt.show()

    img_tensor = torch.tensor(img_inverted).float().unsqueeze(0).unsqueeze(0) / 255.0
    print(f"Tensor — min: {img_tensor.min():.3f}, max: {img_tensor.max():.3f}, mean: {img_tensor.mean():.3f}")

    model.eval()
    with torch.no_grad():
        output = model(img_tensor.to(device))
        probs = torch.nn.functional.softmax(output, dim=1)[0]
        print(f"Raw logits: {output.cpu().numpy()}")
        print(f"Square: {probs[0]:.2%} | Triangle: {probs[1]:.2%} | Circle: {probs[2]:.2%}")

    labels = ["Square", "Triangle", "Circle"]
    return {labels[i]: float(probs[i]) for i in range(3)}

debug_interface = gr.Interface(
    fn=debug_predict,
    inputs=gr.Sketchpad(canvas_size=(300, 300), type="numpy"),
    outputs=gr.Label(num_top_classes=3),
    title="DEBUG: Shape Classifier",
    description="Draw a shape — check the print output below for debug info"
)

debug_interface.launch(share=True, debug=True)

def predict_image(data):
    img = data['composite'][:, :, 3]
    img = cv2.resize(img, (64, 64))

    # FIX Bug #1: Invert so drawn strokes (255) become dark (0) on white (255) background
    img = 255 - img

    img_tensor = torch.tensor(img).float().unsqueeze(0).unsqueeze(0) / 255.0

    model.eval()
    with torch.no_grad():
        output = model(img_tensor.to(device))
        probabilities = torch.nn.functional.softmax(output, dim=1)[0]

    labels = ["Square", "Triangle", "Circle"]
    return {labels[i]: float(probabilities[i]) for i in range(3)}

interface = gr.Interface(
    fn=predict_image,
    inputs=gr.Sketchpad(canvas_size=(300, 300), type="numpy"),
    outputs=gr.Label(num_top_classes=3),
    title="Shape Classifier",
    description="Draw a shape in the center. The model will show its confidence levels below!"
)

interface.launch(share=True, debug=True)
