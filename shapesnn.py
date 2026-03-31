
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

# Generate initial data for model to train on
def generate_data(num_samples=2000):
    images = []
    labels = []
    for _ in range(num_samples):
        # white background
        img = np.ones((64, 64), dtype=np.uint8) * 255
        shape_type = np.random.randint(0, 3)

        center_x, center_y = np.random.randint(25, 39), np.random.randint(25, 39)
        size = np.random.randint(12, 18)
        thickness = np.random.randint(1, 3)

        if shape_type == 0: # Square
            cv2.rectangle(img, (center_x-size, center_y-size), (center_x+size, center_y+size), 0, thickness)
        elif shape_type == 1: # Triangle
            pts = np.array([[center_x, center_y-size], [center_x-size, center_y+size], [center_x+size, center_y+size]], np.int32)
            cv2.polylines(img, [pts], True, 0, thickness)
        else: # Circle
            cv2.circle(img, (center_x, center_y), size, 0, thickness)
          
        angle = np.random.randint(-20, 20)
        M = cv2.getRotationMatrix2D((32, 32), angle, 1.0)
        img = cv2.warpAffine(img, M, (64, 64), borderValue=255)

        images.append(img)
        labels.append(shape_type)

    # Return normalized tensors
    return torch.tensor(np.array(images)).float().unsqueeze(1) / 255.0, torch.tensor(labels)

X_train, y_train = generate_data(2500)
print("Generated 2500 Black-on-White shapes with random rotations.")

# Cell 3: The ShapeWaveNet (Model) with Dropout
class ShapeWaveNet(nn.Module):
    def __init__(self):
        super(ShapeWaveNet, self).__init__()

        # Wave 1: Edge detection (64x64 -> 32x32 after pooling)
        self.wave1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        # Wave 2: Pattern combination (32x32 -> 16x16 after pooling)
        self.wave2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)

        # Wave 3: The Decision Logic
        # Dropout: Randomly ignores 25% of neurons to prevent overfitting
        self.dropout = nn.Dropout(0.25)

        self.fc1 = nn.Linear(32 * 16 * 16, 128)
        self.output_layer = nn.Linear(128, 3) # 0: Square, 1: Triangle, 2: Circle

    def forward(self, x):
        
        x = self.pool(torch.relu(self.wave1(x)))
        x = self.pool(torch.relu(self.wave2(x)))
        x = x.view(-1, 32 * 16 * 16)

        # Apply Dropout only during training
        x = self.dropout(x)

        # Final Decision
        x = torch.relu(self.fc1(x))
        x = self.output_layer(x)
        return x

model = ShapeWaveNet().to(device)
print("Model rebuilt with Dropout for better generalization.")

#Extended Training
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
        # Calculate Accuracy
        _, predicted = torch.max(outputs, 1)
        correct = (predicted == y_train.to(device)).sum().item()
        accuracy = (correct / y_train.size(0)) * 100
        print(f"Round {epoch+1:3d} | Loss: {loss.item():.4f} | Accuracy: {accuracy:.2f}%")

print("Training Finished!")

# Cell 5: Interactive UI with Probability Feedback
def predict_image(data):
    img = data['composite'][:, :, 3]
    img = cv2.resize(img, (64, 64))
    img_tensor = torch.tensor(img).float().unsqueeze(0).unsqueeze(0) / 255.0
    model.eval() # Set to evaluation mode (turns off Dropout)
    with torch.no_grad():
        output = model(img_tensor.to(device))
        # Convert raw scores (logits) to probabilities (0% to 100%)
        probabilities = torch.nn.functional.softmax(output, dim=1)[0]

    # Map probabilities to labels for the UI
    labels = ["Square", "Triangle", "Circle"]
    return {labels[i]: float(probabilities[i]) for i in range(3)}

# UI
interface = gr.Interface(
    fn=predict_image,
    inputs=gr.Sketchpad(canvas_size=(300, 300), type="numpy"),
    outputs=gr.Label(num_top_classes=3),
    title="Shape Classifier",
    description="Draw a shape in the center. The model will show its confidence levels below!"
)

interface.launch(share=True, debug=True)
