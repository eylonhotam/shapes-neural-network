import torch
import numpy as np
import cv2
import gradio as gr
from model import ShapeWaveNet

# ── Config ────────────────────────────────────────────────────────────────────
WEIGHTS_PATH = "shapewavenet.pth"
TEMPERATURE  = 2.0   # Softmax temperature — higher = less overconfident
# ──────────────────────────────────────────────────────────────────────────────

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ShapeWaveNet().to(device)
model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device, weights_only=True))
model.eval()
print(f"[LOADED] Model loaded from {WEIGHTS_PATH} | Device: {device}")


def predict_shape(data):
    """
    Accepts a Gradio sketchpad input, preprocesses the drawing,
    and returns softmax confidence scores for Square, Triangle, Circle.

    Preprocessing:
        - Convert RGB to grayscale
        - Resize to 64x64
        - Invert if background is white (sketchpad default)
        - Normalize to [0, 1]

    Temperature scaling is applied before softmax to reduce overconfidence.
    """
    # Preprocess
    img_array = np.array(data["composite"][:, :, :3])
    gray      = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    resized   = cv2.resize(gray, (64, 64))

    # Invert if needed — model trained on white-on-black
    if resized.mean() > 127:
        resized = 255 - resized

    tensor_img = torch.tensor(resized).float().unsqueeze(0).unsqueeze(0) / 255.0

    # Inference with temperature scaling
    with torch.no_grad():
        logits = model(tensor_img.to(device))
        probs  = torch.nn.functional.softmax(logits / TEMPERATURE, dim=1)[0]

    # Confidence diagnostics
    top_prob = float(probs.max())
    sorted_probs, _ = torch.sort(probs, descending=True)
    margin = float(sorted_probs[0] - sorted_probs[1])

    print(f"[INFERENCE] Sq: {probs[0]:.2f} | Tri: {probs[1]:.2f} | Cir: {probs[2]:.2f} | "
          f"Top: {top_prob:.2f} | Margin: {margin:.2f} | Temp: {TEMPERATURE}")

    if top_prob > 0.95:
        print("[WARNING] High confidence — possible softmax overconfidence")
    elif margin < 0.2:
        print("[WARNING] Low margin — drawing may be ambiguous")

    return {
        "Square":   float(probs[0]),
        "Triangle": float(probs[1]),
        "Circle":   float(probs[2])
    }


interface = gr.Interface(
    fn=predict_shape,
    inputs=gr.Sketchpad(canvas_size=(300, 300), type="numpy"),
    outputs=gr.Label(num_top_classes=3),
    title="ShapeWaveNet — Shape Classifier",
    description=(
        "Draw a square, triangle, or circle in the sketchpad. "
        "The model returns confidence scores in real time.\n\n"
        "Tips: Use thick, clear strokes. Close your shapes fully for best results."
    )
)

if __name__ == "__main__":
    interface.launch()
