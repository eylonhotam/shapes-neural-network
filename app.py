import torch
import numpy as np
import cv2
import gradio as gr
from model import ShapeWaveNet

# ── Config ────────────────────────────────────────────────────────────────────
WEIGHTS_PATH = "shapewavenet.pth"
TEMPERATURE  = 2.0   # Softmax temperature — higher = less overconfident
DEBUG_IMG        = False  # Saves debug_input.png so you can inspect preprocessed input
DEBUG_WEIGHTS    = False  # Prints statements regarding weights for inspection
# ──────────────────────────────────────────────────────────────────────────────

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ShapeWaveNet().to(device)
model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device, weights_only=True))
model.eval()

if DEBUG_WEIGHTS:
    p = list(model.parameters())
    print("[W0]", p[0].flatten()[:5])
    print("[W1]", p[1].flatten()[:5])

print(f"[LOADED] Model loaded from {WEIGHTS_PATH} | Device: {device}")


def predict_shape(data):
    """
    Accepts a Gradio sketchpad input, preprocesses the drawing,
    and returns softmax confidence scores for Square, Triangle, Circle.

    Preprocessing:
        - Convert RGB to grayscale
        - Resize to 64x64
        - Invert if background is white (sketchpad default)
        - Slight Gaussian blur to match training augmentation
        - Normalize to [0, 1]

    Temperature scaling is applied before softmax to reduce overconfidence.
    """
    # Preprocess
    if isinstance(data, dict):
        img_array = np.array(data["composite"][:, :, :3])
    else:
        img_array = np.array(data[:, :, :3])

    gray    = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, (64, 64))

    # Invert if needed — model trained on white-on-black
    if resized.mean() > 127:
        resized = 255 - resized

    if resized.max() > 0:
        resized = (resized.astype(float) / resized.max() * 120).astype(np.uint8)

    # After inversion, crop to content and resize to fill frame
    coords = cv2.findNonZero(resized)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        padding = 4
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(64 - x, w + padding * 2)
        h = min(64 - y, h + padding * 2)
        cropped = resized[y:y+h, x:x+w]
        resized = cv2.resize(cropped, (64, 64), interpolation=cv2.INTER_AREA)

    # Slight blur to match training augmentation
    resized = cv2.GaussianBlur(resized, (3, 3), 0)

    # Save debug image so you can inspect what the model actually sees
    if DEBUG_IMG:
        cv2.imwrite("debug_input.png", resized)
        print(f"[DEBUG] Saved debug_input.png | Mean pixel: {resized.mean():.2f} | Max: {resized.max()}")

    tensor_img = torch.tensor(resized).float().unsqueeze(0).unsqueeze(0) / 255.0

    if DEBUG_WEIGHTS:
        print("[INPUT STATS] mean:", tensor_img.mean().item(), "max:", tensor_img.max().item())
    
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
        "Tips: Use thin, clear strokes. Close your shapes fully for best results."
    )
)

if __name__ == "__main__":
    interface.launch()
