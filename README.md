# 🔷 ShapeWaveNet — Hand-Drawn Shape Classifier

A lightweight PyTorch CNN that classifies hand-drawn **squares**, **triangles**, and **circles** in real time via a Gradio sketchpad interface. Trained on a **hybrid dataset** — synthetic OpenCV images combined with real hand-drawn examples from the [Quick Draw dataset](https://quickdraw.withgoogle.com/data).

> **No external image dataset required beyond Quick Draw** — synthetic data is generated at runtime.

---

## 🚀 Live Demo
![Live Demo Gif](assets/demo.gif)

**Try it here:** [Hugging Face Spaces - ShapeWaveNet](https://huggingface.co/spaces/eylonhotam/shapes-neural-network)

---
## How It Works

Training combines two data sources: 5,000 synthetic 64×64 grayscale images per class (generated with OpenCV) and 10,000 real hand-drawn images per class from Google's Quick Draw dataset. Both are augmented with random rotation and stroke thickness variation. The best checkpoint is saved to `shapewavenet.pth` so the app loads instantly without retraining.

### Pipeline

```
┌──────────────────────────────────────────────────────────────────────┐
│  1. Data generation — two sources combined                           │
│                                                                      │
│  Synthetic (OpenCV) → 5,000 imgs/class (64×64, randomized)           │
│    · Random position, size, thickness, rotation                      │
│    · Gaussian noise (30% chance) + blur (40% chance)                 │
│    · Full 360° rotation for squares/circles; ±30° for triangles      │
│                                                                      │
│  Quick Draw → 10,000 imgs/class (recognized drawings only)           │
│    · Inverted to white-on-black, resized to 64×64                    │
│    · Random rotation + erode/dilate thickness augmentation           │
│                                                                      │
│  Combined → 45,000 total samples → 80/20 train/val split             │
├──────────────────────────────────────────────────────────────────────┤
│  2. ShapeWaveNet architecture                                        │
│                                                                      │
│  Input [1×64×64]                                                     │
│    → Conv2d(1→16, 3×3) + ReLU → MaxPool  → [16×32×32]                │
│    → Conv2d(16→32, 3×3) + ReLU → MaxPool → [32×16×16]                │
│    → Dropout(0.25)                                                   │
│    → Flatten → Linear(8192→128) + ReLU                               │
│    → Dropout(0.50)                                                   │
│    → Linear(128→3) → Temperature Scaling → Softmax                   │
│    → [Square | Triangle | Circle]                                    │
├──────────────────────────────────────────────────────────────────────┤
│  3. Training                                                         │
│  Batch 64, shuffled → CrossEntropyLoss(label_smoothing=0.1)          │
│  Adam (lr=0.001) · 30 epochs · val accuracy + per-class acc logged   │
│  Best checkpoint saved automatically                                 │
├──────────────────────────────────────────────────────────────────────┤
│  4. Inference                                                        │
│  Gradio sketchpad → grayscale → 64×64 → model.eval()                 │
│  → temperature scaling (T=1.0) → softmax → confidence scores         │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Setup

```bash
pip install -r requirements.txt

# Step 1 — train and save weights (run once)
python train.py

# Step 2 — launch the app
python app.py
```

Open the Gradio link printed in the terminal, draw a shape, and see the confidence scores. Runs on CPU by default; a CUDA GPU is used automatically if available.



---

## File Structure

```
shapes-neural-network/
├── model.py          # ShapeWaveNet architecture
├── train.py          # Data generation + training loop
├── app.py            # Gradio inference UI
├── requirements.txt
└── README.md
```

---

## Key Design Decisions

| Decision | Reason |
|---|---|
| Hybrid synthetic + Quick Draw data | Synthetic data gives clean geometric variety; Quick Draw closes the synthetic-to-real gap with 10k genuine human sketches per class |
| Stroke thickness augmentation (erode/dilate) | Quick Draw strokes vary widely in weight; augmenting thickness teaches the model to ignore it |
| Gaussian noise + blur on synthetic images | Simulates the imprecision of sketchpad input at inference time |
| Triangle rotation capped at ±30° | Full rotation makes upward/downward triangles ambiguous at inference; limited range preserves recognisability |
| `label_smoothing=0.1` | Prevents the model from pushing logits to extremes during training |
| Temperature scaling (T=2.0) | Reduces softmax overconfidence at inference time |
| Dropout at 0.25 + 0.50 | Regularization in both the feature extractor and classifier |
| Save best checkpoint only | Avoids overwriting a good model with a worse late-epoch result |
| `weights_only=True` on load | Follows PyTorch best practices for safe model loading |

---

## Training Config

| Parameter | Value |
|---|---|
| Synthetic samples / class | 5,000 |
| Quick Draw samples / class | 10,000 |
| Total training samples (approx.) | ~36,000 |
| Batch size | 64 |
| Epochs | 30 |
| Learning rate | 0.001 |
| Val split | 20% |
| Accurary | 94.6%|

---

## Usage Tips

- Draw with **thin, clear strokes** — the model was trained on outline-style shapes, not filled ones.
- **Close your shapes fully** — open corners on squares or triangles can confuse the classifier.
- The app auto-inverts your drawing if it detects a white background, to match the training format (white shapes on black).
- If the margin between the top two predictions is under 0.2, the app will log a warning that the drawing may be ambiguous — try redrawing more deliberately.

---

## Roadmap / Future Steps

- [x] **`requirements.txt`** — pin dependency versions for reproducible installs
- [x] **Hugging Face Spaces deployment** — host the Gradio app publicly without needing a local tunnel 
- [x] **Real dataset support** — integrate Google Quick, Draw! data alongside synthetically generated data to improve robustness on actual handwriting
- [ ] **Add more shape classes** — pentagon, star, arrow, cross
- [ ] **Per-class accuracy logging** — add a confusion matrix at the end of training to identify which shape is hardest to classify
- [ ] **Learning rate scheduler** — experiment with `CosineAnnealingLR` or `ReduceLROnPlateau` for better convergence

---

## Built With

[PyTorch](https://pytorch.org/) · [OpenCV](https://opencv.org/) · [Gradio](https://gradio.app/) · [NumPy](https://numpy.org/) · [Google Quick, Draw!](https://quickdraw.withgoogle.com/data)
