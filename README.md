# 🔷 ShapeWaveNet — Hand-Drawn Shape Classifier

A lightweight CNN trained entirely on **synthetic data** that classifies hand-drawn squares, triangles, and circles in real time — with a live Gradio sketchpad UI.

Built with PyTorch. Originally developed in Google Colab.

---

## Demo

<!-- Replace with your actual demo GIF. Recommended: record with LICEcap or Kap, crop to the Gradio window, 600px wide -->
![Demo](demo.gif)

> *Draw a shape in the sketchpad — the model returns confidence scores for Square, Triangle, and Circle in real time.*

---

## How It Works

**No dataset needed.** The model generates its own training data at runtime using OpenCV — 4,000 synthetic 64×64 grayscale images of squares, triangles, and circles, randomized in position, size, and rotation angle.

The CNN (`ShapeWaveNet`) trains on an 80/20 train/val split using shuffled mini-batches to avoid class bias. At inference time, the Gradio sketchpad captures your drawing, converts it to grayscale, resizes to 64×64, inverts if needed, and feeds it through the model.

### Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│  1. Data generation                                             │
│  OpenCV → 4,000 images (64×64, randomized) → 80/20 split        │
├─────────────────────────────────────────────────────────────────┤
│  2. ShapeWaveNet architecture                                   │
│                                                                 │
│  Input [1×64×64]                                                │
│    → Conv2d(1→16, 3×3) + ReLU → MaxPool → [16×32×32]            │
│    → Conv2d(16→32, 3×3) + ReLU → MaxPool → [32×16×16]           │
│    → Dropout(0.25)                                              │
│    → Flatten → Linear(8192→128) + ReLU                          │
│    → Dropout(0.50)                                              │
│    → Linear(128→3) → Softmax                                    │
│    → [Square | Triangle | Circle]                               │
├─────────────────────────────────────────────────────────────────┤
│  3. Training                                                    │
│  Batch size 64, shuffled → CrossEntropyLoss → Adam (lr=0.001)   │
│  50 epochs, val accuracy logged every 5                         │
├─────────────────────────────────────────────────────────────────┤
│  4. Inference                                                   │
│  Gradio sketchpad → grayscale → 64×64 → model.eval() → scores   │
└─────────────────────────────────────────────────────────────────┘
```

---

## Setup

```bash
pip install torch torchvision opencv-python gradio numpy
python shapesnn.py
```

Open the Gradio link printed in the terminal, draw a shape, and see the confidence scores.

> Runs on CPU by default. A CUDA GPU is used automatically if available.

---

## Key Design Decisions

| Choice | Reason |
|---|---|
| Synthetic data generation | No external dataset needed; full control over shape variation |
| DataLoader with shuffle | Prevents the model from memorizing class order |
| 80/20 train/val split | Tracks generalization, not just memorization |
| Dropout at 0.25 + 0.50 | Regularization in both the feature extractor and classifier |
| Grayscale inversion at inference | Normalizes sketchpad output (white-on-black) to match training format |

---

## Training Details

| Hyperparameter | Value |
|---|---|
| Epochs | 50 |
| Batch size | 64 |
| Optimizer | Adam |
| Learning rate | 0.001 |
| Loss function | CrossEntropyLoss |
| Val logging | Every 5 epochs |

---

## File Structure

```
shapes-neural-network/
├── shapesnn.py     # Data generation, model, training loop, Gradio UI
├── demo.gif        # Screen recording of the live interface
└── README.md
```

---

## Built With

[PyTorch](https://pytorch.org/) · [OpenCV](https://opencv.org/) · [Gradio](https://gradio.app/) · [NumPy](https://numpy.org/)
