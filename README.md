# Drywall Defect Detection Using Fine-Tuned CLIPSeg

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?logo=pytorch&logoColor=white)
![CLIPSeg](https://img.shields.io/badge/Model-CLIPSeg--rd64-blueviolet)
![License](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey)

Prompt-based semantic segmentation for drywall defect detection using a fine-tuned [CLIPSeg](https://huggingface.co/CIDAS/clipseg-rd64-refined) model. Given an image and a text prompt (e.g., `"crack"` or `"drywall joint"`), the model produces a pixel-level segmentation mask highlighting the defect region.

## Key Features

- **Prompt-driven detection** — segment defects by simply changing the text prompt at inference time
- **Multi-defect support** — handles both *cracks* and *drywall joints* with a single model
- **COCO-to-mask pipeline** — `masker.py` converts COCO polygon annotations into binary masks and generates a metadata CSV
- **Class-balanced training** — majority class (cracks) downsampled to match minority class (drywall joints) for fair learning
- **Reproducible** — fixed seed (42) across Python, NumPy, PyTorch, and CUDA for deterministic results
- **Per-class evaluation** — IoU and Dice scores reported overall and per prompt class

## Architecture

CLIPSeg extends [CLIP](https://openai.com/research/clip) with a lightweight decoder for dense prediction:

| Component | Details |
|---|---|
| **Vision Encoder** | ViT-B/16 (Vision Transformer, patch size 16) |
| **Text Encoder** | CLIP text transformer (BPE tokenizer, 77 max tokens) |
| **Decoder** | Lightweight upsampling decoder (FiLM conditioning) |
| **Output Resolution** | 352 x 352 |
| **Pretrained Checkpoint** | `CIDAS/clipseg-rd64-refined` |

The text encoder produces a prompt embedding that conditions the decoder via FiLM layers, enabling zero-shot-style segmentation guided by natural language.

## Project Structure

```
drywall-defect-detection-using-fine-tuned-CLIPSeg/
├── crack-and-drywall.ipynb          # Main notebook: training, evaluation, inference
├── masker.py                        # COCO annotation → binary mask converter
├── Prompted_Segmentation_for_Drywall_QA.pdf  # Project report / documentation
├── raw_roboflow_dataset/            # Original Roboflow datasets
│   ├── cracks/                      #   Crack detection dataset (COCO format)
│   │   ├── train/                   #     5,164 images + _annotations.coco.json
│   │   └── test/                    #     201 images + _annotations.coco.json
│   └── joins/                       #   Drywall joint dataset (COCO format)
│       ├── train/                   #     820 images + _annotations.coco.json
│       └── test/                    #     202 images + _annotations.coco.json
├── clipseg_dataset/                 # Processed dataset for CLIPSeg
│   ├── images/                      #   6,387 images (all splits combined)
│   ├── masks/                       #   6,387 binary masks (PNG, 0/255)
│   └── metadata.csv                 #   image filename ↔ prompt mapping
├── predicted_masks/                 # Exported prediction masks from validation set
├── model/                           # Saved fine-tuned model artifacts
│   ├── config.json
│   ├── tokenizer.json
│   ├── tokenizer_config.json
│   └── processor_config.json
└── .gitignore
```

## Dataset

### Raw Sources (Roboflow)

| Dataset | Train | Test | Format | License |
|---|---|---|---|---|
| [Cracks](https://universe.roboflow.com/shaisthas-workspace/cracks-3ii36-rtmdi) | 5,164 | 201 | COCO JSON | CC BY 4.0 |
| [Drywall Joints](https://universe.roboflow.com/objectdetect-pu6rn/drywall-join-detect) | 820 | 202 | COCO JSON | CC BY 4.0 |

### Processed CLIPSeg Dataset

After running `masker.py`, all images and their binary masks are stored in `clipseg_dataset/` with a unified `metadata.csv`:

| Metric | Count |
|---|---|
| Total images | 6,387 |
| Crack images | 5,365 |
| Drywall joint images | 1,022 |

### Balancing & Splits

The crack class is downsampled to **1,022** samples (matching drywall joints), yielding **2,044** balanced samples. These are then split with stratification:

| Split | Samples |
|---|---|
| Train | 1,430 (70%) |
| Validation | 307 (15%) |
| Test | 307 (15%) |

## Data Preparation

`masker.py` handles the full COCO → CLIPSeg conversion pipeline:

1. **Reads** COCO JSON annotations for each dataset (cracks train/test, joins train/test)
2. **Converts** polygon segmentation annotations to binary masks using `PIL.ImageDraw`; falls back to bounding-box masks when polygons are absent
3. **Merges** multiple annotations per image into a single binary mask via element-wise maximum
4. **Saves** images and masks (scaled to 0/255) into `clipseg_dataset/images/` and `clipseg_dataset/masks/`
5. **Generates** `metadata.csv` with columns: `image` (filename) and `prompt` (`"crack"` or `"drywall joint"`)

## Training

Training is performed in `crack-and-drywall.ipynb` on Kaggle with CUDA GPU acceleration.

| Hyperparameter | Value |
|---|---|
| Epochs | 100 |
| Batch size | 16 |
| Learning rate | 1e-5 |
| Optimizer | AdamW |
| Seed | 42 |
| Input resolution | 352 x 352 |
| Pretrained model | `CIDAS/clipseg-rd64-refined` |

The entire CLIPSeg model (vision encoder, text encoder, and decoder) is fine-tuned end-to-end using the built-in segmentation loss from `CLIPSegForImageSegmentation`.

## Evaluation & Results

Evaluation uses a **0.5 threshold** on sigmoid outputs to produce binary masks, scored with IoU (Intersection over Union) and Dice coefficient.

### Overall (Validation Set)

| Metric | Score |
|---|---|
| **IoU** | 0.4502 |
| **Dice** | 0.5930 |

### Per-Class Breakdown

| Prompt | IoU | Dice |
|---|---|---|
| `crack` | 0.3718 | 0.5162 |
| `drywall joint` | 0.5280 | 0.6693 |

Drywall joints achieve stronger segmentation, likely because joint regions are larger and more uniform compared to thin, irregular cracks.

## Inference & Visualization

The notebook provides two modes of inference:

1. **Visual overlay** — Loads a random validation image, runs the model with a given prompt, and displays a 3-panel figure: original image, ground-truth mask, and prediction overlay (defect pixels highlighted in red)

2. **Batch mask export** — Iterates over the full validation set, generates binary prediction masks (0/255 PNG), and saves them to `predicted_masks/` with the naming convention `{image_id}__{prompt}.png`

### Running Inference

```python
# Change the prompt to detect different defects
prompt = "crack"          # or "drywall joint"

inputs = processor(text=prompt, images=image, return_tensors="pt", padding="max_length")
outputs = model(**inputs)
pred = torch.sigmoid(outputs.logits).squeeze().numpy()
binary_mask = (pred > 0.5).astype(np.uint8) * 255
```

## Installation & Usage

### Prerequisites

- Python 3.10+
- CUDA-compatible GPU (recommended for training)

### Setup

```bash
pip install torch torchvision transformers scikit-learn opencv-python pillow pandas numpy matplotlib tqdm
```

### Running the Data Preparation Pipeline

Update the input/output paths in `masker.py` to point to your local Roboflow dataset directories, then:

```bash
python masker.py
```

This produces the `clipseg_dataset/` folder with images, masks, and `metadata.csv`.

### Running the Training & Evaluation Notebook

Open `crack-and-drywall.ipynb` in Jupyter or Kaggle and run all cells. The notebook handles:
- Loading and balancing the dataset
- Training the CLIPSeg model for 100 epochs
- Evaluating on the validation set (overall + per-class metrics)
- Visualizing predictions with overlays
- Exporting prediction masks

## Model Artifacts

The `model/` directory contains the saved fine-tuned model:

| File | Description |
|---|---|
| `config.json` | Model architecture configuration |
| `tokenizer.json` | BPE tokenizer vocabulary |
| `tokenizer_config.json` | Tokenizer settings |
| `processor_config.json` | Image processor configuration |

> **Note:** Model weight files (`*.safetensors`) are excluded via `.gitignore` due to their size. To obtain the weights, either train the model using the notebook or download from the project's Kaggle output.

## Acknowledgments

- **CLIPSeg** — [CIDAS/clipseg-rd64-refined](https://huggingface.co/CIDAS/clipseg-rd64-refined) by Lüddecke & Ecker
- **Cracks Dataset** — [Roboflow](https://universe.roboflow.com/shaisthas-workspace/cracks-3ii36-rtmdi) (CC BY 4.0)
- **Drywall Joints Dataset** — [Roboflow](https://universe.roboflow.com/objectdetect-pu6rn/drywall-join-detect) (CC BY 4.0)
- **HuggingFace Transformers** — Model loading, processing, and training utilities
