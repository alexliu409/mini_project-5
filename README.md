# Mini Project V — Forest Fire, Smoke & Non-Fire CNN Classifier

## Problem Description
This project builds a custom CNN from scratch to classify images into three categories:
**fire**, **smoke**, and **non-fire**. Early wildfire detection is critical for emergency
response, and distinguishing smoke from actual fire adds real-world complexity beyond
binary classification.

## Dataset
- **Source:** [Forest Fire, Smoke & Non-Fire Image Dataset](https://www.kaggle.com/datasets/amerzishminha/forest-fire-smoke-and-non-fire-image-dataset)
- **Classes:** Smoke / fire / non fire (3-class)
- **Size:** 31,708 train images + 9,446 test images
- **Split:** Pre-split into `train/` and `test/`; validation created from train (80/20)
- **Image size:** Resized to 128×128 RGB
- **Class balance (train):** Smoke 10,108 / fire 10,798 / non fire 10,800 — imbalance ratio 1.07x (well balanced)

## Data Access Instructions
1. Go to https://www.kaggle.com/datasets/amerzishminha/forest-fire-smoke-and-non-fire-image-dataset
2. Download and unzip the dataset
3. Place files in the following structure:
```
data/
  train/
    Smoke/
    fire/
    non fire/
  test/
    Smoke/
    fire/
    non fire/
```
4. Update `DATA_DIR` in Cell [4] to point to your local path

## Setup & Running

```bash
pip install -r requirements.txt
```

Open and run all cells **in order** in `notebooks/mini_project_5.ipynb`.

## Results Summary

| Model | Train Acc | Val Acc | Val F1 | Gap |
|-------|-----------|---------|--------|-----|
| Baseline CNN | 0.9946 | 0.9696 | 0.9666 | 0.0251 |
| Augmented CNN | 0.9780 | 0.9708 | 0.9672 | 0.0072 |

**Key observations:**
- Augmentation reduced the train-val gap from 2.51% → 0.72%
- Only 208 / 6,341 val images misclassified (3.3%) by the augmented model
- Most confused pair: **Smoke → non fire** (68 cases) and **Smoke → fire** (49 cases)

## Sample Predictions

<img width="1267" height="593" alt="image" src="https://github.com/user-attachments/assets/c4680cc7-3ed6-4f6f-bd82-52cbf3adb329" />


## Key Findings
- The dataset is well balanced (imbalance ratio 1.07x), so class weighting was not needed
- Smoke is the hardest class — it shares visual properties with both fire (haze, orange glow) and non-fire (overcast skies, fog)
- Non-fire scenes with warm lighting (e.g. sunsets) were sometimes confused with fire (39 cases)

## What We'd Try Next
- Increase resolution from 128 → 224 to preserve smoke texture detail
- Add `RandomBrightness` augmentation to handle sunset/warm lighting confusion
- Use class weights to address test set Smoke imbalance (2,446 vs 3,500)
- Transfer learning in Mini Project 6 (EfficientNet / ResNet)

## Team Contributions
- Hsuan Chen Liu - Augmented CNN design & training, model comparison table, activation heatmap, misclassified analysis, report
- Brendan - Data exploration, baseline CNN design & training, feature maps visualization, report 
