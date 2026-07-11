# Zero-Shot Image Classification with CLIP

A comparative study of zero-shot image classification using OpenAI's CLIP model against a traditionally fine-tuned CNN (ResNet18), evaluated across three diverse image datasets.

## Overview

This project investigates whether a pretrained vision-language model can classify images **without ever being trained on them**, and how that performance compares to a conventional supervised deep learning approach. Using CLIP (Contrastive Language-Image Pretraining), images are classified purely by matching them against natural language descriptions of each class — no labeled training data or model fine-tuning required.

To evaluate this fairly, zero-shot CLIP was benchmarked against a ResNet18 CNN, fine-tuned in the standard supervised manner, across three datasets with different visual characteristics.

## Key Finding

Zero-shot CLIP outperformed the fine-tuned CNN on **2 out of 3 datasets**, despite requiring no training data or training time.

| Dataset | Classes | Zero-Shot CLIP | Fine-Tuned ResNet18 | Winner |
|---|---|---|---|---|
| CIFAR-10 | 10 | 88.80% | 94.15% | Fine-tuned CNN |
| Intel Image Classification | 6 | 93.00% | 92.47% | Zero-shot CLIP |
| Flowers Recognition | 5 | 94.79% | 92.48% | Zero-shot CLIP |

This suggests that when labeled training data is limited, a strong pretrained foundation model can match or exceed the performance of a model trained specifically for the task — a practically useful insight for real-world scenarios where large labeled datasets aren't available.

## Datasets

| Dataset | Description | Source |
|---|---|---|
| **CIFAR-10** | 10 classes of common objects (airplane, cat, truck, etc.), 10,000 test images | Auto-downloaded via `torchvision.datasets.CIFAR10` |
| **Intel Image Classification** | 6 natural scene classes (buildings, forest, glacier, mountain, sea, street), ~25,000 images | [Kaggle: puneet6060/intel-image-classification](https://www.kaggle.com/datasets/puneet6060/intel-image-classification) |
| **Flowers Recognition** | 5 flower species (daisy, dandelion, rose, sunflower, tulip), ~4,300 images | [Kaggle: alxmamaev/flowers-recognition](https://www.kaggle.com/datasets/alxmamaev/flowers-recognition) |

## Methodology

1. **Zero-shot classification**: For each dataset, class names were converted into natural language prompts (e.g., `"a photo of a {class}"`). CLIP computed similarity scores between each image and every prompt; the highest-scoring prompt determined the predicted class. No training was performed.

2. **Supervised baseline**: A ResNet18 model, pretrained on ImageNet, was fine-tuned on each dataset's training data for 3 epochs using standard supervised learning, then evaluated on a held-out test set.

3. **Evaluation**: Both approaches were assessed using accuracy, precision, recall, and F1-score, with confusion matrices generated to identify class-level confusion patterns.

4. **Error analysis**: Misclassified examples were manually inspected to understand failure patterns (e.g., visual similarity between classes, image quality, ambiguous content).

5. **Prompt engineering**: Multiple prompt templates were tested to measure how wording affects zero-shot accuracy.

## Results in Detail

### CIFAR-10
- **Zero-shot accuracy**: 88.80% | **Fine-tuned CNN accuracy**: 94.15%
- 1,120 of 10,000 images misclassified in the zero-shot setting
- Most confusion occurred between visually similar animal classes (e.g., deer/horse, cat/dog)

### Intel Image Classification
- **Zero-shot accuracy**: 93.00% | **Fine-tuned CNN accuracy**: 92.47%
- 210 of 3,000 images misclassified in the zero-shot setting
- Most confusion occurred between glacier and mountain, likely due to overlapping icy/rocky terrain

### Flowers Recognition
- **Zero-shot accuracy**: 94.79% | **Fine-tuned CNN accuracy**: 92.48%
- 225 of 4,317 images misclassified in the zero-shot setting
- Despite flowers generally being considered a fine-grained classification challenge, this dataset's 5 classes are visually distinct enough that CLIP performed strongly

### Prompt Engineering Experiment (CIFAR-10)

Different prompt templates were tested on a 2,000-image subset to measure their effect on accuracy:

| Prompt Template | Accuracy |
|---|---|
| `"{}"` | 88.25% |
| `"a photo of a {}"` | 90.45% |
| `"a blurry photo of a {}"` | 89.25% |
| `"a photo of the small {}"` | 89.95% |
| `"a picture of a {}"` | 90.40% |

A second experiment on the Flowers dataset (500-image subset) compared a plain prompt against a class-context prompt:

| Prompt Template | Accuracy |
|---|---|
| `"a photo of a {}"` | 96.00% |
| `"a photo of a {}, a type of flower"` | 92.00% |

Interestingly, the additional context ("a type of flower") slightly *reduced* accuracy on this subset. Since the five flower classes are already visually distinct, the added wording likely introduced noise rather than useful disambiguation — a reminder that more descriptive prompts don't always improve results, and their effect depends on how visually similar the target classes are.

## Why Zero-Shot Won on Two Datasets

CLIP is pretrained on hundreds of millions of image-text pairs collected from the internet, giving it broad, general visual knowledge before ever seeing these specific datasets. In contrast, the CNN was fine-tuned for only 3 epochs on relatively small datasets (14,000 images for Intel, 4,300 for Flowers) — not enough data or training time to fully surpass CLIP's pretraining advantage. On CIFAR-10, however, the CNN pulled ahead, likely because CIFAR-10's low-resolution (32x32) images represent a narrower, more specialized visual domain that benefited more from direct supervised training.

## Tech Stack

- **Python 3**
- **PyTorch** — model training and inference
- **Hugging Face Transformers** — CLIP model and processor (`openai/clip-vit-base-patch32`)
- **torchvision** — datasets, ResNet18 architecture, image transforms
- **scikit-learn** — evaluation metrics and confusion matrices
- **Matplotlib** — visualizations
- **Kaggle Notebooks** — development and GPU compute environment

## Project Structure

```
├── zero_shot_clip_classification.ipynb   # Main notebook: all experiments and results
├── final_results.txt                     # Saved summary of accuracy results
└── README.md                             # Project documentation (this file)
```

## How to Run

1. Open the notebook in [Kaggle](https://www.kaggle.com/) or Jupyter with GPU support enabled.
2. Install dependencies:
   ```bash
   pip install transformers torch torchvision scikit-learn matplotlib
   ```
3. If running on Kaggle, add the following datasets via "Add Data":
   - Intel Image Classification (`puneet6060/intel-image-classification`)
   - Flowers Recognition (`alxmamaev/flowers-recognition`)
   
   CIFAR-10 downloads automatically via `torchvision`.
4. Run all cells in order. GPU is recommended (CLIP inference and CNN training run significantly faster).

## Key Takeaways

- Zero-shot models like CLIP can match or exceed traditionally trained models, particularly when labeled training data is limited.
- Prompt wording has a measurable, but dataset-dependent, effect on zero-shot accuracy — there is no single "best" prompt that works universally.
- Visually distinct classes (e.g., natural scenes, flower species in this dataset) favor zero-shot approaches, while narrower, low-resolution, or highly specialized domains (e.g., CIFAR-10) may still benefit more from dedicated supervised training.

## Future Improvements

- Test larger CLIP variants (e.g., ViT-L/14) to measure accuracy/speed trade-offs
- Expand the CNN baseline with more training epochs or data augmentation for a fairer comparison
- Explore few-shot fine-tuning of CLIP as a middle ground between zero-shot and full supervised training
- Extend prompt engineering experiments across all three datasets

## Author

Sazid Hossain
Kaggle Notebook: [link to your public Kaggle notebook]
