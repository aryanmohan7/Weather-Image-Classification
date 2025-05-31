# Weather Image Classification using Deep Learning & Feature Optimization

This project aims to classify weather phenomena from images using a deep learning-based pipeline enhanced with machine learning classifiers and genetic feature optimization.  
The dataset comprises images across 11 distinct weather categories such as **fog**, **lightning**, **snow**, and **sandstorm**.

The pipeline follows a modular and hybrid approach involving **transfer learning with ResNet101**, **feature extraction**, **XGBoost classification**, and **genetic algorithm-based feature selection**, enabling highly accurate and interpretable classification of weather types from raw images.

---

## Key Features

- **Custom Dataset Loader**: Structured image directory loading and labeling using PyTorch `Dataset` class.
- **Data Augmentation**: Applied transformations like resizing, flipping, and rotation for generalization.
- **Transfer Learning**: Fine-tuned a pretrained `ResNet101` model on the weather image dataset for deep feature extraction.
- **Model Evaluation**: Evaluated on training, validation, and test sets using precision, recall, and F1 scores.
- **XGBoost Classification**: Used deep features to train an optimized XGBoost classifier via `GridSearchCV`.
- **Genetic Algorithm for Feature Selection**: Reduced feature dimensionality and improved performance using a custom genetic mutation-based optimizer.
- **Pipeline Comparison**: Compared performance across three strategies:
  - `ResNet101 + Softmax`
  - `ResNet101 + XGBoost`
  - `ResNet101 + XGBoost + Genetic Feature Selection`

---

## Results

- **~92% Training Accuracy**
- **~89% Validation Accuracy**
- **~86% Test Accuracy**

Performance was analyzed using classification reports and comparative bar plots for Accuracy and F1-score across all pipelines.  
The optimized pipeline with genetic feature selection provided improved interpretability and computational efficiency with minimal performance trade-offs.

---

## Tech Stack

- **Frameworks**: PyTorch, scikit-learn, XGBoost
- **Tools**: GridSearchCV, Matplotlib, Seaborn, PIL, tqdm
- **Models Used**:
  - ResNet101 (Transfer Learning)
  - XGBoost Classifier
  - Genetic Algorithm (Custom Implementation)
