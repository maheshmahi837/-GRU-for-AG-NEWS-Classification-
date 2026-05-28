# AG News Text Classification using GRU Networks in PyTorch

## Overview

This repository contains the implementation of GRU-based deep learning models for text classification using the AG News dataset in PyTorch. The project was developed as part of the **AI61002: Deep Learning Foundations and Applications** course assignment.

The project focuses on building and comparing two recurrent neural network architectures using `nn.GRUCell` and `nn.GRU` for multi-class news classification. It includes complete NLP preprocessing, sequence modeling, training, evaluation, and performance comparison.

---

## Project Objectives

* Load and preprocess the AG News dataset
* Perform text cleaning and tokenization
* Convert text into numerical representations
* Build GRU-based text classification models
* Compare custom `nn.GRUCell` implementation with built-in `nn.GRU`
* Train and evaluate both models
* Analyze performance using classification metrics

---

## Technologies Used

* Python
* PyTorch
* TorchText
* NumPy
* Matplotlib
* Scikit-learn
* Jupyter Notebook

---

## Dataset

The project uses the **AG News** dataset consisting of news articles classified into four categories:

* World
* Sports
* Business
* Science/Technology

The dataset was loaded using `torchtext.datasets`.

---

## Data Preprocessing

The following NLP preprocessing steps were implemented:

* Stopword removal
* Punctuation removal
* Tokenization
* Vocabulary generation using `torchtext.vocab`
* Text-to-index conversion
* Sequence padding
* One-hot encoding of labels

### Sequence Length

```bash id="o7h9ye"
Max Sequence Length = 256
```

---

## Model Architectures

Two GRU-based architectures were implemented and compared.

### Model 1 — Custom GRU using `nn.GRUCell`

Features:

* Two GRU layers
* Dropout between GRU layers
* Manual hidden state handling
* Custom recurrent processing

### Model 2 — Built-in GRU using `nn.GRU`

Features:

* Two-layer GRU architecture
* Built-in recurrent operations
* Dropout regularization
* Simplified implementation

---

## Training Details

### Optimizer

* Adam Optimizer

### Loss Function

* Cross Entropy Loss

### Training Features

* Validation monitoring
* Loss curve plotting
* Hyperparameter tuning
* Performance comparison between architectures

---

## Evaluation Metrics

Both models were evaluated using:

* Accuracy
* Precision
* Recall
* Confusion Matrix

The performance comparison demonstrated that the built-in `nn.GRU` model achieved results comparable to the custom `nn.GRUCell` implementation.

---

## Project Structure

```bash id="vc4u5i"
├── assignment_3.ipynb          # Jupyter Notebook implementation
├── assignment_3.py             # Python script implementation
├── models/                     # Saved trained models
├── plots/                      # Training and validation curves
├── results/                    # Evaluation outputs
├── README.md                   # Project documentation
```

---

## How to Run

### Clone the Repository

```bash id="9n5m74"
git clone <your-repository-link>
cd <repository-folder>
```

### Install Dependencies

```bash id="6h5sdp"
pip install torch torchtext numpy matplotlib scikit-learn
```

### Run the Notebook

```bash id="o52h8k"
jupyter notebook
```

Open:

```bash id="jlwm48"
assignment_3.ipynb
```

---

## Results

The GRU-based models successfully learned semantic patterns from news articles and achieved strong text classification performance.

Key observations:

* GRU networks effectively captured sequential text dependencies
* Built-in `nn.GRU` simplified implementation while maintaining performance
* Dropout helped reduce overfitting
* Proper preprocessing significantly improved classification accuracy

---

## Learning Outcomes

This project helped in understanding:

* Natural Language Processing (NLP)
* Text preprocessing techniques
* Sequence modeling
* Recurrent Neural Networks (RNNs)
* Gated Recurrent Units (GRUs)
* Custom recurrent layer implementation
* Text classification pipelines
* Deep learning for NLP tasks

---

## Assignment Reference

This implementation follows the requirements specified in the AI61002 Assignment 3 guidelines.

---

## Author

Mahesh Kusireddy

AI61002 - Deep Learning Foundations and Applications
Indian Institute of Technology Kharagpur
