# Federated Learning TP2 Project

An advanced Federated Learning project using Flower and PyTorch with privacy-preserving mechanisms on the FashionMNIST dataset.

This project extends the classical Federated Learning workflow by integrating concepts related to secure and privacy-aware distributed machine learning.

---

# Overview

This project simulates a Federated Learning environment where multiple distributed clients collaboratively train a global deep learning model without sharing their raw data.

The project focuses on:
- Federated training
- Non-IID client distributions
- Privacy-preserving learning
- Distributed CNN training
- Federated aggregation
- Accuracy evaluation and visualization

---

# Features

- Federated Learning simulation
- Flower framework integration
- CNN image classification
- FashionMNIST dataset
- Non-IID data partitioning
- Multiple federated clients
- Privacy-aware training concepts
- Validation accuracy tracking
- Result visualization

---

# Technologies Used

- Python 3
- PyTorch
- Flower (FLWR)
- NumPy
- Matplotlib
- torchvision

---

# Project Structure

```bash
Federated_Learning_TP2_Project/
│
├── tp2.py                 # Main federated learning workflow
├── dataset.py             # Dataset partitioning
├── plot.py                # Visualization utilities
├── results.json           # Simulation metrics
├── val_accuracy.png       # Accuracy graph
├── client_data/           # Distributed datasets
│   ├── client_0.pt
│   ├── client_1.pt
│   └── ...
│
└── README.md
```

---

# Dataset

This project uses the FashionMNIST dataset.

Dataset characteristics:
- 70,000 grayscale images
- 10 clothing categories
- 28×28 image resolution

The dataset is distributed across clients using a non-IID partitioning strategy based on the Dirichlet distribution.

---

# Federated Learning Workflow

## 1. Data Distribution
The dataset is partitioned among several clients.

## 2. Local Client Training
Each client trains locally on private data.

## 3. Privacy-Aware Federated Updates
Clients send model updates instead of raw data.

## 4. Federated Aggregation
The server aggregates client weights using Federated Averaging (FedAvg).

## 5. Global Evaluation
The global model is evaluated after each communication round.

## 6. Visualization
Accuracy metrics are stored and plotted.

---

# CNN Architecture

The CNN model contains:
- Convolutional layers
- ReLU activations
- MaxPooling layers
- Fully connected layers
- Classification output layer

The model is optimized for FashionMNIST image classification tasks.

---

# Non-IID Client Distribution

The project simulates realistic federated environments using heterogeneous client datasets.

A Dirichlet distribution parameter `alpha` controls the level of heterogeneity:
- Lower alpha → highly non-IID data
- Higher alpha → more balanced distributions

---

# Privacy-Preserving Concepts

This project introduces concepts related to:
- Data privacy in Federated Learning
- Decentralized model training
- Secure parameter sharing
- Privacy-aware aggregation

Clients never share raw training data with the server.

---

# Installation

## Clone the Repository

```bash
git clone https://github.com/Khalil-Boukhalil/Federated_Learning_TP2_Project.git
cd Federated_Learning_TP2_Project
```

## Install Dependencies

```bash
pip install torch torchvision flwr matplotlib numpy
```

---

# Running the Project

## Step 1 — Generate Client Datasets

```bash
python dataset.py
```

## Step 2 — Run Federated Learning Simulation

```bash
python tp2.py
```

## Step 3 — Plot Results

```bash
python plot.py
```

---

# Results

The simulation generates:
- Federated training metrics
- Validation accuracy evolution
- Aggregated performance statistics

Generated files:
- `results.json`
- `val_accuracy.png`

---

# Example Federated Workflow

```text
Client Local Training
          ↓
Model Parameter Updates
          ↓
Federated Aggregation
          ↓
Updated Global Model
          ↓
Next Communication Round
```

---

# Key Concepts Demonstrated

- Federated Learning
- Distributed AI
- Privacy-preserving Machine Learning
- CNN image classification
- Non-IID data partitioning
- Federated Averaging (FedAvg)
- Decentralized training systems

---

# Future Improvements

Possible future enhancements:
- Differential Privacy implementation
- Secure Aggregation protocols
- Real distributed deployment
- GPU acceleration
- Larger datasets
- Transformer-based models
- Communication optimization

---

# Educational Purpose

This project was developed for educational purposes to better understand:
- Federated Learning systems
- Client-server ML workflows
- Data privacy challenges
- Distributed deep learning

---

# Author

Khalil Bou Khalil

Master’s Student in Artificial Intelligence for Connected Industries (AI4CI)  
CNAM Paris

GitHub:  
https://github.com/Khalil-Boukhalil

---

# License

This project is intended for educational and research purposes.
