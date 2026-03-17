# Assignment 3 — MLOps with MLflow & GitHub Actions

## Overview
This project trains a CNN on the MNIST dataset with full MLflow experiment tracking
and an automated GitHub Actions CI pipeline.

## Files
- `train.py` — PyTorch training script with MLflow instrumentation
- `requirements.txt` — Python dependencies
- `.github/workflows/ml-pipeline.yml` — GitHub Actions CI pipeline

## How to Run
```bash
pip install -r requirements.txt
mlflow ui --port 5000
python train.py --lr 0.001 --batch_size 64 --epochs 5
```

## Results
Best run: lr=0.001, batch_size=64 → Val Accuracy = 99.19%
