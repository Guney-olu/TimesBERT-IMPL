# TimesBERT-AdPerformance

This repository contains an implementation of **TimesBERT** (from the paper [TimesBERT: A BERT-Style Foundation Model for Time Series Understanding](https://arxiv.org/abs/2502.21245)) adapted for **ad performance data**.

---

## 📌 Overview

TimesBERT is a **BERT-style encoder-only Transformer** designed to learn structured representations of multivariate time series.  
Instead of focusing only on forecasting, TimesBERT enables a broader paradigm called **time series understanding**, including:
- Classification  
- Anomaly detection  
- Imputation  
- Short-term forecasting  

This implementation adapts TimesBERT to **Facebook Ads performance datasets**, pretraining on ad-level time series features (spend, impressions, CTR, ROAS, etc.).

---

## ⚙️ Features

- **Masked Patch Modeling (MPM):** Learns temporal representations by masking and reconstructing patches.  
- **Functional Token Prediction (FTP):** Predicts variate-level and domain-level tokens for better multi-granularity structure.  
- **Multi-domain Training:** Handles ad campaigns across different domains (campaign names).  
- **Scalable Training:** Supports multi-GPU training with PyTorch `DataParallel`.  
- **SafeTensors Saving:** Uses `safetensors` for efficient checkpoint saving.

---

## 📂 Project Structure

```
├── data/                          # Input CSV files (one per ad/campaign)
├── timesbert_ad_performance/      # Output directory for checkpoints & final models
│   ├── checkpoints/               # Intermediate checkpoints
│   ├── model.safetensors          # Final trained model
│   ├── config.json                # Saved model config
│   └── domain_map.json            # Domain (campaign) mappings
├── train.py                       # Main training script (this repo's core file)
└── README.md                      # Project documentation
```

---

## 🚀 Training

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Place your ad performance CSV files under the `data/` folder.

3. Run training:
```bash
python train.py
```

The model will:
- Pretrain using MPM + FTP  
- Save checkpoints every 10 epochs  
- Save the final model at the end  

---

## 🔎 Inference

After training, you can run masked patch reconstruction on validation data:

```python
from train import run_inference
plot_path = run_inference("timesbert_ad_performance/checkpoints/model_epoch_10.safetensors", epoch=10)
print("Saved reconstruction plot:", plot_path)
```

---
## 📚 Reference

If you use this repo, please cite the original paper:

**TimesBERT: A BERT-Style Foundation Model for Time Series Understanding**  
Haoran Zhang, Yong Liu, Yunzhong Qiu, Haixuan Liu, Zhongyi Pei, Jianmin Wang, Mingsheng Long  
[ArXiv:2502.21245](https://arxiv.org/abs/2502.21245)

---

## 📝 License

This implementation is for research purposes only.  
Please check the original paper for licensing details.

---
