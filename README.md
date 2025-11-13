# 🩺 Pneumonia Detection using Deep Learning (MobileNetV2)

This project builds an end-to-end deep-learning pipeline to detect **Pneumonia** from chest X-ray images using **MobileNetV2 transfer learning**.  
It includes dataset handling, duplicate removal, class-imbalance correction, model training, evaluation, and visualization.

---

## 🚀 Features

- 📥 **Automatic dataset download** from Kaggle using `kagglehub`
- 🧽 **Duplicate image detection** using MD5 hashing (removes noisy duplicates)
- 📊 **70/20/10 dataset split** (Train/Val/Test)
- 🎛️ **Advanced Image Augmentation** for robustness:
  - rotation, brightness shift, zoom, shift, shear, flip
- 🎯 **Transfer Learning with MobileNetV2** (ImageNet weights)
- 🔧 **Fine-tuning last 4 layers** + L1/L2 regularization
- ⚖️ **Class weight balancing** to handle dataset imbalance
- 🧠 **Optimizers & Callbacks**
  - AdamW optimizer
  - EarlyStopping
  - ModelCheckpoint
  - ReduceLROnPlateau
- 📈 **Training Graphs**
  - Accuracy curves
  - Loss curves
- 📉 **Evaluation Metrics**
  - Confusion matrix (heatmap)
  - Precision, Recall, F1-score
  - Full classification report
- 💾 **Final model saved as**:  
  `best_model_pneumonia.keras`

---

## 📁 Dataset

Dataset: **Chest X-Ray Pneumonia Dataset**  
(From Kaggle: `paultimothymooney/chest-xray-pneumonia`)

Downloaded automatically into:

