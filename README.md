# CNN Image Classifier (Fashion-MNIST)

A simple Convolutional Neural Network (CNN) built using **PyTorch** to classify images from the **Fashion-MNIST** dataset.  
The project includes training, evaluation, and inference on external images.

---

## 📁 Project Structure

```
cnn-image-classifier-pytorch/
│
├── models/
│   └── fashion_cnn_best.pth          # Saved trained model
│
├── notebooks/
│   └── cnn.ipynb                     # Full training notebook
│
├── src/
│   ├── model.py                      # CNN architecture
│   └── inference.py                  # Load model + predict on images
│
└── requirements.txt                  # Package dependencies
```

---

## 🚀 How to Run Inference

### 1. Install dependencies
```
pip install -r requirements.txt
```

### 2. Run prediction on an image
Navigate to the `src/` folder and run:

```
python inference.py
```

You can change the image path inside the script.

---

## 🧠 Model Overview

- 2 convolutional layers  
- ReLU activation  
- MaxPooling (2×2)  
- 2 fully connected layers  
- Trained on 10 Fashion-MNIST classes

Achieves **~90% accuracy** on the test dataset.

---

## 📦 Dataset

Fashion-MNIST is automatically downloaded via `torchvision` when running the notebook.

Classes:
```
0 = T-shirt/top      5 = Sandal
1 = Trouser          6 = Shirt
2 = Pullover         7 = Sneaker
3 = Dress            8 = Bag
4 = Coat             9 = Ankle boot
```

---

## 🖼 Notebook

The training process, evaluation, and sample predictions are available in:

```
notebooks/cnn.ipynb
```

---



This is a personal practice project.
