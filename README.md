# CIFAR-10 Image Classification using CNN

A deep learning project that uses a **Convolutional Neural Network (CNN)** to classify images from the **CIFAR-10 dataset** into 10 different categories.

The model is built using **TensorFlow/Keras** and includes convolutional layers, batch normalization, max pooling, dropout, and early stopping to improve generalization and reduce overfitting.

## 📌 Project Overview

Image classification is one of the fundamental applications of computer vision. In this project, a CNN is trained on the CIFAR-10 dataset to recognize and classify small RGB images into one of 10 classes.

The model learns visual features such as edges, shapes, textures, and patterns through multiple convolutional layers.

### CIFAR-10 Classes

The dataset contains 10 classes:

* ✈️ Airplane
* 🚗 Automobile
* 🐦 Bird
* 🐱 Cat
* 🦌 Deer
* 🐶 Dog
* 🐸 Frog
* 🐴 Horse
* 🚢 Ship
* 🚚 Truck

---

## 🎯 Objective

The main objective of this project is to build a CNN-based image classification model that can accurately classify CIFAR-10 images into their respective categories.

The project focuses on:

* Loading and preprocessing the CIFAR-10 dataset
* Building a CNN architecture using TensorFlow/Keras
* Applying Batch Normalization and Dropout for better generalization
* Training the model with validation monitoring
* Using Early Stopping to prevent unnecessary training
* Evaluating model performance on the test dataset
* Visualizing predictions with predicted labels and confidence scores

---

## 🛠️ Technologies Used

* **Python**
* **TensorFlow / Keras**
* **NumPy**
* **Matplotlib**
* **Scikit-learn**
* **Google Colab / Jupyter Notebook**

---

## 📂 Dataset

This project uses the **CIFAR-10 dataset**, which contains:

* **50,000 training images**
* **10,000 test images**
* **32 × 32 pixel RGB images**
* **10 classes**

Each image has 3 color channels:

```text
Image Shape = (32, 32, 3)
```

The original CIFAR-10 binary batches are loaded using Python's `pickle` module.

---

## 🔄 Data Preprocessing

The CIFAR-10 binary data is initially stored in the shape:

```text
(number_of_images, 3, 32, 32)
```

The data is reshaped and transposed into the format expected by Keras:

```text
(number_of_images, 32, 32, 3)
```

Pixel values originally range from:

```text
0 - 255
```

They are normalized to:

```text
0 - 1
```

using:

```python
X_train = X_train.astype("float32") / 255.0
X_test = X_test.astype("float32") / 255.0
```

This helps the neural network train more efficiently.

---

## 🧠 CNN Architecture

The model consists of three convolutional blocks followed by fully connected layers.

```text
Input
(32 × 32 × 3)
       ↓
Conv2D (32 filters, 3×3)
       ↓
Batch Normalization
       ↓
Max Pooling
       ↓
Dropout (0.25)
       ↓
Conv2D (64 filters, 3×3)
       ↓
Batch Normalization
       ↓
Max Pooling
       ↓
Dropout (0.25)
       ↓
Conv2D (128 filters, 3×3)
       ↓
Batch Normalization
       ↓
Max Pooling
       ↓
Dropout (0.30)
       ↓
Flatten
       ↓
Dense (128)
       ↓
Batch Normalization
       ↓
Dropout (0.50)
       ↓
Dense (10, Softmax)
```

### Why these layers are used

**Conv2D**

Extracts spatial features such as edges, textures, shapes, and patterns from the images.

**Batch Normalization**

Helps stabilize and speed up training by normalizing activations within the network.

**MaxPooling2D**

Reduces the spatial dimensions of feature maps while retaining important features.

**Dropout**

Randomly disables a fraction of neurons during training, helping reduce overfitting.

**Flatten**

Converts the extracted feature maps into a one-dimensional vector before the dense layers.

**Dense**

Learns higher-level combinations of the extracted features for classification.

**Softmax**

Produces probabilities for the 10 CIFAR-10 classes.

---

## ⚙️ Model Configuration

The model is compiled using:

```python
cnn.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)
```

### Training Configuration

| Parameter         | Value                           |
| ----------------- | ------------------------------- |
| Optimizer         | Adam                            |
| Loss Function     | Sparse Categorical Crossentropy |
| Batch Size        | 128                             |
| Maximum Epochs    | 30                              |
| Activation        | ReLU                            |
| Output Activation | Softmax                         |
| Input Shape       | 32 × 32 × 3                     |
| Output Classes    | 10                              |

---

## 🛡️ Overfitting Prevention

Several techniques were incorporated to improve model generalization:

### Batch Normalization

Batch normalization is applied after convolutional layers and the dense layer to stabilize the training process.

### Dropout

Different dropout rates are used throughout the network:

```text
0.25 → 0.25 → 0.30 → 0.50
```

This prevents the network from relying too heavily on specific neurons.

### Early Stopping

The model uses:

```python
EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)
```

Training stops when validation loss does not improve for 5 consecutive epochs, and the best-performing weights are restored.

---

## 📊 Results

The model achieved the following performance:

```text
Best Validation Accuracy : 79.81%
Validation Loss          : 0.5806

Training Accuracy        : 78.62%
Training Loss            : 0.6146
```

The best validation performance was achieved around **Epoch 19**, after which the best weights were restored using Early Stopping.

### Final Test Accuracy

```text
Test Accuracy ≈ 79.81%
```

This indicates that the trained CNN can correctly classify approximately **8 out of every 10 CIFAR-10 test images**.

---

## 🔍 Prediction Visualization

The project also includes a visualization function that randomly selects test images and displays:

* Original image
* Actual class
* Predicted class
* Prediction confidence

Correct predictions are displayed in **green**, while incorrect predictions are displayed in **red**.

Example output format:

```text
True: cat
Pred: cat (87.4%)
```

or

```text
True: dog
Pred: cat (64.2%)
```

This provides an intuitive way to inspect how the model performs on individual images.

---

## 📈 Model Performance

The training process tracks:

* Training Accuracy
* Training Loss
* Validation Accuracy
* Validation Loss

These metrics can be used to analyze the model's learning behavior and identify potential overfitting or underfitting.

---

## 🚀 How to Run

### 1. Clone the repository

```bash
git clone https://github.com/your-username/cifar10-image-classification-cnn.git
```

### 2. Install dependencies

```bash
pip install tensorflow numpy matplotlib scikit-learn
```

### 3. Download CIFAR-10

Download the CIFAR-10 Python dataset and place the extracted files in the appropriate project directory.

The code expects files such as:

```text
data_batch_1
data_batch_2
data_batch_3
data_batch_4
data_batch_5
test_batch
```

### 4. Run the notebook

Open the Jupyter Notebook or Google Colab notebook and execute the cells sequentially.

---

## 📁 Project Structure

```text
cifar10-image-classification-cnn/
│
├── cifar10_cnn.ipynb
├── README.md
├── requirements.txt
│
└── dataset/
    ├── data_batch_1
    ├── data_batch_2
    ├── data_batch_3
    ├── data_batch_4
    ├── data_batch_5
    └── test_batch
```

> Dataset files can be excluded from the repository because of their size. Users can download the CIFAR-10 dataset separately.

---

## 💡 Key Learnings

Through this project, I gained practical experience with:

* Convolutional Neural Networks
* Image preprocessing
* TensorFlow/Keras model development
* Convolution and pooling operations
* Batch Normalization
* Dropout regularization
* Early Stopping
* Multi-class image classification
* Model evaluation
* Prediction confidence visualization
* Understanding training vs. validation performance

---

## 🔮 Future Improvements

The model could potentially be improved further by experimenting with:

* Data augmentation
* Learning-rate scheduling
* More advanced CNN architectures
* Global Average Pooling
* L2 regularization
* Hyperparameter tuning
* Transfer learning
* Confusion matrix and per-class performance analysis

---

## 👩‍💻 Author

**Sneha Sharma**
B.Tech – Computer Science Engineering
