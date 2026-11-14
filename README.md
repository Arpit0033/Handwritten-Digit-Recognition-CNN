# Handwritten Digit Recognition using CNN

## 1. Project Overview
This project uses a Convolutional Neural Network (CNN) to recognize handwritten digits from 0 to 9.

The model is trained using the MNIST dataset, which contains 28×28 grayscale images of handwritten digits.

## 2. Problem Statement
How can a neural network learn visual patterns in handwritten digits and correctly classify unseen examples?

## 3. Project Objectives
- Load and inspect the MNIST dataset.
- Normalize pixel values.
- Build a CNN model using TensorFlow/Keras.
- Train and validate the model.
- Evaluate model performance on test data.
- Generate learning curves and a confusion matrix.
- Display sample predictions.

## 4. Technologies Used
- Python 3.13
- TensorFlow / Keras
- NumPy
- Matplotlib
- Scikit-learn
- VS Code

## 5. Dataset

MNIST handwritten digit dataset:

- Original training images: 60,000
- Training images after split: 54,000
- Validation images: 6,000
- Testing images: 10,000
- Image dimensions: 28 × 28 pixels
- Color format: Grayscale
- Classes: 0 to 9
- Total images: 70,000
## 6. CNN Model Architecture
The model contains:
1. Convolutional Layer (32 filters)
2. Max Pooling Layer
3. Convolutional Layer (64 filters)
4. Max Pooling Layer
5. Flatten Layer
6. Dense Layer (64 neurons)
7. Dropout Layer
8. Output Layer (10 neurons)

## 7. Model Results
Latest recorded test results:

- Test Accuracy: 99.18%
- Test Loss: 0.0243
- Test Dataset: 10,000 images

## 8. Project Outputs
- Accuracy Graph
- Loss Graph
- Confusion Matrix
- Sample Predictions

## 9. Real-World Applications
- Digit recognition in documents.
- Bank cheque digitization.
- Postal code recognition.
- Automated form processing.

These are potential applications; this educational model is not yet validated for production use.

## 10. Limitations
- Trained only on the MNIST dataset.
- Works with 28×28 grayscale images.
- May misclassify unusual handwriting.
- Recognizes individual digits rather than complete handwritten text.
- Real-world performance may differ from test accuracy.

## 11. How to Run

Activate the virtual environment:

```powershell
.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
pip install tensorflow numpy matplotlib scikit-learn
```

Run the project:

```powershell
python notebooks/digit_recognition.py
```

## 12. Conclusion
The project successfully demonstrates how CNNs can learn visual patterns and classify handwritten digits. The trained model achieved 99.18% test accuracy on the MNIST dataset.
