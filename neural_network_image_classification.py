# Task 1: Data Preparation
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# Load MNIST dataset (handwritten digits 0–9)
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Normalize pixel values to range [0,1]
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

# Reshape to include channel dimension (28x28x1)
X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)

# Convert labels to one-hot encoding
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Split train into train/validation (80/20)
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

print("Training data:", X_train.shape)
print("Validation data:", X_val.shape)
print("Test data:", X_test.shape)

# Task 2: Baseline CNN Model

def create_baseline_model():
    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
        MaxPooling2D(2,2),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(10, activation='softmax')
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), 
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])
    return model

baseline_model = create_baseline_model()
history_baseline = baseline_model.fit(X_train, y_train, 
                                      epochs=10, batch_size=32, 
                                      validation_data=(X_val, y_val), verbose=2)

# Task 3: Hyperparameter Tuning

def build_tuned_model(filters=64, kernel_size=3, dropout_rate=0.3, optimizer_choice='Adam', lr=0.001):
    if optimizer_choice == 'Adam':
        opt = Adam(learning_rate=lr)
    elif optimizer_choice == 'RMSprop':
        opt = RMSprop(learning_rate=lr)
    else:
        opt = SGD(learning_rate=lr)

    model = Sequential([
        Conv2D(filters, (kernel_size, kernel_size), activation='relu', input_shape=(28,28,1)),
        MaxPooling2D(2,2),
        Dropout(dropout_rate),
        Conv2D(filters*2, (3,3), activation='relu'),
        MaxPooling2D(2,2),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(dropout_rate),
        Dense(10, activation='softmax')
    ])
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Example tuning configurations
tuned_models = [
    {'filters': 32, 'kernel_size': 3, 'dropout_rate': 0.25, 'optimizer_choice': 'Adam', 'lr': 0.001},
    {'filters': 64, 'kernel_size': 5, 'dropout_rate': 0.3, 'optimizer_choice': 'Adam', 'lr': 0.001},
    {'filters': 64, 'kernel_size': 3, 'dropout_rate': 0.4, 'optimizer_choice': 'RMSprop', 'lr': 0.0005},
    {'filters': 128, 'kernel_size': 3, 'dropout_rate': 0.5, 'optimizer_choice': 'Adam', 'lr': 0.0001},
]

results = []

for params in tuned_models:
    print(f"\nTraining with params: {params}")
    model = build_tuned_model(**params)
    history = model.fit(X_train, y_train, epochs=8, batch_size=64, 
                        validation_data=(X_val, y_val), verbose=2)
    val_acc = max(history.history['val_accuracy'])
    results.append((params, val_acc))

print("\nTuning Results:")
for r in results:
    print(r)

# Select best model (manually or automatically)
best_params = max(results, key=lambda x: x[1])[0]
print("\nBest Hyperparameters:", best_params)

best_model = build_tuned_model(**best_params)
history_best = best_model.fit(X_train, y_train, epochs=10, batch_size=64,
                              validation_data=(X_val, y_val), verbose=2)

# Task 4: Model Evaluation
test_loss, test_acc = best_model.evaluate(X_test, y_test, verbose=0)
print("Test Accuracy:", round(test_acc*100, 2), "%")

# Predictions
y_pred = best_model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

# Classification Report
print("\nClassification Report:")
print(classification_report(y_true, y_pred_classes, digits=4))

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred_classes)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()

# Task 5: Analysis
"""
Different hyperparameters affect how well the model learns.

Filters
Increasing the number of filters helps the model detect more features in images, which usually improves accuracy.
However, using too many filters can cause the model to memorize the training data instead of learning general patterns, which leads to overfitting.

Kernel Size
A kernel size of 3×3 works best in most cases because it captures important details without adding unnecessary complexity.
Larger kernels increase computation and do not always improve performance.

Learning Rate
The learning rate controls how fast the model learns. If it is too high, training becomes unstable and the model may not converge.
If it is too low, the training process becomes slow but more stable.

Batch Size
Smaller batch sizes usually help the model generalize better, but training takes more time.
Larger batch sizes make training faster, but may slightly reduce accuracy.

Dropout
Dropout helps prevent overfitting by randomly deactivating some neurons during training.
However, too much dropout can make the model too weak and lead to underfitting.

Optimizer
The optimizer controls how the model updates its weights. Adam is commonly used because it performs well in most cases without requiring much tuning.

Overfitting vs Underfitting

Overfitting
Overfitting occurs when the model performs very well on training data but poorly on validation or test data. 
A common sign is high training accuracy and low validation accuracy.
This can be reduced by using dropout, data augmentation, or simplifying the model.

Underfitting
Underfitting occurs when the model cannot learn the data properly. In this case, both training and validation accuracy remain low.
This can be improved by increasing model complexity, such as adding more layers or filters.

"""

# Task 6: Visualization
def plot_history(hist, title):
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.plot(hist.history['accuracy'], label='Train Acc')
    plt.plot(hist.history['val_accuracy'], label='Val Acc')
    plt.legend(); plt.title(title + " - Accuracy")

    plt.subplot(1,2,2)
    plt.plot(hist.history['loss'], label='Train Loss')
    plt.plot(hist.history['val_loss'], label='Val Loss')
    plt.legend(); plt.title(title + " - Loss")
    plt.show()

plot_history(history_baseline, "Baseline Model")
plot_history(history_best, "Best Tuned Model")
