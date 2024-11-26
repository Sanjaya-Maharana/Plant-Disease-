import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from tensorflow.keras.applications import VGG16

# Set paths to different image types
data_dir = r'D:\PlantVillage-Dataset\raw'  # Root directory with color, grayscale, and segmented images

# Define subdirectories for color, grayscale, and segmented data
color_dir = os.path.join(data_dir, 'color')
grayscale_dir = os.path.join(data_dir, 'grayscale')
segmented_dir = os.path.join(data_dir, 'segmented')

# Image size and parameters
.
image_size = (128, 128)  # Resize images to 128x128
batch_size = 32

# Data augmentation and preprocessing for color, grayscale, and segmented images
datagen = ImageDataGenerator(
    rescale=1./255,  # Normalize pixel values between 0 and 1
    validation_split=0.2  # Split 20% of data for validation
)

# Load training and validation datasets for color images
train_generator_color = datagen.flow_from_directory(
    color_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='training',  # Training set
    shuffle=True
)

validation_generator_color = datagen.flow_from_directory(
    color_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'  # Validation set
)

# Load grayscale images for SSL
train_generator_grayscale = datagen.flow_from_directory(
    grayscale_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='training',  # Training set
    shuffle=True
)

# Load segmented images for SSL
train_generator_segmented = datagen.flow_from_directory(
    segmented_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='training',  # Training set
    shuffle=True
)

# Use a pretrained CNN (VGG16) as a feature extractor for SSL and SVM classification
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(128, 128, 3))
base_model.trainable = False  # Freeze the pretrained layers

# Build a model using the VGG16 base and additional layers
model = Sequential([
    base_model,
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(train_generator_color.num_classes, activation='softmax')  # Output layer with softmax activation
])

# Compile the model
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Model summary
model.summary()

# Train the CNN model with color data (SSL can also be applied to grayscale and segmented images)
history = model.fit(
    train_generator_color,
    validation_data=validation_generator_color,
    epochs=10,  # Adjust based on resources
    steps_per_epoch=train_generator_color.samples // batch_size,
    validation_steps=validation_generator_color.samples // batch_size
)

# Extract features from the CNN for training an SVM classifier
def extract_features(generator, model):
    features = []
    labels = []
    for i in range(len(generator)):
        x_batch, y_batch = generator[i]
        features_batch = model.predict(x_batch)
        features.append(features_batch)
        labels.append(y_batch)
    return np.vstack(features), np.vstack(labels)

# Extract features from the color images
train_features, train_labels = extract_features(train_generator_color, base_model)
val_features, val_labels = extract_features(validation_generator_color, base_model)

# Reshape the features to be used with SVM
train_features = train_features.reshape(train_features.shape[0], -1)
val_features = val_features.reshape(val_features.shape[0], -1)

# Train an SVM classifier using the extracted features
svm = SVC(kernel='linear')
svm.fit(train_features, np.argmax(train_labels, axis=1))

# Evaluate the SVM classifier
val_predictions = svm.predict(val_features)
val_accuracy = accuracy_score(np.argmax(val_labels, axis=1), val_predictions)
print(f"SVM Validation Accuracy: {val_accuracy * 100:.2f}%")

# Save the CNN model for later use
model.save('plant_disease_model_with_ssl_svm.h5')

# Plot training & validation accuracy/loss values
import matplotlib.pyplot as plt

def plot_training_history(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc) + 1)

    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, 'b', label='Training accuracy')
    plt.plot(epochs, val_acc, 'r', label='Validation accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, 'b', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    plt.show()

# Plot the results
plot_training_history(history)
