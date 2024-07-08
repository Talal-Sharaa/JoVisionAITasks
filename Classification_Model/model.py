# Import necessary libraries
import os
import shutil
from PIL import Image
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define the input shape
input_shape = (101, 180, 3)

# Load the base model (pre-trained on ImageNet)
base_model = MobileNetV2(input_shape=input_shape, include_top=False, weights='imagenet')

# Freeze the base model
base_model.trainable = False

# Add custom layers on top of the base model
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(4, activation='softmax')  # 4 output classes
])

# Compile the model
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# Data augmentation and normalization for training
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,  # 20% for validation
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

train_generator = train_datagen.flow_from_directory(
    r'C:\Users\talal\Downloads\TIRES',
    target_size=(101, 180),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    r'C:\Users\talal\Downloads\TIRES',
    target_size=(101, 180),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_steps=validation_generator.samples // validation_generator.batch_size,
    epochs=25,
    validation_data=validation_generator
)

# Save the trained model
model.save('modal.keras')

# Define class labels
class_labels = {0: 'SIDE', 1: 'OCR', 2: 'MultiTires', 3: 'Front'}

# Preprocess and classify image
def preprocess_image(img_path, target_size=(180, 101)):
    img = Image.open(img_path)
    img = img.resize(target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)  # Preprocess for MobileNetV2
    return img_array

def classify_image(img_path):
    img_array = preprocess_image(img_path, target_size=(180, 101))
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction, axis=1)[0]
    return class_labels[predicted_class]

# Classify all images in a folder
def classify_images_in_folder(input_folder, output_base_folder):
    for class_name in class_labels.values():
        os.makedirs(os.path.join(output_base_folder, class_name), exist_ok=True)
    
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(input_folder, filename)
            predicted_class = classify_image(img_path)
            output_folder = os.path.join(output_base_folder, predicted_class)
            shutil.move(img_path, os.path.join(output_folder, filename))
            print(f'Classified {filename} as {predicted_class} and moved to {output_folder}')

# Paths to directories
input_folder = r'C:\Users\talal\Downloads\random'
output_base_folder =  r'C:\Users\talal\Downloads\random\classified'

# Classify all images in the input folder
classify_images_in_folder(input_folder, output_base_folder)
