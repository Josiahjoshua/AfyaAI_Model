import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG19
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import pydicom

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

#LOAD AND PREPROCESS DATA
def load_dicom_images(data_dir):
    images = []
    labels = []
    for class_dir in os.listdir(data_dir):
        class_path = os.path.join(data_dir, class_dir)
        if os.path.isdir(class_path):
            for filename in os.listdir(class_path):
                if filename.endswith('.dcm'):
                    filepath = os.path.join(class_path, filename)
                    ds = pydicom.dcmread(filepath)
                    image = ds.pixel_array
                    image = np.stack((image,) * 3, axis=-1)  # Convert to 3-channel image
                    image = tf.image.resize(image, (224, 224)).numpy()  # Resize to VGG input size
                    images.append(image)
                    labels.append(0 if class_dir == "no_cancer" else 1)
    return np.array(images), np.array(labels)

data_dir = 'path_to_the_datapipeline'
X, y = load_dicom_images(data_dir)

#SPLIT DATASET
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)


#ImageDataGenerator

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow(X_train, y_train, batch_size=32)
val_generator = val_test_datagen.flow(X_val, y_val, batch_size=32)
test_generator = val_test_datagen.flow(X_test, y_test, batch_size=32, shuffle=False)

#BUILDING THE MODEL
base_model = VGG19(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze the base model
for layer in base_model.layers:
    layer.trainable = False

# Add custom layers on top of VGG-19
x = base_model.output
x = Flatten()(x)
x = Dense(512, activation='relu')(x)
x = Dense(1, activation='sigmoid')(x)  # Binary classification

model = Model(inputs=base_model.input, outputs=x)

model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])


#TRAINING THE MODEL
history = model.fit(
    train_generator,
    steps_per_epoch=len(X_train) // 32,
    validation_data=val_generator,
    validation_steps=len(X_val) // 32,
    epochs=10
)

#EVALUATE THE MODEL

test_loss, test_accuracy = model.evaluate(test_generator, steps=len(X_test) // 32)
print(f'Test accuracy: {test_accuracy:.2f}')

#SAVING THE MODEL
model.save('path_to_the_model.h5')