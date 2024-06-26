import os
import numpy as np
import pydicom
import tensorflow as tf
from tensorflow.keras.applications import VGG19
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import pandas as pd

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

#LOAD AND PREPROCESS DATA
def load_dicom_images_and_labels(image_dir, metadata_file):
    metadata = pd.read_csv(metadata_file)
    images = []
    labels = []

    for root, _, files in os.walk(image_dir):
        for filename in files:
            if filename.endswith('.dcm'):
                filepath = os.path.join(root, filename)
                ds = pydicom.dcmread(filepath)
                image = ds.pixel_array
                image = np.stack((image,) * 3, axis=-1)  # Convert to 3-channel image
                image = tf.image.resize(image, (224, 224)).numpy()  # Resize to VGG input size

                # Extract patient ID from filepath and find the label
                patient_id = filepath.split(os.sep)[-3]  # Assuming patient ID is in this position in the path
                label_row = metadata[metadata['patient_id'] == patient_id]
                if not label_row.empty:
                    label = 0 if label_row['pathology'].values[0] == "Benign" else 1
                    images.append(image)
                    labels.append(label)
                else:
                    print(f"No metadata found for patient ID: {patient_id}")

    return np.array(images), np.array(labels)

data_dir = 'E:\JOSIAH CANCER DATASET_DONT F_TOUCH IT\manifest-ZkhPvrLo5216730872708713142\CBIS-DDSM'
metadata_file = 'E:\JOSIAH CANCER DATASET_DONT F_TOUCH IT\manifest-ZkhPvrLo5216730872708713142\metadata'
X, y = load_dicom_images_and_labels(data_dir, metadata_file)

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
x = BatchNormalization()(x)
x = Dropout(0.5)(x)
x = Dense(256, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)
x = Dense(128, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)

# This model will be used to extract features
feature_extractor_model = Model(inputs=base_model.input, outputs=x)

# Compile the model to make it ready for feature extraction
feature_extractor_model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

#extract feature
def extract_features(model, data):
    features = model.predict(data)
    return features

# Normalize images for VGG-19
X_train_scaled = X_train / 255.0
X_val_scaled = X_val / 255.0
X_test_scaled = X_test / 255.0

# Extract features
train_features = extract_features(feature_extractor_model, X_train_scaled)
val_features = extract_features(feature_extractor_model, X_val_scaled)
test_features = extract_features(feature_extractor_model, X_test_scaled)

#standardize feature
scaler = StandardScaler()
train_features = scaler.fit_transform(train_features)
val_features = scaler.transform(val_features)
test_features = scaler.transform(test_features)

#Train Xgradient model
xgb_model = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
xgb_model.fit(train_features, y_train, eval_set=[(val_features, y_val)], early_stopping_rounds=10, eval_metric="logloss", verbose=True)


#Evaluate model
y_pred = xgb_model.predict(test_features)
test_accuracy = accuracy_score(y_test, y_pred)
print(f'Test accuracy: {test_accuracy:.2f}')

#saving model
feature_extractor_model.save('E:\Model_output')
xgb_model.save_model('E:\Model_output')
