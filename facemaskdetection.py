import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator # pyright: ignore[reportMissingImports]
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint # pyright: ignore[reportMissingImports]
from tensorflow.keras.models import load_model # pyright: ignore[reportMissingImports]
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from IPython.display import display, clear_output
import ipywidgets as widgets
import os
import threading
import io
import base64



# Set correct paths
train_dir = r"D:\my learning\python\face_mask detection-20250831T170337Z-1-001\face_mask detection\train"

val_dir = r"D:\my learning\python\face_mask detection-20250831T170337Z-1-001\face_mask detection\validation"

# Check if paths exist
print("Train folder exists:", os.path.exists(train_dir))
print("Validation folder exists:", os.path.exists(val_dir))

# ImageDataGenerator with augmentation
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

val_datagen = ImageDataGenerator(rescale=1./255)

# Load data
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary',
    classes=['without_mask', 'with_mask']  # Ensure folder names match
)

validation_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary',
    classes=['without_mask', 'with_mask']
)


#  Define the CNN model
model = tf.keras.Sequential([
     tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
     tf.keras.layers.MaxPooling2D(2, 2),
     tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
     tf.keras.layers.MaxPooling2D(2, 2),

     tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
     tf.keras.layers.MaxPooling2D(2, 2),

     tf.keras.layers.Conv2D(256, (3, 3), activation='relu'),
     tf.keras.layers.MaxPooling2D(2, 2),

     tf.keras.layers.Flatten(),
     tf.keras.layers.Dense(512, activation='relu'),
     tf.keras.layers.Dropout(0.5),
     tf.keras.layers.Dense(1, activation='sigmoid')
 ])

# Compile the model
model.compile(
     optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
     loss='binary_crossentropy',
     metrics=['accuracy']
 )

# Callbacks
callbacks = [
    EarlyStopping(monitor='val_loss', patience=3, verbose=1),
    ModelCheckpoint('best_mask_model.keras', monitor='val_accuracy', save_best_only=True, verbose=1)

]

# Train the model
history = model.fit(
     train_generator,
     epochs=5,
     validation_data=validation_generator,
     callbacks=callbacks
 )

plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title("Model Accuracy Over Epochs")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.show()

 # Evaluate the model
test_loss, test_acc = model.evaluate(validation_generator)
print(f'\nTest accuracy: {test_acc:.4f}')

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# # Load the best saved model
# model = load_model('best_mask_model.h5')  # or .keras if you switched
# face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def process_frame(frame):
    # Convert to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    if len(faces) == 0:
        cv2.putText(frame, "No face detected", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
    else:
        for (x, y, w, h) in faces:
            # Extract face ROI
            face_roi = frame[y:y+h, x:x+w]

            # Preprocess for model
            face_img = cv2.resize(face_roi, (150, 150))
            face_img = face_img / 255.0
            face_img = np.expand_dims(face_img, axis=0)

            # Make prediction
            prediction = model.predict(face_img, verbose=0)[0][0]

            # Draw rectangle and label
            # color = (0, 255, 0)  
            if prediction < 0.4:
             color = (0, 255, 0)
             label = "Mask"
            elif prediction > 0.6:
             color = (0, 0, 255)
             label = "No Mask"
            else:
             color = (0, 255, 255)
             label = "Uncertain"
            confidence = f"{max(prediction, 1-prediction)*100:.1f}%"

            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, f"{label} {confidence}", (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    return frame




# Start webcam for live detection
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = process_frame(frame)  # 👈 Handles detection and annotation
    cv2.imshow("Face Mask Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

