import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import os

# Load model
model = load_model("transfer_model_aksara_1.h5")

# Load class names dari file .npy (urutan harus sama dengan saat training)
class_names = np.load("class_names.npy")

print("Class names:", class_names)

# Fungsi prediksi gambar
def predict_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))  # MobileNetV2 input size
    x = image.img_to_array(img)
    x = preprocess_input(x)  # gunakan preprocessing resmi MobileNetV2
    x = np.expand_dims(x, axis=0)

    predictions = model.predict(x, verbose=0)
    predicted_class = class_names[np.argmax(predictions)]
    confidence = np.max(predictions)

    print(f"Gambar: {os.path.basename(img_path)}")
    print(f"Prediksi: {predicted_class} ({confidence * 100:.2f}%)")

# Contoh uji satu gambar (ganti path sesuai lokasi gambar kamu)
image_path = r"raa.png"
predict_image(image_path)
