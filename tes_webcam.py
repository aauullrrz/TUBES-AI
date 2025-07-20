import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import os

# Load model dan input size
model = load_model("transfer_model_aksara_1.h5")
_, h, w, _ = model.input_shape
input_size = (w, h)

# Load urutan nama kelas dari file class_names.npy
class_names = np.load("class_names.npy")
print("Class names loaded:", class_names)

# Inisialisasi webcam
cap = cv2.VideoCapture(0)


print("Tekan [SPACE] untuk ambil gambar dan prediksi.")
print("Tekan [Q] untuk keluar.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    
    cv2.putText(frame, "Tekan [SPACE] untuk prediksi", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.imshow("Kamera Aksara Jawa", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord(' '):
        snapshot = frame.copy()

        
        gray = cv2.cvtColor(snapshot, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 110, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        result = snapshot.copy()

        for cnt in contours:
            x, y, w_box, h_box = cv2.boundingRect(cnt)

            if w_box > 50 and h_box > 50:  
                roi = snapshot[y:y+h_box, x:x+w_box]
                roi_resized = cv2.resize(roi, input_size)
                img_array = img_to_array(roi_resized) / 255.0
                img_array = np.expand_dims(img_array, axis=0)

                predictions = model.predict(img_array, verbose=0) #CNN
                confidence = np.max(predictions)
                class_id = np.argmax(predictions)

                if confidence >= 0.5:
                    label = f"{class_names[class_id]} ({confidence*100:.1f}%)"
                else:
                    label = "Tidak dikenali (0%)"

                cv2.rectangle(result, (x, y), (x+w_box, y+h_box), (0, 255, 0), 2)
                cv2.putText(result, label, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        
                cv2.imshow("Hasil Prediksi", result)

        # Simpan hasil prediksi ke folder
        if not os.path.exists("hasil_prediksi"):
            os.makedirs("hasil_prediksi")

        filename = f"hasil_prediksi/prediksi_{cv2.getTickCount()}.png"
        cv2.imwrite(filename, result)
        print(f"Hasil prediksi disimpan di: {filename}")

        cv2.waitKey(5000)
        cv2.destroyWindow("Hasil Prediksi")


    elif key == ord('q'):
        break

       

cap.release()
cv2.destroyAllWindows()
