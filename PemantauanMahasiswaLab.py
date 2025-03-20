from ultralytics import YOLO
import cv2
import pandas as pd
from datetime import datetime
from minio import Minio
from deepface import DeepFace
import os
import numpy as np

model = YOLO("model_mahasiswa_bangku.pt")
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

video_path = "20250320_125813.mp4"
cap = cv2.VideoCapture(video_path)

tracked_data = {}
frame_id = 0
fps = cap.get(cv2.CAP_PROP_FPS) if cap.get(cv2.CAP_PROP_FPS) > 0 else 30

# Map kelas
class_mahasiswa, class_dosen, class_bangku = 0, 1, 2

# Load database wajah (folder foto)
face_db_path = "dataset_wajah"

def is_overlap(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    return interArea > 0

def recognize_face(face_img):
    try:
        result = DeepFace.find(img_path=face_img, db_path=face_db_path, enforce_detection=False, silent=True)
        if len(result) > 0 and not result[0].empty:
            identity = os.path.basename(result[0].iloc[0]['identity']).split('.')[0]
            return identity
    except:
        pass
    return "Unknown"

def detect_eye_state(face_img_gray):
    eyes = eye_cascade.detectMultiScale(face_img_gray, scaleFactor=1.1, minNeighbors=5)
    return "Open" if len(eyes) > 0 else "Closed"

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.rotate(frame, cv2.ROTATE_180)
    results = model.track(frame, persist=True)
    bangku_boxes = []

    if results and results[0].boxes.id is not None:
        # Simpan posisi bangku
        for box in results[0].boxes:
            if int(box.cls) == class_bangku:
                bangku_boxes.append([int(i) for i in box.xyxy[0]])

        for box, obj_id in zip(results[0].boxes, results[0].boxes.id):
            obj_id = int(obj_id)
            cls = int(box.cls)
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            mahasiswa_box = [x1, y1, x2, y2]

            if cls == class_dosen:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(frame, "Dosen", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                continue

            # Deteksi Mahasiswa
            status = "Tidak Praktik"
            for bangku in bangku_boxes:
                if is_overlap(mahasiswa_box, bangku):
                    status = "Praktik"
                    break

            # Crop wajah
            face_img = frame[y1:y2, x1:x2]
            gray_face = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)

            # DeepFace Recognize
            identity = recognize_face(face_img)
            # Eye state
            eye_state = detect_eye_state(gray_face)

            # Tambahkan status tidur jika di bangku dan mata tertutup
            if status == "Praktik" and eye_state == "Closed":
                status = "Tidur"

            color = (0, 255, 0) if status == "Praktik" else (0, 255, 255)
            if status == "Tidur":
                color = (0, 0, 255)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{identity} - {status}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # Tracking data
            if obj_id not in tracked_data:
                tracked_data[obj_id] = {"name": identity, "start_time": frame_id / fps, "last_status": status,
                                        "Praktik": 0.0, "Tidak Praktik": 0.0, "Tidur": 0.0}
            else:
                if status != tracked_data[obj_id]["last_status"]:
                    duration = (frame_id / fps) - tracked_data[obj_id]["start_time"]
                    tracked_data[obj_id][tracked_data[obj_id]["last_status"]] += duration
                    tracked_data[obj_id]["start_time"] = frame_id / fps
                    tracked_data[obj_id]["last_status"] = status

    cv2.imshow("Monitoring Mahasiswa", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_id += 1

cap.release()
cv2.destroyAllWindows()

# Akumulasi terakhir
for obj_id, data in tracked_data.items():
    duration = (frame_id / fps) - data["start_time"]
    data[data["last_status"]] += duration

# Simpan CSV
log_data = [{
    "ID": obj_id,
    "Nama": data["name"],
    "Praktik (detik)": round(data["Praktik"], 2),
    "Tidak Praktik (detik)": round(data["Tidak Praktik"], 2),
    "Tidur (detik)": round(data["Tidur"], 2)
} for obj_id, data in tracked_data.items()]

df = pd.DataFrame(log_data)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_csv = f"activity_log_{timestamp}.csv"
df.to_csv(output_csv, index=False)
print(f"[✅] Log aktivitas disimpan ke: {output_csv}")

# Upload ke MinIO
minio_client = Minio("localhost:9000", access_key="Andra", secret_key="Andra123", secure=False)
bucket_name = "heehee"
if not minio_client.bucket_exists(bucket_name):
    minio_client.make_bucket(bucket_name)

minio_client.fput_object(bucket_name, output_csv, output_csv, content_type="application/csv")
print(f"[✅] CSV berhasil di-upload ke MinIO Bucket '{bucket_name}' sebagai '{output_csv}'")
os.remove(output_csv)
print(f"[✅] File lokal {output_csv} dihapus setelah upload.")
