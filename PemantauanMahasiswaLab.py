import cv2
import pandas as pd
import numpy as np
import os
from datetime import datetime
from minio import Minio
from deepface import DeepFace
from ultralytics import YOLO
import torch



# Inisialisasi model YOLO dan database wajah
model = YOLO("yukbisa.pt").to('cuda')
face_db_path = "Dataset-wajah"
video_path = 0
cap = cv2.VideoCapture(video_path)

# Konstanta class YOLO
class_dosen, class_bangku, class_mahasiswa = 2, 0, 1
tracked_data = {}
frame_id = 0
fps = cap.get(cv2.CAP_PROP_FPS) or 30

def is_overlap(boxA, boxB):
    xA, yA = max(boxA[0], boxB[0]), max(boxA[1], boxB[1])
    xB, yB = min(boxA[2], boxB[2]), min(boxA[3], boxB[3])
    return max(0, xB - xA) * max(0, yB - yA) > 0

def recognize_face(face_img):
    temp_file = "temp_face.jpg"
    cv2.imwrite(temp_file, face_img)
    try:
        result = DeepFace.find(img_path=temp_file, db_path=face_db_path,
                               detector_backend='retinaface', model_name='ArcFace', enforce_detection=False, silent=True)
        if result and len(result) > 0 and not result[0].empty:
            identity = os.path.basename(result[0].iloc[0]['identity']).split('.')[0]
            return identity
    except Exception as e:
        print(f"Face Recognition Error: {e}")
    finally:
        if os.path.exists(temp_file):
            os.remove(temp_file)
    return "Unknown"

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Resize frame agar lebih ringan
    frame = cv2.resize(frame, (1280, 720))
    results = model.track(frame, persist=True, verbose=False)
    bangku_boxes = []

    if results and results[0].boxes and results[0].boxes.id is not None:
        for box in results[0].boxes:
            if int(box.cls) == class_bangku:
                bangku_boxes.append([int(i) for i in box.xyxy[0]])

        for box, obj_id in zip(results[0].boxes, results[0].boxes.id):
            obj_id, cls = int(obj_id), int(box.cls)
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            if cls == class_dosen:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(frame, "Dosen", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                continue
            if cls != class_mahasiswa:
                continue

            status = "Tidak Praktik"
            mahasiswa_box = [x1, y1, x2, y2]
            for bangku in bangku_boxes:
                if is_overlap(mahasiswa_box, bangku):
                    status = "Praktik"
                    break

            # Face Recognition (langsung pakai frame yang sudah kecil)
            face_img = frame[y1:y2, x1:x2]
            identity = recognize_face(face_img)

            color = (0, 255, 0) if status == "Praktik" else (0, 255, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{identity} - {status}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # Durasi Tracking
            if obj_id not in tracked_data:
                tracked_data[obj_id] = {"name": identity, "start_time": frame_id / fps, "last_status": status,
                                        "Praktik": 0.0, "Tidak Praktik": 0.0}
            else:
                if status != tracked_data[obj_id]["last_status"]:
                    duration = (frame_id / fps) - tracked_data[obj_id]["start_time"]
                    tracked_data[obj_id][tracked_data[obj_id]["last_status"]] += duration
                    tracked_data[obj_id]["start_time"] = frame_id / fps
                    tracked_data[obj_id]["last_status"] = status

    print(torch.cuda.is_available())
    cv2.imshow("Monitoring Mahasiswa", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    frame_id += 1

cap.release()
cv2.destroyAllWindows()

# Akumulasi waktu terakhir
for obj_id, data in tracked_data.items():
    duration = (frame_id / fps) - data["start_time"]
    tracked_data[obj_id][data["last_status"]] += duration

# Simpan ke CSV
log_data = [{"ID": obj_id, "Nama": data["name"],
             "Praktik (detik)": round(data["Praktik"], 2),
             "Tidak Praktik (detik)": round(data["Tidak Praktik"], 2)}
            for obj_id, data in tracked_data.items()]

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

try:
    minio_client.fput_object(bucket_name, output_csv, output_csv, content_type="application/csv")
    print(f"[✅] CSV berhasil di-upload ke MinIO Bucket '{bucket_name}' sebagai '{output_csv}'")
    os.remove(output_csv)
    print(f"[✅] File lokal {output_csv} dihapus setelah upload.")
except Exception as e:
    print(f"[❌] Gagal upload ke MinIO: {e}")
