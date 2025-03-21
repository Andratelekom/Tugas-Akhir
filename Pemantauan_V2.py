from ultralytics import YOLO
import cv2
import pandas as pd
from datetime import datetime
from minio import Minio
import os

model = YOLO("barulagi.pt")
video_path = "20250320_125813.mp4"
cap = cv2.VideoCapture(video_path)

tracked_data = {}
frame_id = 0
fps = cap.get(cv2.CAP_PROP_FPS) if cap.get(cv2.CAP_PROP_FPS) > 0 else 30

# Map kelas sesuai permintaan
class_dosen, class_bangku, class_mahasiswa = 0, 1, 2

def is_overlap(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    return interArea > 0

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
            obj_box = [x1, y1, x2, y2]

            if cls == class_dosen:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(frame, "Dosen", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                continue

            if cls == class_mahasiswa:
                # Default status tidak praktik
                status = "Tidak Praktik"
                # Cek apakah mahasiswa overlap dengan bangku
                for bangku in bangku_boxes:
                    if is_overlap(obj_box, bangku):
                        status = "Praktik"
                        break

                # Warna bounding box
                color = (0, 255, 0) if status == "Praktik" else (0, 255, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"Mahasiswa - {status}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                # Tracking waktu per mahasiswa
                if obj_id not in tracked_data:
                    tracked_data[obj_id] = {"start_time": frame_id / fps, "last_status": status,
                                            "Praktik": 0.0, "Tidak Praktik": 0.0}
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

# Simpan ke CSV
log_data = [{
    "ID": obj_id,
    "Praktik (detik)": round(data["Praktik"], 2),
    "Tidak Praktik (detik)": round(data["Tidak Praktik"], 2)
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
