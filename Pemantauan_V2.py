from ultralytics import YOLO
import cv2
import pandas as pd
from datetime import datetime

# Load model YOLO
model = YOLO("terbaru.pt")  # Ganti sesuai path modelmu

# Video source
video_path = "video_testing.mp4"
cap = cv2.VideoCapture(video_path)

# Tracking data untuk mahasiswa saja
tracked_data = {}
frame_id = 0
fps = cap.get(cv2.CAP_PROP_FPS) if cap.get(cv2.CAP_PROP_FPS) > 0 else 30

# Mapping class ke aktivitas (pastikan class ke-0 adalah dosen)
activity_mapping = {0: "Dosen", 1: "Praktik", 2: "Tidak_praktik", 3: "Tidur"}

# Fungsi mendeteksi aktivitas
def get_activity(box):
    cls, conf = int(box.cls), float(box.conf)
    return activity_mapping.get(cls, "Tidak_praktik") if conf > 0.5 else "Tidak_praktik"

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640, 480))
    frame = cv2.rotate(frame, cv2.ROTATE_180)

    results = model.track(frame, persist=True)

    if results and results[0].boxes.id is not None:
        for box, obj_id in zip(results[0].boxes, results[0].boxes.id):
            obj_id = int(obj_id)
            cls = int(box.cls)
            activity = get_activity(box)
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Warna untuk setiap aktivitas
            if activity == "Dosen":
                color = (255, 0, 0)  # Biru untuk dosen
            elif activity == "Praktik":
                color = (0, 255, 0)
            elif activity == "Tidak_praktik":
                color = (0, 255, 255)
            else:
                color = (0, 0, 255)

            # Gambar bounding box dan label
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"ID {obj_id}: {activity}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # Skip pencatatan jika objek adalah Dosen (class 0)
            if cls == 0:
                continue

            # Tracking aktivitas per Mahasiswa
            if obj_id not in tracked_data:
                tracked_data[obj_id] = {
                    "start_time": frame_id / fps,
                    "last_activity": activity,
                    "Praktik": 0.0,
                    "Tidak_praktik": 0.0,
                    "Tidur": 0.0
                }
            else:
                if activity != tracked_data[obj_id]["last_activity"]:
                    duration = (frame_id / fps) - tracked_data[obj_id]["start_time"]
                    tracked_data[obj_id][tracked_data[obj_id]["last_activity"]] += duration
                    tracked_data[obj_id]["start_time"] = frame_id / fps
                    tracked_data[obj_id]["last_activity"] = activity

    cv2.imshow("Pemantauan Aktivitas", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_id += 1

cap.release()
cv2.destroyAllWindows()

# Akumulasi durasi terakhir setelah video selesai
for obj_id, data in tracked_data.items():
    duration = (frame_id / fps) - data["start_time"]
    data[data["last_activity"]] += duration

# Simpan hasil ke CSV dengan timestamp unik (hanya mahasiswa)
log_data = [{
    "ID": obj_id,
    "Praktik (detik)": round(data["Praktik"], 2),
    "Tidak_praktik (detik)": round(data["Tidak_praktik"], 2),
    "Tidur (detik)": round(data["Tidur"], 2)
} for obj_id, data in tracked_data.items()]

df = pd.DataFrame(log_data)

# Buat nama file dengan waktu
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_csv = f"activity_log_{timestamp}.csv"
df.to_csv(output_csv, index=False)

print(f"[✅] Log aktivitas mahasiswa disimpan ke: {output_csv}")
