from ultralytics import YOLO
import cv2
import time
import pandas as pd

# Load model yang sudah dilatih
model = YOLO("terbaru.pt")  # Sesuaikan path model

# Inisialisasi variabel
activity_log = {}  # Untuk menyimpan log aktivitas (tanpa duplikat ID)
tracked_students = {}  # Untuk melacak mahasiswa dan aktivitasnya

# Fungsi untuk mendeteksi aktivitas
def detect_activity(box):
    cls = int(box.cls)  # Class ID
    conf = float(box.conf)  # Confidence score

    # Class 1: Praktik, Class 2: Tidak_praktik, Class 3: Tidur
    if cls == 1 and conf > 0.5:  # Jika sedang praktik
        return "Praktik"
    elif cls == 2 and conf > 0.5:  # Jika tidak praktik
        return "Tidak_praktik"
    elif cls == 3 and conf > 0.5:  # Jika sedang tidur
        return "Tidur"
    else:
        return "Tidak_praktik"  # Default: Tidak praktik

# Buka video
video_path = "video_testing.mp4"  # Ganti dengan path video di komputer
cap = cv2.VideoCapture(video_path)

# Inisialisasi variabel untuk video processing
frame_id = 0
fps = cap.get(cv2.CAP_PROP_FPS)  # Frame rate video

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Resize frame untuk mengurangi penggunaan memori (opsional)
    frame = cv2.resize(frame, (640, 480))
    frame = cv2.rotate(frame, cv2.ROTATE_180)

    # Jalankan inferensi YOLO pada frame
    results = model.track(frame, persist=True)  # Gunakan tracking untuk melacak ID

    # Deteksi aktivitas dan catat untuk setiap mahasiswa
    if results[0].boxes.id is not None:  # Jika ada ID yang terdeteksi
        for box, student_id in zip(results[0].boxes, results[0].boxes.id):
            student_id = int(student_id)  # ID mahasiswa
            activity = detect_activity(box)  # Aktivitas mahasiswa

            # Jika mahasiswa baru, tambahkan ke tracked_students
            if student_id not in tracked_students:
                tracked_students[student_id] = {
                    "start_time": frame_id / fps,
                    "last_activity": activity,
                    "Praktik": 0.0,  # Durasi aktivitas Praktik
                    "Tidak_praktik": 0.0,  # Durasi aktivitas Tidak_praktik
                    "Tidur": 0.0,  # Durasi aktivitas Tidur
                }
            else:
                # Jika aktivitas berubah, akumulasi durasi aktivitas sebelumnya
                if activity != tracked_students[student_id]["last_activity"]:
                    duration = (frame_id / fps) - tracked_students[student_id]["start_time"]
                    # Akumulasi durasi ke aktivitas sebelumnya
                    tracked_students[student_id][tracked_students[student_id]["last_activity"]] += duration
                    # Perbarui aktivitas dan waktu mulai
                    tracked_students[student_id]["start_time"] = frame_id / fps
                    tracked_students[student_id]["last_activity"] = activity

    # Tampilkan aktivitas di frame (opsional)
    for student_id, info in tracked_students.items():
        cv2.putText(frame, f"ID {student_id}: {info['last_activity']}", (10, 30 + 30 * student_id), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("Aktivitas Mahasiswa Lab", frame)

    # Keluar dengan menekan tombol 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_id += 1

# Tutup video dan jendela
cap.release()
cv2.destroyAllWindows()

# Catat aktivitas terakhir untuk setiap mahasiswa
for student_id, info in tracked_students.items():
    duration = (frame_id / fps) - info["start_time"]
    # Akumulasi durasi terakhir ke aktivitas sebelumnya
    info[info["last_activity"]] += duration
    activity_log[student_id] = {
        "ID": student_id,
        "Praktik": info["Praktik"],
        "Tidak_praktik": info["Tidak_praktik"],
        "Tidur": info["Tidur"],
    }

# Simpan log aktivitas ke DataFrame
df = pd.DataFrame(activity_log.values())

# Simpan ke file CSV
output_csv_path = "activity_log.csv"  # Sesuaikan path output
df.to_csv(output_csv_path, index=False)
print(f"Log aktivitas disimpan ke: {output_csv_path}")
print(f"Activity Log: {activity_log}")