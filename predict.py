from ultralytics import YOLO
import cv2

# Load model yang sudah dilatih
model = YOLO("korsi.pt")  # Sesuaikan path model

 #Tes pada satu gambar
#image_path = "apis.jpg"  # Ganti dengan path gambar di komputer
#results = model(image_path)

 #Tampilkan hasil
#for r in results:
    #r.show()

video_path = "progres.mp4"  # Ganti dengan path video di komputer
model.predict(video_path, save=True, conf=0.5)