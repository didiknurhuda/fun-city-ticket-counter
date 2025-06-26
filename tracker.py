import os
import cv2
import numpy as np
from ultralytics import YOLO
from sort import Sort  # Pastikan sort.py ada di folder yang sama

# Hindari error OpenMP
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Load model YOLO
model = YOLO("D:\\UNINDRA\\Penelitian\\Computer Vision\\Deteksi Tiket Fun City\\dataset\\runs\\detect\\train2\\weights\\best.pt")

# Buka webcam
cap = cv2.VideoCapture(1)

# Inisialisasi tracker
tracker = Sort(max_age=10, min_hits=3, iou_threshold=0.3)

# Set untuk ID unik
tracked_ids = set()
crops = set()

# Area polygon (ubah sesuai kebutuhan)
crossing_area = np.array([[(0, 155), (635, 155), (635, 215), (0, 215)]], dtype=np.int32)

# Simpan hitungan sebelumnya
prev_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Deteksi YOLO
    results = model.predict(frame, conf=0.3, verbose=False)

    detections = []
    if results and results[0].boxes is not None:
        for box in results[0].boxes.data.tolist():
            x1, y1, x2, y2, conf, cls = box
            detections.append([x1, y1, x2, y2, conf])

    detections_np = np.array(detections)

    # Update tracker
    if detections_np.shape[0] == 0:
        tracked_objects = np.empty((0, 5))
    else:
        tracked_objects = tracker.update(detections_np)

    # Gambar area hitung
    cv2.polylines(frame, crossing_area, isClosed=True, color=(255, 0, 0), thickness=2)
    cv2.putText(frame, "Area Hitung", (crossing_area[0][0][0] + 10, crossing_area[0][0][1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    for obj in tracked_objects:
        x1, y1, x2, y2, obj_id = obj
        x1, y1, x2, y2, obj_id = int(x1), int(y1), int(x2), int(y2), int(obj_id)

        tracked_ids.add(obj_id)

        # Cek apakah objek masuk area
        cx, cy = x1 + (x2 - x1) // 2, y1 + (y2 - y1) // 2
        if cv2.pointPolygonTest(crossing_area[0], (cx, cy), False) > 0:
            crops.add(obj_id)

        # Gambar bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"Tiket ID: {obj_id}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Hitung total tiket dalam area
    count = len(crops)
    cv2.putText(frame, f"Total Tiket: {count}", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
    
    # Jika ada perubahan, simpan ke file dan cetak
    if count != prev_count:
        print(f"Total tiket saat ini: {count}")
        prev_count = count
        with open("D:\\UNINDRA\\Penelitian\\Computer Vision\\Deteksi Tiket Fun City\\dataset\\jumlah_tiket.txt", "w") as f:
            f.write(f"Total tiket terdeteksi: {count}\n")

    # Tampilkan frame
    cv2.imshow("Tracking Tiket", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Bersihkan
cap.release()
cv2.destroyAllWindows()

# Cetak total akhir
print(f"\nTotal akhir tiket terdeteksi: {len(crops)}")