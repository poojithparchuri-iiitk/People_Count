import cv2
from ultralytics import YOLO

# Load YOLOv8 pretrained model
model = YOLO("yolov8n.pt")

# Open external webcam (index = 1)
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print(" Camera not accessible")
    exit()

# Full screen window
window_name = "Human Detection with Count"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.setWindowProperty(
    window_name,
    cv2.WND_PROP_FULLSCREEN,
    cv2.WINDOW_FULLSCREEN
)

print(" Human detection + counting started. Press Q to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    person_count = 0 
    # Run YOLO detection
    results = model(frame, conf=0.4, classes=[0], verbose=False)

    # Draw bounding boxes
    for box in results[0].boxes:
        cls = int(box.cls[0])
        label = model.names[cls]

        if label == "person":
            person_count += 1  

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0]

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"Person {conf:.2f}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

    # DISPLAY PERSON COUNT
    cv2.putText(
        frame,
        f"Person Count: {person_count}",
        (30, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (0, 0, 255),
        3
    )

    cv2.imshow(window_name, frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()