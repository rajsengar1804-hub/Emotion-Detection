import cv2
from predict import predict_all_probabilities

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Press 'q' to quit. Press 'p' to print full probability breakdown to console.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to grab frame.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))

    key = cv2.waitKey(1) & 0xFF

    for (x, y, w, h) in faces:
        pad_w = int(0.1 * w)
        pad_h = int(0.1 * h)
        y1 = max(0, y - pad_h)
        y2 = min(gray.shape[0], y + h + pad_h)
        x1 = max(0, x - pad_w)
        x2 = min(gray.shape[1], x + w + pad_w)

        face_crop = gray[y1:y2, x1:x2]
        face_crop = cv2.equalizeHist(face_crop)

        probs = predict_all_probabilities(face_crop)
        top_emotion, top_conf = next(iter(probs.items()))

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(
            frame, f"{top_emotion} ({top_conf:.0%})", (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
        )

        if key == ord('p'):
            print("\n--- Full probability breakdown ---")
            for emotion, prob in probs.items():
                print(f"{emotion:10s}: {prob:.2%}")

    cv2.imshow("Emotion Detection - Press 'q' to quit, 'p' for full breakdown", frame)

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()