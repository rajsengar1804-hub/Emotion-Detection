import cv2
from predict import predict_emotion_from_array

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


def detect_faces(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
    )

    return img, gray, faces


if __name__ == "__main__":
    test_image_path = "test_face.png"
    img, gray, faces = detect_faces(test_image_path)

    print(f"Faces detected: {len(faces)}")

    for (x, y, w, h) in faces:
        pad_w = int(0.1 * w)
        pad_h = int(0.1 * h)
        y1 = max(0, y - pad_h)
        y2 = min(gray.shape[0], y + h + pad_h)
        x1 = max(0, x - pad_w)
        x2 = min(gray.shape[1], x + w + pad_w)

        face_crop = gray[y1:y2, x1:x2]
        face_crop = cv2.equalizeHist(face_crop)

        emotion, confidence = predict_emotion_from_array(face_crop)
        print(f"Face at x={x}, y={y}, w={w}, h={h} -> {emotion} ({confidence:.2%})")

        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(
            img, f"{emotion} ({confidence:.0%})", (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
        )

    cv2.imwrite("face_detected_output.png", img)
    print("Saved annotated image as face_detected_output.png")