import cv2

def analyze_face(image_path):
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    eye_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_eye.xml"
    )
    mouth_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_smile.xml"
    )

    img = cv2.imread(image_path)
    if img is None:
        return "Image could not be read"

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        return "No face detected"

    # Analyze ONLY first face
    (x, y, w, h) = faces[0]
    roi_gray = gray[y:y+h, x:x+w]

    personality = ""

    # E / I — Eyes
    eyes = eye_cascade.detectMultiScale(roi_gray, 1.3, 5)
    personality += "E" if len(eyes) >= 2 else "I"

    # S / N — Nose (approx via face width)
    personality += "S" if w > 150 else "N"

    # T / F — Mouth
    mouths = mouth_cascade.detectMultiScale(roi_gray, 1.7, 20)
    mouth_added = False
    for (mx, my, mw, mh) in mouths:
        if my > h / 2:
            personality += "T" if mw > w * 0.4 else "F"
            mouth_added = True
            break
    if not mouth_added:
        personality += "F"

    # J / P — Jawline (face ratio)
    face_ratio = h / w
    personality += "J" if face_ratio < 1.25 else "P"

    return personality