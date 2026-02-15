import cv2

# Load cascades
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_eye.xml"
)
mouth_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_smile.xml"
)

# Read image
img = cv2.imread("test.jpg")

if img is None:
    print("❌ Image not found")
    exit()

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect face
faces = face_cascade.detectMultiScale(gray, 1.3, 5)

for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]

    # Detect eyes
    eyes = eye_cascade.detectMultiScale(roi_gray, 1.3, 5)
    for (ex, ey, ew, eh) in eyes:
        cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)

    # Detect mouth (lower half of face)
    mouth = mouth_cascade.detectMultiScale(
        roi_gray, 1.7, 20
    )
    for (mx, my, mw, mh) in mouth:
        if my > h / 2:  # lower face only
            cv2.rectangle(
                roi_color, (mx, my), (mx+mw, my+mh), (0, 0, 255), 2
            )
            break

# Show result
cv2.imshow("Facial Features", img)
cv2.waitKey(0)
cv2.destroyAllWindows()