import cv2

# Personality descriptions
personality_descriptions = {
    "ENFP": "Enthusiastic, creative, and sociable free spirits who can always find a reason to smile.",
    "ENFJ": "Charismatic and inspiring leaders, able to mesmerize their listeners.",
    "ENTJ": "Bold, imaginative, and strong-willed leaders, always finding a way – or making one.",
    "ENTP": "Smart and curious thinkers who cannot resist an intellectual challenge.",
    "ESFP": "Spontaneous, energetic, and enthusiastic people – life is never boring around them.",
    "ESFJ": "Extraordinarily caring, social, and popular people, always eager to help.",
    "ESTJ": "Excellent administrators, unsurpassed at managing things – or people.",
    "ESTP": "Smart, energetic, and perceptive people, who truly enjoy living on the edge.",
    "INFP": "Poetic, kind, and altruistic people, always eager to help a good cause.",
    "INFJ": "Creative, insightful, and principled, they have a strong sense of purpose.",
    "INTJ": "Imaginative and strategic thinkers, with a plan for everything.",
    "INTP": "Innovative inventors with an unquenchable thirst for knowledge.",
    "ISFP": "Flexible and charming artists, always ready to explore and experience something new.",
    "ISFJ": "Very dedicated and warm protectors, always ready to defend their loved ones.",
    "ISTJ": "Practical and fact-focused individuals, whose reliability cannot be doubted.",
    "ISTP": "Bold and practical experimenters, masters of all kinds of tools."
}

def analyze_face_from_frame(frame):
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    eye_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_eye.xml"
    )
    mouth_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_smile.xml"
    )

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)  # Adjusted for better detection

    personality = "No face detected"
    description = ""

    if len(faces) > 0:
        # Analyze ONLY first face
        (x, y, w, h) = faces[0]
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        personality = ""

        # E / I — Eyes
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 5)
        personality += "E" if len(eyes) >= 2 else "I"

        # Draw eyes
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)

        # S / N — Nose (approx via face width)
        personality += "S" if w > 120 else "N"  # Adjusted threshold

        # T / F — Mouth
        mouths = mouth_cascade.detectMultiScale(roi_gray, 1.7, 20)
        mouth_added = False
        for (mx, my, mw, mh) in mouths:
            if my > h / 2:
                personality += "T" if mw > w * 0.3 else "F"  # Adjusted threshold
                # Draw mouth
                cv2.rectangle(roi_color, (mx, my), (mx+mw, my+mh), (0, 0, 255), 2)
                mouth_added = True
                break
        if not mouth_added:
            personality += "F"

        # J / P — Jawline (face ratio)
        face_ratio = h / w
        personality += "J" if face_ratio < 1.3 else "P"  # Adjusted ratio

        # Draw rectangle around face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # Get description
        description = personality_descriptions.get(personality, "Unknown personality type")

    return frame, personality, description

# Initialize webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame")
        break

    # Analyze the frame
    analyzed_frame, personality, description = analyze_face_from_frame(frame)

    # Display personality and description on the frame
    cv2.putText(analyzed_frame, f"Personality: {personality}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    # Wrap description text
    y_offset = 60
    for line in description.split(', '):
        cv2.putText(analyzed_frame, line, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y_offset += 20

    # Show the frame
    cv2.imshow("Live Face Analysis", analyzed_frame)

    # Break on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()