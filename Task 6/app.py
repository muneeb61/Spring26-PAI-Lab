from flask import Flask, render_template, request
import os
from profile_logic import analyze_face

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ["jpg", "jpeg", "png"]

@app.route("/", methods=["GET", "POST"])
def index():
    personality = "Waiting for image..."
    image = None   # 🔥 THIS IS IMPORTANT

    if request.method == "POST":
        file = request.files.get("image")

        if file and allowed_file(file.filename):
            image = file.filename
            image_path = os.path.join(app.config["UPLOAD_FOLDER"], image)
            file.save(image_path)

            personality = analyze_face(image_path)

    return render_template(
        "index.html",
        personality=personality,
        image=image   # 🔥 filename pass ho raha hai
    )

if __name__ == "__main__":
    app.run(debug=True)