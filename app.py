import os
import cv2  # type: ignore
import numpy as np  # type: ignore
import tensorflow as tf  # type: ignore
from flask import Flask, render_template, Response, request, url_for, redirect, jsonify # type: ignore
from PIL import Image  # type: ignore
from werkzeug.utils import secure_filename  # type: ignore
from tensorflow.keras.applications.efficientnet import preprocess_input  # type: ignore

# ================================
# CONFIG
# ================================
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# ================================
# LOAD MODEL
# ================================
model = tf.keras.models.load_model("hybrid_final.h5")

# ================================
# CLASS NAMES
# ================================
class_names = [
    "Iris_yellow_virus",
    "Stemphylium_leaf_blight",
    "healthy",
    "purple_blotch"
]

descriptions = {
    "Iris_yellow_virus": (
        "Penyakit virus yang menyebabkan daun menguning, "
        "pertumbuhan terhambat, dan hasil panen menurun."
    ),

    "Stemphylium_leaf_blight": (
        "Penyakit jamur yang menyebabkan bercak coklat keabu-abuan "
        "pada daun dan dapat menyebabkan daun mengering."
    ),

    "healthy": (
        "Daun bawang dalam kondisi sehat tanpa gejala penyakit."
    ),

    "purple_blotch": (
        "Penyakit jamur yang ditandai bercak ungu kehitaman "
        "dengan tepi kekuningan pada daun."
    )
}

# ================================
# REALTIME WEBCAM
# ================================
def gen_frames():
    cap = cv2.VideoCapture(0)

    while True:
        success, frame = cap.read()
        if not success:
            break

        img = cv2.resize(frame, (224, 224))
        img = preprocess_input(img.astype(np.float32))
        img = np.expand_dims(img, axis=0)

        preds = model.predict(img, verbose=0)[0]
        class_id = np.argmax(preds)
        confidence = float(preds[class_id]) * 100

        label = class_names[class_id] if confidence > 50 else "Unknown"
        color = (0, 255, 0) if confidence > 50 else (0, 0, 255)

        cv2.putText(
            frame,
            f"{label} ({confidence:.2f}%)",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            color,
            2
        )

        _, buffer = cv2.imencode(".jpg", frame)
        frame = buffer.tobytes()

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
        )

# ================================
# ROUTES
# ================================
@app.route("/")
def index():
    return render_template(
        "index.html",
        image_path=None,
        prediction=None,
        confidence=None,
        description=None
    )

@app.route("/video_feed")
def video_feed():
    return Response(
        gen_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )

# ================================
# IMAGE UPLOAD & PREDICT
# ================================
@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    # === SAVE IMAGE ===
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)

    # === PREPROCESS ===
    img = Image.open(filepath).convert("RGB")
    img = img.resize((224, 224))
    img_array = np.array(img).astype(np.float32)
    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)

    # === PREDICT ===
    preds = model.predict(img_array, verbose=0)[0]
    class_id = np.argmax(preds)
    confidence = round(float(preds[class_id]) * 100, 2)

    if confidence > 40:
        label = class_names[class_id]
        desc = descriptions[label]
    else:
        label = "Tidak Terdeteksi"
        desc = "Model kurang yakin terhadap kondisi daun tomat."

    image_url = url_for("static", filename=f"uploads/{filename}")

    # === AJAX RESPONSE (TANPA RELOAD) ===
    if request.headers.get("X-Requested-With") == "XMLHttpRequest":
        return jsonify({
            "prediction": label.replace("_", " "),
            "confidence": confidence,
            "description": desc,
            "image_path": image_url
        })

    # === FALLBACK (JIKA TANPA JS) ===
    return render_template(
        "index.html",
        prediction=label.replace("_", " "),
        confidence=confidence,
        description=desc,
        image_path=image_url
    )

# ================================
# DELETE IMAGE (AJAX)
# ================================
@app.route("/delete", methods=["POST"])
def delete_image():
    image_path = request.form.get("image_path")

    if image_path:
        file_path = image_path.replace("/static/", "static/")
        if os.path.exists(file_path):
            os.remove(file_path)

    return jsonify({"status": "deleted"})

# ================================
# RUN APP
# ================================
if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8080, debug=True)
