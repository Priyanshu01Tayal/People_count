from flask import Flask, render_template, request, redirect, Response
import cv2
import os
from ultralytics import YOLO
import smtplib
from email.mime.text import MIMEText

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = "uploads"

# YOLO model
model = YOLO("yolov8n.pt")

current_count = 0
video_path = None
last_alert_count = 0   # to track last alert



# ---------------- EMAIL SETTINGS ------------------

ADMIN_EMAIL =  "manojtayal07@gmail.com"         # WHERE ALERT IS SENT
SENDER_EMAIL =   "priyanshutayal35@gmail.com" # SENDER EMAIL ADDRESS
SENDER_PASSWORD = "grwo yjxm dwwa rinn"  # APP PASSWORD (NOT NORMAL PASSWORD)

def send_alert_email(new_count):
    subject = "People Gathering Alert!"
    body = f"Alert! People gathering increased.\nCurrent Count: {new_count}"

    msg = MIMEText(body)
    msg['Subject'] = subject
    msg['From'] = SENDER_EMAIL
    msg['To'] = ADMIN_EMAIL

    try:
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(SENDER_EMAIL, SENDER_PASSWORD)
        server.sendmail(SENDER_EMAIL, ADMIN_EMAIL, msg.as_string())
        server.quit()
        print("Alert email sent!")
    except Exception as e:
        print("Error sending email:", e)




# ---------------- ROUTES ------------------

@app.route('/')
def upload_page():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_video():
    global video_path

    if "video" not in request.files:
        return "No file uploaded"

    file = request.files["video"]
    if file.filename == "":
        return "No file selected"

    save_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(save_path)

    video_path = save_path
    return redirect("/player")


@app.route('/player')
def player_page():
    return render_template("player.html")




# ---------------- VIDEO PROCESSING ------------------

def generate_frames():
    global current_count, video_path, last_alert_count

    cap = cv2.VideoCapture(video_path)

    while True:
        success, frame = cap.read()
        if not success:
            break

        results = model(frame, stream=True)

        person_count = 0

        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                if cls == 0:
                    person_count += 1

                cv2.rectangle(
                    frame,
                    (int(box.xyxy[0][0]), int(box.xyxy[0][1])),
                    (int(box.xyxy[0][2]), int(box.xyxy[0][3])),
                    (0, 255, 0), 2
                )

        current_count = person_count

        # -------------- ALERT CONDITION -----------------
        if current_count - last_alert_count >= 5:
            send_alert_email(current_count)
            last_alert_count = current_count
        # -------------------------------------------------

        _, buffer = cv2.imencode(".jpg", frame)
        frame = buffer.tobytes()

        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")




@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route('/count')
def get_count():
    return str(current_count)



# ---------------- MAIN ------------------

if __name__ == "__main__":
    app.run(debug=True)
