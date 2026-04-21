from flask import Flask, render_template, Response, request
import cv2
import face_recognition
import numpy as np
import pandas as pd
from datetime import datetime
import sqlite3
import os
import time

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "attendance.db")
DATASET_PATH = "dataset"

os.makedirs(DATASET_PATH, exist_ok=True)

# -------------------------
# PERIOD TIMINGS
# -------------------------
PERIODS = {
    1: ("09:10:00","10:00:00"),
    2: ("10:00:00","10:50:00"),
    3: ("10:50:00","11:40:00"),
    4: ("11:40:00","12:30:00"),
    5: ("13:30:00","14:20:00"),
    6: ("14:20:00","15:10:00"),
    7: ("15:10:00","16:00:00")
}

# -------------------------
# DATABASE
# -------------------------
def create_table():

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS attendance(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT,
        date TEXT,
        period INTEGER,
        in_time TEXT,
        out_time TEXT,
        duration TEXT,
        UNIQUE(name,date,period)
    )
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS movement(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT,
        date TEXT,
        time TEXT,
        event TEXT
    )
    """)

    conn.commit()
    conn.close()

create_table()

# -------------------------
# CURRENT PERIOD
# -------------------------
def get_current_period():

    now = datetime.now().strftime("%H:%M:%S")

    for period,(start,end) in PERIODS.items():
        if start <= now <= end:
            return period

    return None

# -------------------------
# LOAD DATASET
# -------------------------
def load_dataset():

    images = []
    classNames = []

    for file in os.listdir(DATASET_PATH):

        if not file.lower().endswith(('.png','.jpg','.jpeg')):
            continue

        img_path = os.path.join(DATASET_PATH, file)
        img = cv2.imread(img_path)

        if img is None:
            continue

        images.append(img)
        classNames.append(os.path.splitext(file)[0])

    return images, classNames

# -------------------------
# FACE ENCODINGS
# -------------------------
def findEncodings(images):

    encodeList = []

    for img in images:

        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        encodes = face_recognition.face_encodings(img)

        if len(encodes) > 0:
            encodeList.append(encodes[0])

    return encodeList


images, classNames = load_dataset()
encodeListKnown = findEncodings(images)

# -------------------------
# TRACK LAST SEEN
# -------------------------
last_seen = {}
movement_log = {}

# -------------------------
# SAVE MOVEMENT
# -------------------------
def save_movement(name):

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    now = datetime.now()
    date = now.strftime("%Y-%m-%d")
    time_now = now.strftime("%H:%M:%S")

    cursor.execute("""
    INSERT INTO movement(name,date,time,event)
    VALUES(?,?,?,?)
    """,(name,date,time_now,"Movement"))

    conn.commit()
    conn.close()

# -------------------------
# MARK IN
# -------------------------
def mark_in(name):

    period = get_current_period()
    if period is None:
        return

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    now = datetime.now()
    date = now.strftime("%Y-%m-%d")
    time_now = now.strftime("%H:%M:%S")

    cursor.execute("""
    SELECT * FROM attendance
    WHERE name=? AND date=? AND period=?
    """,(name,date,period))

    record = cursor.fetchone()

    if record is None:

        cursor.execute("""
        INSERT INTO attendance(name,date,period,in_time)
        VALUES(?,?,?,?)
        """,(name,date,period,time_now))

    conn.commit()
    conn.close()

# -------------------------
# MARK OUT
# -------------------------
def mark_out(name):

    period = get_current_period()
    if period is None:
        return

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    now = datetime.now()
    date = now.strftime("%Y-%m-%d")
    time_now = now.strftime("%H:%M:%S")

    cursor.execute("""
    SELECT * FROM attendance
    WHERE name=? AND date=? AND period=?
    """,(name,date,period))

    record = cursor.fetchone()

    if record and record[5] is None:

        in_time = datetime.strptime(record[4],"%H:%M:%S")
        out_time = datetime.strptime(time_now,"%H:%M:%S")

        duration = out_time - in_time

        cursor.execute("""
        UPDATE attendance
        SET out_time=?, duration=?
        WHERE id=?
        """,(time_now,str(duration),record[0]))

    conn.commit()
    conn.close()

# -------------------------
# CAMERA
# -------------------------
camera = cv2.VideoCapture(0)

camera.set(cv2.CAP_PROP_FRAME_WIDTH,640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT,480)

prev_frame = None

def gen_frames():

    global last_seen, prev_frame

    while True:

        success, img = camera.read()

        if not success or img is None:
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray,(21,21),0)

        movement_detected = False

        if prev_frame is None:
            prev_frame = gray
        else:

            frame_diff = cv2.absdiff(prev_frame,gray)
            thresh = cv2.threshold(frame_diff,25,255,cv2.THRESH_BINARY)[1]

            movement = np.sum(thresh)

            if movement > 150000:
                movement_detected = True

        prev_frame = gray

        imgS = cv2.resize(img,(0,0),None,0.25,0.25)
        imgS = cv2.cvtColor(imgS,cv2.COLOR_BGR2RGB)

        faces = face_recognition.face_locations(imgS)
        encodes = face_recognition.face_encodings(imgS,faces)

        for encodeFace,faceLoc in zip(encodes,faces):

            matches = face_recognition.compare_faces(encodeListKnown,encodeFace)
            faceDis = face_recognition.face_distance(encodeListKnown,encodeFace)

            matchIndex = np.argmin(faceDis)

            name = "UNKNOWN"
            color = (0,0,255)

            if matches[matchIndex]:

                name = classNames[matchIndex].upper()
                color = (0,255,0)

                if name not in last_seen:
                    mark_in(name)

                last_seen[name] = time.time()

                now = time.time()

                if movement_detected and (name not in movement_log or now - movement_log[name] > 30):
                    save_movement(name)
                    movement_log[name] = now

            y1,x2,y2,x1 = faceLoc
            y1,x2,y2,x1 = y1*4,x2*4,y2*4,x1*4

            cv2.rectangle(img,(x1,y1),(x2,y2),color,2)
            cv2.rectangle(img,(x1,y1-35),(x2,y1),color,cv2.FILLED)

            cv2.putText(img,name,(x1+6,y1-6),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,(255,255,255),2)

        current_time = time.time()

        for name in list(last_seen.keys()):

            if current_time - last_seen[name] > 60:

                mark_out(name)
                del last_seen[name]

        ret,buffer = cv2.imencode(".jpg",img)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n'+frame+b'\r\n')

# -------------------------
# ROUTES
# -------------------------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/video")
def video():
    return Response(gen_frames(),
    mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/attendance")
def attendance():

    conn = sqlite3.connect(DB_PATH)

    df = pd.read_sql_query("SELECT * FROM attendance", conn)

    conn.close()

    period = get_current_period()
    today = datetime.now().strftime("%Y-%m-%d")

    if period is not None:
        current_period_df = df[(df["date"] == today) & (df["period"] == period)]
        total = len(current_period_df)
    else:
        total = 0

    return render_template(
        "attendance.html",
        tables=[df.to_html(classes="data", index=False)],
        period=period,
        total=total
    )

@app.route('/register')
def register():
    return render_template('register.html')

@app.route('/capture', methods=['POST'])
def capture():

    global images, classNames, encodeListKnown

    name = request.form['name']
    roll = request.form['roll']

    cap = cv2.VideoCapture(0)

    count = 0
    max_images = 10

    while count < max_images:

        success, frame = cap.read()

        if not success:
            continue

        faces = face_recognition.face_locations(frame)

        if len(faces) > 0:

            filename = f"{DATASET_PATH}/{name}_{roll}_{count}.jpg"
            cv2.imwrite(filename, frame)

            count += 1

        cv2.putText(frame, f"Images Captured: {count}/{max_images}",
                    (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0,255,0), 2)

        cv2.imshow("Capturing Faces - Look at Camera", frame)

        cv2.waitKey(200)

    cap.release()
    cv2.destroyAllWindows()

    images, classNames = load_dataset()
    encodeListKnown = findEncodings(images)

    return "<h2>Student Registered Successfully</h2><a href='/'>Back</a>"

if __name__ == "__main__":
    app.run(debug=True)