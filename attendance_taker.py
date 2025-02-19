import dlib
import numpy as np
import cv2
import os
import pandas as pd
import time
import logging
import sqlite3
import datetime

# Dlib setup
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('data/data_dlib/shape_predictor_68_face_landmarks.dat')
face_reco_model = dlib.face_recognition_model_v1("data/data_dlib/dlib_face_recognition_resnet_model_v1.dat")

# Database setup
conn = sqlite3.connect("attendance.db")
cursor = conn.cursor()
cursor.execute("""
CREATE TABLE IF NOT EXISTS attendance (
    rollno TEXT, 
    name TEXT, 
    section TEXT DEFAULT 'CSE-B', 
    date DATE, 
    time TIME,
    UNIQUE(rollno, date)
)
""")
conn.commit()
conn.close()


class Face_Recognizer:
    def __init__(self):
        self.font = cv2.FONT_ITALIC
        self.face_features_known_list = []
        self.face_rollno_known_list = []
        self.face_name_known_list = []
        self.last_attendance_time = time.time()  # Timer for attendance marking
        self.last_marked_faces = set()  # Store last recognized faces

    # Read features from "features_all.csv"
    def get_face_database(self):
        if os.path.exists("data/features_all.csv"):
            csv_rd = pd.read_csv("data/features_all.csv", header=None)
            for i in range(csv_rd.shape[0]):
                rollno, name = csv_rd.iloc[i][0], csv_rd.iloc[i][1]
                self.face_rollno_known_list.append(rollno)
                self.face_name_known_list.append(name)
                features = [float(csv_rd.iloc[i][j]) if csv_rd.iloc[i][j] != '' else 0 for j in range(2, 130)]
                self.face_features_known_list.append(features)

            logging.info(f"Faces in Database: {len(self.face_features_known_list)}")
            return True
        else:
            logging.warning("'features_all.csv' not found!")
            return False

    # Mark attendance in the database
    def mark_attendance(self, rollno, name):
        current_date = datetime.datetime.now().strftime('%Y-%m-%d')
        current_time = datetime.datetime.now().strftime('%H:%M:%S')

        conn = sqlite3.connect("attendance.db")
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM attendance WHERE rollno = ? AND date = ?", (rollno, current_date))
        if cursor.fetchone():
            print(f"{name} (Roll No: {rollno}) is already marked present for {current_date}")
        else:
            cursor.execute("INSERT INTO attendance (rollno, name, section, date, time) VALUES (?, ?, ?, ?, ?)",
                           (rollno, name, 'CSE-B', current_date, current_time))
            conn.commit()
            print(f"{name} (Roll No: {rollno}) marked as present at {current_time}")
            print("✅ Marked Present, Next Student ➡️")

        conn.close()

    @staticmethod
    def return_euclidean_distance(feature_1, feature_2):
        return np.linalg.norm(np.array(feature_1) - np.array(feature_2))

    # Process video stream
    def process(self, stream):
        if not self.get_face_database():
            return

        while stream.isOpened():
            flag, img_rd = stream.read()
            if not flag:
                break

            faces = detector(img_rd, 0)
            current_time = time.time()

            if len(faces) > 0:
                for face in faces:
                    shape = predictor(img_rd, face)
                    face_feature = face_reco_model.compute_face_descriptor(img_rd, shape)

                    min_distance = float("inf")
                    best_rollno, best_name = "unknown", "unknown"

                    # Compare with known faces
                    for j in range(len(self.face_features_known_list)):
                        e_distance = self.return_euclidean_distance(face_feature, self.face_features_known_list[j])
                        if e_distance < min_distance:
                            min_distance = e_distance
                            best_rollno = self.face_rollno_known_list[j]
                            best_name = self.face_name_known_list[j]

                    if min_distance < 0.6:  # Recognition threshold
                        cv2.rectangle(img_rd, (face.left(), face.top()), (face.right(), face.bottom()), (0, 255, 0), 2)
                        cv2.putText(img_rd, best_name, (face.left(), face.top() - 10), self.font, 0.8, (0, 255, 255), 1)

                        # Take attendance only once every 5 seconds
                        if current_time - self.last_attendance_time >= 5 and best_rollno not in self.last_marked_faces:
                            self.mark_attendance(best_rollno, best_name)
                            self.last_attendance_time = current_time  # Reset timer
                            self.last_marked_faces.add(best_rollno)  # Prevent immediate re-marking

                    else:
                        cv2.rectangle(img_rd, (face.left(), face.top()), (face.right(), face.bottom()), (0, 0, 255), 2)
                        cv2.putText(img_rd, "Unknown", (face.left(), face.top() - 10), self.font, 0.8, (0, 0, 255), 1)

            cv2.imshow("Attendance System", img_rd)

            if cv2.waitKey(1) == ord('q'):  # Press 'q' to quit
                break

        stream.release()
        cv2.destroyAllWindows()

    def run(self):
        cap = cv2.VideoCapture(0)  # Open webcam
        self.process(cap)


def main():
    logging.basicConfig(level=logging.INFO)
    Face_Recognizer().run()


if __name__ == '__main__':
    main()
