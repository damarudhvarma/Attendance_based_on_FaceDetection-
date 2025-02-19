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
    login_time TIME,
    logout_time TIME DEFAULT NULL,
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
        self.last_attendance_time = time.time()
        self.last_marked_faces = {}

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

    def mark_attendance(self, rollno, name):
        current_date = datetime.datetime.now().strftime('%Y-%m-%d')
        current_time = datetime.datetime.now().strftime('%H:%M:%S')

        conn = sqlite3.connect("attendance.db")
        cursor = conn.cursor()
        cursor.execute("SELECT login_time FROM attendance WHERE rollno = ? AND date = ?", (rollno, current_date))
        record = cursor.fetchone()

        if record:
            login_time = datetime.datetime.strptime(record[0], '%H:%M:%S')
            elapsed_time = (datetime.datetime.strptime(current_time, '%H:%M:%S') - login_time).total_seconds()
            
            if elapsed_time >= 20 and rollno not in self.last_marked_faces:  # update time elapsed here
                cursor.execute("UPDATE attendance SET logout_time = ? WHERE rollno = ? AND date = ?", (current_time, rollno, current_date))
                print(f"{name} (Roll No: {rollno}) logged out at {current_time}")
            else:
                print(f"{name} (Roll No: {rollno}) cannot logout before 5 minutes")
        else:
            cursor.execute("INSERT INTO attendance (rollno, name, section, date, login_time) VALUES (?, ?, ?, ?, ?)",
                           (rollno, name, 'CSE-B', current_date, current_time))
            print(f"{name} (Roll No: {rollno}) logged in at {current_time}")
            self.last_marked_faces[rollno] = current_time  # Store login time
        
        conn.commit()
        conn.close()

    @staticmethod
    def return_euclidean_distance(feature_1, feature_2):
        return np.linalg.norm(np.array(feature_1) - np.array(feature_2))

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

                    for j in range(len(self.face_features_known_list)):
                        e_distance = self.return_euclidean_distance(face_feature, self.face_features_known_list[j])
                        if e_distance < min_distance:
                            min_distance = e_distance
                            best_rollno = self.face_rollno_known_list[j]
                            best_name = self.face_name_known_list[j]

                    if min_distance < 0.6:
                        cv2.rectangle(img_rd, (face.left(), face.top()), (face.right(), face.bottom()), (0, 255, 0), 2)
                        cv2.putText(img_rd, best_name, (face.left(), face.top() - 10), self.font, 0.8, (0, 255, 255), 1)
                        
                        if current_time - self.last_attendance_time >= 5:
                            self.mark_attendance(best_rollno, best_name)
                            self.last_attendance_time = current_time
                    else:
                        cv2.rectangle(img_rd, (face.left(), face.top()), (face.right(), face.bottom()), (0, 0, 255), 2)
                        cv2.putText(img_rd, "Unknown", (face.left(), face.top() - 10), self.font, 0.8, (0, 0, 255), 1)

            cv2.imshow("Attendance System", img_rd)

            if cv2.waitKey(1) == ord('q'):
                break

        stream.release()
        cv2.destroyAllWindows()

    def run(self):
        cap = cv2.VideoCapture(0)
        self.process(cap)


def main():
    logging.basicConfig(level=logging.INFO)
    Face_Recognizer().run()


if __name__ == '__main__':
    main()
