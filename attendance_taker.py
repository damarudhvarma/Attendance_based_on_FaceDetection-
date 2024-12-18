import dlib
import numpy as np
import cv2
import os
import pandas as pd
import time
import logging
import sqlite3
import datetime

# Dlib / Use frontal face detector of Dlib
detector = dlib.get_frontal_face_detector()

# Dlib landmark / Get face landmarks
predictor = dlib.shape_predictor('data/data_dlib/shape_predictor_68_face_landmarks.dat')

# Dlib Resnet / Use Dlib ResNet50 model to get 128D face descriptor
face_reco_model = dlib.face_recognition_model_v1("data/data_dlib/dlib_face_recognition_resnet_model_v1.dat")

# Create a connection to the database
conn = sqlite3.connect("attendance.db")
cursor = conn.cursor()

# Create a table for attendance with Rollno, Name, Section, Date, and Time columns
table_name = "attendance" 
create_table_sql = f"""
CREATE TABLE IF NOT EXISTS {table_name} (
    rollno TEXT, 
    name TEXT, 
    section TEXT DEFAULT 'CSE-B', 
    date DATE, 
    time TIME,
    UNIQUE(rollno, date)
)"""
cursor.execute(create_table_sql)

# Commit changes and close the connection
conn.commit()
conn.close()


class Face_Recognizer:
    def __init__(self):
        self.font = cv2.FONT_ITALIC
        
        # FPS
        self.frame_time = 0
        self.frame_start_time = 0
        self.fps = 0
        self.fps_show = 0
        self.start_time = time.time()

        # Frame counter
        self.frame_cnt = 0

        # Known face features and names
        self.face_features_known_list = []
        self.face_rollno_known_list = []  # To store roll numbers
        self.face_name_known_list = []  # To store names

        # Centroid positions of ROI
        self.last_frame_face_centroid_list = []
        self.current_frame_face_centroid_list = []

        # Names of detected faces
        self.last_frame_face_name_list = []
        self.current_frame_face_name_list = []

        # Face count
        self.last_frame_face_cnt = 0
        self.current_frame_face_cnt = 0

        # E-distances for recognition
        self.current_frame_face_X_e_distance_list = []
        self.current_frame_face_position_list = []
        self.current_frame_face_feature_list = []

        # Frame reclassification interval
        self.reclassify_interval_cnt = 0
        self.reclassify_interval = 10

    # Read features from "features_all.csv"
    def get_face_database(self):
        if os.path.exists("data/features_all.csv"):
            path_features_known_csv = "data/features_all.csv"
            csv_rd = pd.read_csv(path_features_known_csv, header=None)

            for i in range(csv_rd.shape[0]):
                features_someone_arr = []
                
                # Extract roll number and name
                rollno = csv_rd.iloc[i][0]
                name = csv_rd.iloc[i][1]
                
                self.face_rollno_known_list.append(rollno)
                self.face_name_known_list.append(name)

                # Extract features starting from the 3rd column
                for j in range(2, 130):
                    if csv_rd.iloc[i][j] == '':
                        features_someone_arr.append('0')
                    else:
                        features_someone_arr.append(csv_rd.iloc[i][j])
                
                self.face_features_known_list.append(features_someone_arr)
            
            logging.info("Faces in Database: %d", len(self.face_features_known_list))
            return 1
        else:
            logging.warning("'features_all.csv' not found!")
            logging.warning("Please run 'get_faces_from_camera.py' and 'features_extraction_to_csv.py'")
            return 0

    # Insert attendance into the database
    def attendance(self, rollno, name):
        current_date = datetime.datetime.now().strftime('%Y-%m-%d')
        current_time = datetime.datetime.now().strftime('%H:%M:%S')
        conn = sqlite3.connect("attendance.db")
        cursor = conn.cursor()

        # Check for existing entry
        cursor.execute("SELECT * FROM attendance WHERE rollno = ? AND date = ?", (rollno, current_date))
        existing_entry = cursor.fetchone()

        if existing_entry:
            print(f"{name} (Roll No: {rollno}) is already marked as present for {current_date}")
        else:
            cursor.execute("INSERT INTO attendance (rollno, name, section, date, time) VALUES (?, ?, ?, ?, ?)",
                           (rollno, name, 'CSE-B', current_date, current_time))
            conn.commit()
            print(f"{name} (Roll No: {rollno}) marked as present for {current_date} at {current_time}")

        conn.close()

    @staticmethod
    def return_euclidean_distance(feature_1, feature_2):
        feature_1 = np.array(feature_1)
        feature_2 = np.array(feature_2)
        return np.sqrt(np.sum(np.square(feature_1 - feature_2)))

    # Process video stream
    def process(self, stream):
        if self.get_face_database():
            while stream.isOpened():
                self.frame_cnt += 1
                flag, img_rd = stream.read()
                kk = cv2.waitKey(1)

                # Detect faces in the frame
                faces = detector(img_rd, 0)
                self.current_frame_face_cnt = len(faces)

                self.current_frame_face_name_list = []
                self.current_frame_face_X_e_distance_list = []
                self.current_frame_face_centroid_list = []

                if self.current_frame_face_cnt > 0:
                    for i, face in enumerate(faces):
                        shape = predictor(img_rd, face)
                        face_feature = face_reco_model.compute_face_descriptor(img_rd, shape)
                        self.current_frame_face_feature_list.append(face_feature)

                        min_e_distance = float("inf")
                        rollno = "unknown"
                        name = "unknown"

                        # Compare with known faces
                        for j in range(len(self.face_features_known_list)):
                            e_distance = self.return_euclidean_distance(face_feature, self.face_features_known_list[j])
                            if e_distance < min_e_distance:
                                min_e_distance = e_distance
                                rollno = self.face_rollno_known_list[j]
                                name = self.face_name_known_list[j]

                        if min_e_distance < 0.6:  # Recognition threshold
                            self.current_frame_face_name_list.append(name)
                            self.attendance(rollno, name)  # Mark attendance
                        else:
                            self.current_frame_face_name_list.append("unknown")

                        img_rd = cv2.rectangle(img_rd, (face.left(), face.top()), (face.right(), face.bottom()), (255, 255, 255), 2)
                        img_rd = cv2.putText(img_rd, name, (face.left(), face.top() - 10), self.font, 0.8, (0, 255, 255), 1, cv2.LINE_AA)

                cv2.imshow("camera", img_rd)

                if kk == ord('q'):  # Quit on 'q'
                    break

        stream.release()
        cv2.destroyAllWindows()

    def run(self):
        cap = cv2.VideoCapture(0)  # Use camera
        self.process(cap)


def main():
    logging.basicConfig(level=logging.INFO)
    Face_Recognizer_con = Face_Recognizer()
    Face_Recognizer_con.run()


if __name__ == '__main__':
    main()
