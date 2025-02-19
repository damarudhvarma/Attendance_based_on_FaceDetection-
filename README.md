

# Face Recognition-Based Attendance System

This project implements a *Face Recognition-Based Attendance System* using Python, OpenCV, Dlib, and SQLite. It captures images using a webcam, recognizes individuals based on their face features, and marks attendance automatically in a database.

---

## *Features*
- Real-time face detection and recognition.
- Automatic attendance marking.
- Secure and scalable SQLite database for storing attendance.
- Easy-to-use scripts for dataset collection and feature extraction.

---

## *Installation*
Follow these steps to set up and run the project:

### *1. Clone the Repository*
```
git clone https://github.com/damarudhvarma/Attendance_based_on_FaceDetection-.git
```

### *2. Install Dependencies*
Navigate to the project directory and install the required Python packages:
```
pip install -r requirements.txt
```

### *3. Download Dlib Models*
Download the required Dlib models from the [Google Drive link](https://drive.google.com/drive/folders/1h07D2K8SQ2AjgMPUs2KKS52u8xT3Uzqy?usp=sharing).  
- Place the downloaded data folder inside the root directory of the project.

---

## *Usage*

1. *Collect Face Dataset*  
   Run the following script to capture face images for the dataset:  
   ```
   python get_faces_from_camera.py
   ```
   
   This will create a folder with captured face images for each individual.

2. *Extract Face Features*  
   Extract 128D face features from the dataset and save them in a CSV file:  
   ```
   python features_extraction.py
   ```
3. *Register Student Details*  
   Register student details (Roll No, Name, Section, etc.) in the database by running:  
   ```
   python class_registration.py
   ```

   

4. *Mark Attendance*  
   Start the attendance system to detect faces and mark attendance:  
   ```
   python attendance_taker.py
   ```
   

5. *View Attendance Database*  
   Run the following script to view attendance records stored in the SQLite database:  
   ```
   python app.py
   ```

---

## *Directory Structure*
```
Attendance_based_on_FaceDetection-
├── data/
│   ├── features_all.csv           # Extracted face features
│   ├── shape_predictor_68_face_landmarks.dat  # Dlib face landmark model
│   ├── dlib_face_recognition_resnet_model_v1.dat  # Dlib face recognition model
├── attendance.db                  # SQLite database for attendance
├── get_faces_from_camera.py       # Script to collect face datasets
├── class_registration.py          # Script to register students
├── features_extraction.py         # Script to extract face features
├── attendance_taker.py            # Script for face recognition and attendance
├── app.py                         # Script to interact with the database
├── requirements.txt               # Python dependencies
└── README.md                      # Documentation
```

---

## *Prerequisites*
- Python 3.7 or higher
- Dlib (with shape_predictor and ResNet models)
- OpenCV
- Pandas
- SQLite3

---

## *Demo*
1. *Face Detection and Registration:* Captures images of individuals for database creation.  
2. *Real-Time Attendance:* Automatically recognizes faces and marks attendance in the database.  
3. *Database View:* View the recorded attendance with date and time.



## *License*
This project is licensed under the MIT License. See the LICENSE file for more details.

---

## *Acknowledgements*
- Dlib library for face detection and recognition models.
- OpenCV for real-time image processing.
- SQLite for database management.

--- 

Let me know if you'd like any further customization!
