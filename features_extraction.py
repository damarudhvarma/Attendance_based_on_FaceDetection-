import os
import dlib
import csv
import numpy as np
import logging
import cv2

# Path of cropped faces
path_images_from_camera = "data/data_faces_from_camera/"

# Use Dlib's frontal face detector
detector = dlib.get_frontal_face_detector()

# Get face landmarks
predictor = dlib.shape_predictor('data/data_dlib/shape_predictor_68_face_landmarks.dat')

# Use Dlib resnet50 model to get 128D face descriptor
face_reco_model = dlib.face_recognition_model_v1("data/data_dlib/dlib_face_recognition_resnet_model_v1.dat")


def return_128d_features(path_img):
    """
    Extract 128D features for a single image.
    Returns the feature vector if a face is detected, else returns 0.
    """
    img_rd = cv2.imread(path_img)
    faces = detector(img_rd, 1)

    if len(faces) != 0:
        shape = predictor(img_rd, faces[0])
        face_descriptor = face_reco_model.compute_face_descriptor(img_rd, shape)
    else:
        face_descriptor = 0
        logging.warning("No face detected in: %s", path_img)
    return face_descriptor


def return_features_mean_personX(path_face_personX):
    """
    Calculate the mean face feature vector for all images in a folder.
    Skips images where no face is detected.
    """
    features_list_personX = []
    photos_list = os.listdir(path_face_personX)

    for photo in photos_list:
        photo_path = os.path.join(path_face_personX, photo)
        logging.info("Reading image: %s", photo_path)
        features_128d = return_128d_features(photo_path)
        if features_128d != 0:
            features_list_personX.append(features_128d)

    if features_list_personX:
        features_mean_personX = np.array(features_list_personX, dtype=object).mean(axis=0)
    else:
        features_mean_personX = np.zeros(128, dtype=object)
        logging.warning("No valid face features for: %s", path_face_personX)

    return features_mean_personX


def main():
    logging.basicConfig(level=logging.INFO)

    # Prepare output CSV file
    output_csv = "data/features_all.csv"
    with open(output_csv, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        # writer.writerow(["folder_name", "person_name"] + [f"feature_{i}" for i in range(128)])

        # Traverse the sections and persons
        for section_name in os.listdir(path_images_from_camera):
            section_path = os.path.join(path_images_from_camera, section_name)
            if os.path.isdir(section_path):
                for person_folder in os.listdir(section_path):
                    person_path = os.path.join(section_path, person_folder)
                    if os.path.isdir(person_path):
                        logging.info("Processing folder: %s", person_path)

                        # Extract features for the person
                        features_mean = return_features_mean_personX(person_path)
                        
                        # Split the person name (e.g., "person_1_varma") to extract readable name
                        person_name = person_folder.split("_", 2)[-1]

                        # Combine section name, person name, and feature vector
                        row = [section_name, person_name] + features_mean.tolist()
                        writer.writerow(row)

        logging.info("Saved all face features into: %s", output_csv)


if __name__ == '__main__':
    main()
