import cv2
import dlib
import PIL.Image
import numpy as np
from imutils import face_utils
import argparse
from pathlib import Path
import os
import ntpath

parser = argparse.ArgumentParser(description='Application Simple de Reconnaissance Faciale')
parser.add_argument('-i', '--input', type=str, required=True, help='répertoire des visages connus')

print('[INFO] Démarrage du système...')
print('[INFO] Importation du modèle pré-entraîné..') 
pose_predictor_68_point = dlib.shape_predictor("/home/deo/Téléchargements/dlib/pretrained_model/shape_predictor_68_face_landmarks.dat")
pose_predictor_5_point = dlib.shape_predictor("/home/deo/Téléchargements/dlib/pretrained_model/shape_predictor_5_face_landmarks.dat")
face_encoder = dlib.face_recognition_model_v1("/home/deo/Téléchargements/dlib_face_recognition_resnet_model_v1.dat")
face_detector = dlib.get_frontal_face_detector()
print('[INFO] Importation du modèle pré-entraîné..')

def transform(image, face_locations):
    coord_faces = []
    for face in face_locations:
        rect = face.top(), face.right(), face.bottom(), face.left()
        coord_face = max(rect[0], 0), min(rect[1], image.shape[1]), min(rect[2], image.shape[0]), max(rect[3], 0)
        coord_faces.append(coord_face)
    return coord_faces

def encode_face(image):
    face_locations = face_detector(image, 1)
    face_encodings_list = []
    landmarks_list = []
    for face_location in face_locations:
        # DÉTECTION DES VISAGES
        shape = pose_predictor_68_point(image, face_location)
        face_encodings = np.array(face_encoder.compute_face_descriptor(image, shape, num_jitters=1))
        if len(face_encodings) > 0:
            face_encodings_list.append(face_encodings)
            # OBTENIR LES POINTS DE REPÈRE
            shape = face_utils.shape_to_np(shape)
            landmarks_list.append(shape)
        else:
            print("Aucun visage n'a été détecté dans cette partie de l'image.")

    face_locations = transform(image, face_locations)
    return face_encodings_list, face_locations, landmarks_list

def easy_face_reco(frame, known_face_encodings, known_face_names):
    rgb_small_frame = frame[:, :, ::-1]
    # ENCODAGE DES VISAGES
    face_encodings_list, face_locations_list, landmarks_list = encode_face(rgb_small_frame)
    face_names = []
    for face_encoding in face_encodings_list:
        # VÉRIFIER LA DISTANCE ENTRE LES VISAGES CONNUS ET LES VISAGES DÉTECTÉS
        vectors = np.linalg.norm(known_face_encodings - face_encoding, axis=1)
        tolerance = 0.6
        result = []
        for vector in vectors:
            if vector <= tolerance:
                result.append(True)
            else:
                result.append(False)
        if True in result:
            first_match_index = result.index(True)
            name = known_face_names[first_match_index]
        else:
            name = "Inconnu"
        face_names.append(name)

    for (top, right, bottom, left), name in zip(face_locations_list, face_names):
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.rectangle(frame, (left, bottom - 30), (right, bottom), (0, 255, 0), cv2.FILLED)
        cv2.putText(frame, name, (left + 2, bottom - 2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1)

    for shape in landmarks_list:
        for (x, y) in shape:
            cv2.circle(frame, (x, y), 1, (255, 0, 255), -1)

if __name__ == '__main__':
    args = parser.parse_args()

    print('[INFO] Importation des visages...')
    face_to_encode_path = Path(args.input)
    files = [file_ for file_ in face_to_encode_path.rglob('*.jpg')]

    for file_ in face_to_encode_path.rglob('*.png'):
        files.append(file_)
    if len(files)==0:
        raise ValueError('Aucun visage détecté dans le répertoire : {}'.format(face_to_encode_path))
    known_face_names = [os.path.splitext(ntpath.basename(file_))[0] for file_ in files]

    known_face_encodings = []
    for file_ in files:
        image = PIL.Image.open(file_)
        image = np.array(image)
        encodings = encode_face(image)
        if encodings:
            face_encoded = encodings[0][0]
            known_face_encodings.append(face_encoded)
        else:
            # Gérer le cas où aucun visage n'est détecté
            print("Aucun visage n'a été détecté dans l'image.")

    print('[INFO] Visages bien importés')
    print('[INFO] Démarrage de la webcam...')
    video_capture = cv2.VideoCapture(0)
    print('[INFO] Webcam démarrée avec succès')
    print('[INFO] Détection...')
    while True:
        ret, frame = video_capture.read()
        easy_face_reco(frame, known_face_encodings, known_face_names)
        cv2.imshow('Application Simple de Reconnaissance Faciale', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    print('[INFO] Arrêt du système')
    video_capture.release()
    cv2.destroyAllWindows()

