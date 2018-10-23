import math
import pprint
import os
import shutil
from multiprocessing import Pool
from functools import partial
import numpy as np
import dlib
import cv2


def generate_face(img):
    # Tamanho extra para recortar o rosto
    N = 60
    # Shape para calcular algumas questões...
    if not isinstance(img, np.ndarray):
        return None
    wg, hg = img.shape
    
    # Cortar exatamente a posição da face
    for face_cascade in face_cascades:
        faces = face_cascade.detectMultiScale(img, 1.1, 1, minSize=(15, 15), flags=cv2.CASCADE_SCALE_IMAGE)
        for face in faces:
            (x, y, w, h) = face
            yN, xN = np.abs(y-N), np.abs(x-N)
            yNh, xNw = np.abs((y+N)+h), np.abs((x+N)+w)
            crop = img[yN:yNh, xN:xNw]

            # Adicionar os pontos de identificação da face
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            clahe_image = clahe.apply(crop)
            detections = detector(clahe_image, 1) #Detect the faces in the image
            if len(list(detections)):
                img = crop
                break
    
    # Resize, para ficarem todas do mesmo tamanho
    return cv2.resize(img, (225, 225))


def save_image(emotional_img, i, directory, emt):
    img = cv2.imread(emotional_img, 0)
    if isinstance(img, np.ndarray):
        emot = generate_face(img)
        cv2.imwrite(f'{directory}/{emt}.{i}.png', emot)
    return True


# Face detector
detector = dlib.get_frontal_face_detector()
# Landmark identifier. Set the filename to whatever you named the downloaded file
predictor = dlib.shape_predictor('classifier/shape_predictor_68_face_landmarks.dat')

face_cascades = [
    cv2.CascadeClassifier('classifier/haarcascade_frontalface_default.xml'),
    cv2.CascadeClassifier('classifier/haarcascade_frontalface_alt.xml'),
    cv2.CascadeClassifier('classifier/haarcascade_frontalface_alt2.xml'),
    cv2.CascadeClassifier('classifier/haarcascade_frontalface_alt_tree.xml'),
]

emotions_mapper = {
    '0': 'neutral', '1': 'anger', '2':'contempt',
    '3': 'disgust', '4': 'fear', '5': 'happy',
    '6': 'sadness', '7': 'surprise'
}

pp = pprint.PrettyPrinter(indent=4)

if __name__ == '__main__':
    emotion = {}

    # Emotions
    for em_dr in os.listdir('emotion'):
        for em in os.listdir(f'emotion/{em_dr}'):
            for fl in os.listdir(f'emotion/{em_dr}/{em}'):
                with open(f'emotion/{em_dr}/{em}/{fl}', 'r') as hem:
                    emotion[(em_dr, em)] = {
                        'emotion': int(float(hem.read())),
                        'images': {
                            'neutral': None,
                            'emotional': []
                        }
                    }
    
    # Images
    for k in emotion.keys():
        dr = f'images/{k[0]}/{k[1]}'
        if os.path.isdir(dr):
            imgs = sorted(os.listdir(dr))
            # for img in [imgs[0], imgs[-2]]:
            if len(imgs) > 1:
                emotion[(k[0], k[1])]['images']['neutral'] = f'{dr}/{imgs[0]}'
                emotion[(k[0], k[1])]['images']['emotional'].append(f'{dr}/{imgs[-6]}')
                emotion[(k[0], k[1])]['images']['emotional'].append(f'{dr}/{imgs[-4]}')
                emotion[(k[0], k[1])]['images']['emotional'].append(f'{dr}/{imgs[-2]}')
    
    # pp.pprint(emotion)

    print('Iniciando gravação...')
    
    for k in emotion.items():
        emt = emotions_mapper[str(k[1]['emotion'])]
        directory = f'dataset/{k[0][0]}/{k[0][1]}'
        if not os.path.exists(directory):
            os.makedirs(directory)
        neutral = generate_face(cv2.imread(k[1]['images']['neutral'], 0))
        if isinstance(neutral, np.ndarray):
            cv2.imwrite(f'{directory}/neutral.png', neutral)
        with Pool(3) as pool:
            imgs = k[1]['images']['emotional']
            f = partial(save_image, directory=directory, emt=emt)
            r = pool.starmap(f, zip(imgs, range(len(imgs))))
    
    print('Terminado...')
