import cv2
import numpy as np
import pandas as pd
import re
from pathlib import Path

def execute():
    """ Execute preprocessing for all images """
    df = create_csv()
    final = []

    for image_id in range(df.shape[0]):
        file = df['file'][image_id]
        try:
            final.append(preprocessing(file))
        except:
            next
        # final = preprocessing(file)
        # df.loc[image_id, 'preprocessed'] = [final]
        # cv2.imwrite(f'{image_id}_final.jpg', final)

    # df.to_csv('overview.csv')

    return final

def preprocessing(img):
    """ Preprocess image to get gray-scaled, circled image of the face """

    original_image = cv2.imread(str(img))
    grey_image = greyscale(original_image)
    vj_image = viola_jones(grey_image)
    final_image = cut_circle(vj_image)

    return final_image


def create_csv(path=Path("U:\prosoc", "data", "Nimstim faces")):
    """ Create dataframe with all image IDs and sex, emotion, and mouth position depicted on image"""

    # Read all images
    files_path = list(path.glob('*.bmp'))
    files = list(x.stem for x in files_path)

    # Create codebook with the meaning of all abbreviations
    codebook = {0: {'female', 'male'},
           1: {'calm', 'angry', 'happy', 'fear', 'sad', 'surprised', 'neutral', 'disgust'},
           2: {'open', 'closed', 'exuberant'}}
    output = [[], [], []]

    # Decode sex, emotion, and mouth position depicted on image
    for file in files:
        file_parts = file.split('_')
        file_parts[0] = file_parts[0][-1]

        for i in range(0, 3):
            try:
                code = re.findall('(?<=\')' + file_parts[i].lower() + '\w*', str(codebook[i]))[0]
                output[i].append(code)
            except:
                if file_parts[i].lower() == 'sp':
                    output[i].append('surprised')
                elif file_parts[i].lower() == 'x':
                    output[i].append('exuberant')
                else:
                    next

    # Create dataframe with all collected info about the image
    df = {'file': files_path,
          'sex': output[0],
          'emotion': output[1],
          'mouth': output[2]}
    df = pd.DataFrame(df)

    return df


def greyscale(img):
    """ Create a gray-scaled version of the image """

    img_array = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img_array.reshape(*img_array.shape, 1)


def viola_jones(img, add_perc=20):
    """ Crop the face out of the image"""

    # Get orientation points of face in image
    faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades +
                                        "haarcascade_frontalface_default.xml")
    faces = faceCascade.detectMultiScale(
        img,
        scaleFactor=1.3,
        minNeighbors=3,
        minSize=(30, 30)
    )

    # Crop face (with additional percentage) and safe as 200x200 pixels image
    margin_plus = 1 + add_perc / 100
    margin_min = 1 - add_perc / 100
    for (x, y, w, h) in faces:
        roi_color = img[int(round(y * margin_min)):int(round(y * margin_plus)) + h,
                    int(round(x * margin_min)):int(round(x * margin_plus)) + w]
        roi_color = cv2.resize(roi_color, (200, 200))

    return roi_color


def cut_circle(img):
    """ Crop the image to a circle """

    shape = img.shape
    assert len(img.shape) >= 2

    X, Y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]))
    middle = np.array([shape[0] // 2, shape[1] // 2])
    X -= middle[0]
    Y -= middle[1]

    circle_mask = (np.sqrt(X ** 2 + Y ** 2).reshape(shape[:2]) >
                   min(img.shape[0] // 2, img.shape[1] // 2))
    new_img = np.copy(img)
    new_img[circle_mask] = 0

    return new_img
