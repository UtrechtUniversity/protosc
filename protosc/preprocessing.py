import cv2
import numpy as np


def greyscale(img):
    img_array = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img_array.reshape(*img_array.shape, 1)


def viola_jones(img, add_perc=20):
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
        roi_color = img[round(y*margin_min):round(y*margin_plus) + h,
                        round(x*margin_min):round(x*margin_plus) + w]
        roi_color = cv2.resize(roi_color, (200, 200))
    return roi_color
#         cv2.imwrite(files[i].stem + '_faces.jpg', roi_color)


def cut_circle(img):
    shape = img.shape
    assert len(img.shape) >= 2

    X, Y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]))
    middle = np.array([shape[0]//2, shape[1]//2])
    X -= middle[0]
    Y -= middle[1]

    circle_mask = (np.sqrt(X**2 + Y**2).reshape(shape[:2]) >
                   min(img.shape[0]//2, img.shape[1]//2))
    new_img = np.copy(img)
    new_img[circle_mask] = 0
    return new_img
