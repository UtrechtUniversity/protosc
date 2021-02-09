#!/usr/bin/env python

from pathlib import Path

stim_data_dir = Path("...", "data", "Nimstim faces")
addperc = 20

if __name__ == "__main__":

    add = 1 + addperc / 100
    min = 1 - addperc / 100

    # List of all possible categories
    categories = ['ca_c', 'ca_o', 'an_c', 'an_o', 'ha_c',
                  'ha_o', 'ha_x', 'fe_c', 'fe_o', 'sa_c',
                  'sa_o', 'sp_o', 'ne_c', 'ne_o', 'di_c',
                  'di_o']

    for category in categories:

        # Read all images for specific category
        files = list(stim_data_dir.glob(f"*{category}.bmp"))

        for i in range(len(files)):
            original_image = cv.imread(str(files[i]))

            # Transform image to grayscale
            grayscale_image = cv.cvtColor(original_image, cv.COLOR_BGR2GRAY)

            # Get orientation points of face in image
            faceCascade = cv.CascadeClassifier(cv.data.haarcascades + "haarcascade_frontalface_default.xml")
            faces = faceCascade.detectMultiScale(
                grayscale_image,
                scaleFactor=1.3,
                minNeighbors=3,
                minSize=(30, 30)
            )

            # Crop face (with additional percentage) and safe as 200x200 pixels image
            for (x, y, w, h) in faces:
                roi_color = grayscale_image[int(y*min):int(y*add) + h, int(x*min):int(x*add) + w]
                roi_color = cv.resize(roi_color, (200, 200))
                cv.imwrite(files[i].stem + '_faces.jpg', roi_color)
