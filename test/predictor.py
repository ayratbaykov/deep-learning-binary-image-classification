# Quick test of the model
# Call from the command line like this:
#      Python predictor.py image.jpg

from keras.models import load_model

import numpy as np
import cv2
import sys

np.set_printoptions(precision=2, suppress=True)

def loadImage(filename):
    img = cv2.imread(filename, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (224, 150))
    img = img / 255
    # Reshape from (150,224) to (1,150,224,3) : 1 sample, 150x224 pixels, 3 channels
    img = np.expand_dims(img, axis=0)
    img = np.expand_dims(img, axis=0)
    img = np.reshape(img, (1,150,224,3))
    return np.array(img)

model = load_model("../model/trained_model.h5")

print('=========================')
print('For image: ', sys.argv[1])
img = loadImage(sys.argv[1])
classes = model.predict(img)

print('The class is predicted as', format(classes[0][0],'.4f'), 'between 0.0000 (class 1) and 1.0000 (class 2)')