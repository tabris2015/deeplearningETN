# This code is beerware; if you use it, please buy me
# a cold beverage next time you run into one of
# us at the local.
# 2 de julio de 2015- Jose Laruta - Instituto de Electronica Aplicada
# Code developed in python 2.7.3
#****************************************************************/

# modulos necesarios
import cv2
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.datasets import mnist

from keras.models import model_from_json


def nada(x):
    pass


cv2.namedWindow('imagen')
cv2.namedWindow('umbral')

cv2.createTrackbar('nivel bajo', 'umbral', 0, 255, nada)
cv2.createTrackbar('nivel alto', 'umbral', 0, 255, nada)

# cargamos el clasificador

# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
clf = model_from_json(loaded_model_json)
# load weights into new model
clf.load_weights("model.h5")
print("Loaded model from disk")

camara = cv2.VideoCapture(0)

# utilidades para el preprocesamiento
kernel = np.ones((5, 5), np.uint8)

while(1):

    # leemos la imagen de entrada
    _, im = camara.read()
    alto, ancho = im.shape[:2]
    low = int(cv2.getTrackbarPos('nivel bajo', 'umbral'))
    high = int(cv2.getTrackbarPos('nivel alto', 'umbral'))
    # convertimos a escala de grises y aplicamos un filtro gaussiano
    im_gray = cv2. cvtColor(im, cv2.COLOR_BGR2GRAY)
    im_gray = cv2.GaussianBlur(im_gray, (5, 5), 0)
    im_gray = cv2.morphologyEx(im_gray, cv2.MORPH_OPEN, kernel)

    # aplicarle un umbral para binarizar
    ret, im_th = cv2.threshold(im_gray, low, high, cv2.THRESH_BINARY_INV)
    cv2.imshow("umbral", im_th)
    # encontrar los contornos

    _, ctrs, hier = cv2.findContours(
        im_th.copy(),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    # extraer los rectangulos de cada contorno
    rects = [cv2.boundingRect(ctr) for ctr in ctrs]

    # para cada region rectangular, calcular sus features HOG
    # y predecir los digitos usando el clasificador entrenado

    for rect in rects:
        # dibuja los rectangulos
        cv2.rectangle(
            im,
            (rect[0], rect[1]),  # esquina superior
            (rect[0] + rect[2], rect[1] + rect[3]),  # esquina inferior
            (0, 255, 0),  # color
            3
        )
        # dibujarla alrededor del digito
        leng = int(rect[3] * 1.6)
        pt1 = int(rect[1] + rect[3] // 2 - leng // 2)
        pt2 = int(rect[0] + rect[2] // 2 - leng // 2)
        if pt1 < 0 or pt1 + leng > alto or pt2 < 0 or pt2 + leng > ancho or leng > ancho / 2:
            pass
        else:
            roi = im_th[pt1:pt1 + leng, pt2:pt2 + leng]
        # cambiar el tamanho a la imagen
            roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
            roi = cv2.dilate(roi, (3, 3))

            # calcular su HOG
            # roi_hog_fd = hog(
            #			roi,
            #			orientations=9,
            #			pixels_per_cell=(14, 14),
            #			cells_per_block=(1, 1),
            #			visualise=False
            #			)
            # realizar la prediccion

            nbr = clf.predict(roi.reshape((1, 28, 28, 1)))
            #print np.argmax(nbr)
            cv2.putText(
                im,
                str(np.argmax(nbr)),
                (rect[0], rect[1]),
                cv2.FONT_HERSHEY_DUPLEX,
                2,
                (0, 255, 255),
                3
            )

    # mostramos el resutado
    cv2.imshow("imagen", im)
    tecla = cv2.waitKey(5) & 0xFF
    if tecla == 27:
        break
    #	cv2.waitKey()

cv2.destroyAllWindows()
