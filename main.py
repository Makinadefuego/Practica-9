from Descriptor import sift
import cv2
#Se cargan las imagenes en escala de grises
img1 = cv2.imread('images/ejemplo1_1.jpg')

#Se visualiza la imagen 1
cv2.imshow('img1', img1)
#Se crea un objeto de la clase SIFT
sift1 = sift.SIFT(img1, 4, 1, 36)

#Se construye el espacio de escalas
sift1.construir_piramides(4, sift1.octaves)
#sift1.showPyramids()
#Se encuentran los candidatos a keypoints
sift1.detectar_extremos()
sift1.calcular_magnitud_gradiente()
sift1.calcular_histogama_regiones()
sift1.calcular_descriptor()


#sift1.mostrar_histograma(sift1.descriptor[0])


#Se hace el mismo procesao para una segunda imagen
img2 = cv2.imread('images/ejemplo1_2.jpg')
cv2.imshow('img2', img2)
sift2 = sift.SIFT(img2, 4, 1, 36)
sift2.construir_piramides(4, sift2.octaves)
#sift2.showPyramids()
sift2.detectar_extremos()
sift2.calcular_magnitud_gradiente()
sift2.calcular_histogama_regiones()
sift2.calcular_descriptor()
#sift2.mostrar_histograma(sift2.descriptor[0])

sift1.matching_descriptores(sift1, sift2)
sift1.showMatches(sift1.img, sift2.img)
