#Script en el que se implementa el descriptor SIFT manualmente

#Se construye el espacio de escalas, ademas de que será posible visualizar las imagenes
#Se muestra el paso de la diferencia de gaussianas
#Se muestra la identificación de los puntos clave
#Se muestra el histograma de orientaciones




import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import time
import numpy as np
import heapq
from scipy.spatial import cKDTree


sigma1 = [0.7071, 1, 1.4142, 2, 2.8284]
sigma2 = [1.4142, 2, 2.8284, 4, 5.6569]
sigma3 = [2.8284, 4, 5.6569, 8, 11.3137]
sigma4 = [5.6569, 8, 11.3137, 16, 22.6274]

sigmas = [sigma1, sigma2, sigma3, sigma4]

class SIFT:

    #Constructor de la clase contiene los parametros de la imagen, el numero de octavas y el numero de escalas, sigma y el numero de bins, tendra ademasn un atrubuto para el descriptor final
    def __init__(self, img, octaves, sigma, bins):
        self.img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        self.octaves = octaves
        self.scales = 5
        self.sigma = sigma
        self.gausssiansPyramid = None
        self.dogPyramid = None
        self.keyPoints = None
        self.bins = bins
        self.gradients = None
        self.orientation_matrix = None
        self.gradients_matrix = None
        self.descriptor = None
        self.matches= []

        
    '''#En la siguiente función se construye el espacio de escalas, se especifica el numero de octavas y el numero de escalas por octava

    e choose to divide each octave of scale space (i.e., doubling of σ) into an integer number, s, of intervals, so k = 21/s. We must produce s + 3 images in the stack of blurred images for each octave, so that final extrema detection covers a complete octave
    '''

    def calcular_octava(self, imagen, sigma_n):

        gaussianas = []
        diferencia_gaussianas = []
        for i in range(len(sigma_n)):
            gaussianas.append(cv2.GaussianBlur(imagen, (0,0), sigma_n[i]))
            if i > 0:
                diferencia_gaussianas.append(gaussianas[i] - gaussianas[i-1])
        
        return gaussianas, diferencia_gaussianas
    
    def construir_piramides(self, escalas, octavas):
        pyramid = [self.img]
        #Cuatro octavas, es decir 4 escalas con 5 sigmas cada una
        primeraEscala = self.img[::2,::2]
        pyramid.append(primeraEscala)

        for i in range(octavas - 2):
            aux = pyramid[i][::2,::2]
            pyramid.append(aux)
        
        self.gausssiansPyramid = []
        self.dogPyramid = []
        
        for i in range(len(pyramid)):
            gaussianas, diferencia_gaussianas = self.calcular_octava(pyramid[i], sigmas[i])
            self.gausssiansPyramid.append(gaussianas)
            self.dogPyramid.append(diferencia_gaussianas)

    def showPyramids(self):
        #Se visualiza la piramide de gaussianas
        plt.figure()
        for i in range(self.octaves):
            for j in range(self.scales):
                plt.subplot(self.octaves, self.scales, i*self.scales + j + 1)
                plt.imshow(self.gausssiansPyramid[i][j], cmap='gray')
        plt.show()

        #Se visualiza la piramide de diferencias de gaussianas
        plt.figure()
        for i in range(self.octaves):
            for j in range(self.scales - 1):
                plt.subplot(self.octaves, self.scales - 1, i*(self.scales - 1) + j + 1)
                plt.imshow(self.dogPyramid[i][j], cmap='gray')

        plt.show()
        
    def getNeighbors(self, octava, l, x, y):
        img = self.dogPyramid[octava][l]

        vecino = [img[x-1,y-1], img[x-1,y], img[x-1,y+1], img[x,y-1], img[x,y+1], img[x+1,y-1], img[x+1,y], img[x+1,y+1]]

        if l != 0:
            anterior = self.dogPyramid[octava][l-1]
            vecino += [ anterior[x, y], anterior[x-1,y-1], anterior[x-1,y], anterior[x-1,y+1], anterior[x,y-1], anterior[x,y+1], anterior[x+1,y-1], anterior[x+1,y], anterior[x+1,y+1]]
        
        if l != len(self.dogPyramid[octava]) - 1:
            siguiente = self.dogPyramid[octava][l+1]
            vecino += [ siguiente[x, y], siguiente[x-1,y-1], siguiente[x-1,y], siguiente[x-1,y+1], siguiente[x,y-1], siguiente[x,y+1], siguiente[x+1,y-1], siguiente[x+1,y], siguiente[x+1,y+1]]

        return vecino
    
    def detectar_extremos(self):
        extremos = []
        #Se recorren las escalas
        for i in range(self.octaves):
            #Se recorren las imagenes de cada escala
            for j in range(len(self.dogPyramid[i])):
                #Se recorren las filas
                for x in range(1, self.dogPyramid[i][j].shape[0] - 1):
                    #Se recorren las columnas
                    for y in range(1, self.dogPyramid[i][j].shape[1] - 1):
                        #Se obtiene el valor del pixel
                        pixel = self.dogPyramid[i][j][x,y]
                        #Se obtienen los vecinos del pixel
                        vecinos = self.getNeighbors(i, j, x, y)
                        minimo = True
                        maximo = True

                        for vecino in vecinos:
                            if pixel >= vecino:
                                maximo = False
                            if pixel <= vecino:
                                minimo = False
                        
                        if maximo or minimo:
                            extremos.append((i,j,x,y))
        self.keyPoints = extremos
                            
        
    def showKeyPoints(self):

        print("Numero de keypoints: ", len(self.keyPoints))
              
        plt.figure(figsize=(10,10))
        for i in range(self.octaves):
            for j in range(self.scales - 1):
                plt.subplot(self.octaves, self.scales - 1, i*(self.scales - 1) + j + 1)
                plt.imshow(self.dogPyramid[i][j], cmap='gray')
                for k in range(len(self.keyPoints)):
                    if self.keyPoints[k][0] == i and self.keyPoints[k][1] == j:
                        plt.scatter(self.keyPoints[k][3], self.keyPoints[k][2], c='r', s=5)
        plt.show()


    #Se define la función gaussian
    def gaussian(self, x, y, sigma):
        return (1/(2*math.pi*sigma**2))*math.exp(-(x**2 + y**2)/(2*sigma**2))
    
    #Se define la función para calcular la convolución
    def corr(self, img, kernel):
        img = np.array(img)
        kernel = np.array(kernel)
        img_x = img.shape[0]
        img_y = img.shape[1]
        kernel_x = kernel.shape[0]
        kernel_y = kernel.shape[1]

        new_img = np.zeros((img_x, img_y))

        for i in range(img_x):
            for j in range(img_y):
                for k in range(kernel_x):
                    for l in range(kernel_y):
                        if i + k - kernel_x//2 >= 0 and i + k - kernel_x//2 < img_x and j + l - kernel_y//2 >= 0 and j + l - kernel_y//2 < img_y:
                            new_img[i,j] += img[i + k - kernel_x//2, j + l - kernel_y//2]*kernel[k,l]
        return new_img
    

    def calcular_magnitud_gradiente(self):
        

        ix = [[-1], [0], [1]]
        iy = [[-1, 0, 1]]
        #OR puede ser reemplazado por orientation_matrix
        orientaion_matrix = []

        #GM puede ser reemplazado por gradients_matrix
        gradients_matrix = []

        
        #SE recorren las imagenes de la piramide de gaussianas
        for i in range(self.octaves):
            matriz_octava_orientaciones = []
            matriz_octava_gradientes = []

            #Se recorren las escalas de cada octava
            for j in range(self.scales):
                #Se obtiene la imagen
                #print("i: ", i)
                #print("j: ", j)
                img = self.gausssiansPyramid[i][j]
                
                #Se hace la convolución con el kernel de sobel
                img_y = self.corr(img, iy)
                img_y = np.array(img_y, dtype=np.int32)
                #Se calcula la derivada en x
                img_x = self.corr(img, ix)
                img_x = np.array(img_x, dtype=np.int32)

                
                auxMat1 = np.zeros((img_x.shape[0], img_x.shape[1]))
                auxMat2 = np.zeros((img_x.shape[0], img_x.shape[1]))
                for a in range(img_x.shape[0]):
                    for b in range(img_x.shape[1]):
                        #Se guarda el valor del gradiente por cada pixel
                        auxMat1[a, b] = math.sqrt(img_x[a, b]**2 + img_y[a, b]**2)
                        #Se guarda la orientación del gradiente por cada pixel
                        auxMat2[a, b] = math.atan2(img_y[a, b], img_x[a, b])
                
                matriz_octava_gradientes.append(auxMat1)
                matriz_octava_orientaciones.append(auxMat2)
            
            gradients_matrix.append(matriz_octava_gradientes)
            orientaion_matrix.append(matriz_octava_orientaciones)
    

        #Se guardan las matrices de orientaciones y gradientes
        self.orientation_matrix = orientaion_matrix.copy()
        self.gradients_matrix = gradients_matrix.copy()
        






    #Se calcula el histograma de orientaciones
    def calcular_histogama_regiones(self):
        newKeyPoints = []# x , y son las coordenadas del punto clave
        finalHistogram = []     
        #Se recorren los puntos de interes
        for i in range(len(self.keyPoints)):
            oct_num, scale_num, x, y = self.keyPoints[i]
            r = self.orientation_matrix[oct_num][scale_num]
            m = self.gradients_matrix[oct_num][scale_num]

            #Se hace un padding de 4 pixeles
            r = np.pad(r, 4, 'constant', constant_values=0)
            m = np.pad(m, 4, 'constant', constant_values=0)

            #r_w es la matriz de orientaciones de la ventana, m_w es la matriz de gradientes de la ventana la ventana es de -4 a 4
            
            r_w = r[(x+4)-4:(x+4)+4, (y+4)-4:(y+4)+4]
            m_w = m[(x+4)-4:(x+4)+4, (y+4)-4:(y+4)+4]


            #la funcion gaussian recibe como parametros x, y y sigma
            g = self.gaussian(x, y, sigmas[oct_num][scale_num])
            gm = g*m_w

            r_w = r_w /10
            r_w = r_w.astype(int)

            histogram = np.zeros(self.bins)
            for i in range(8):
                for j in range(8):
                    histogram[r_w[i,j]] += gm[i,j]

            max_orientation = np.argmax(histogram)
            new = [oct_num, scale_num, x, y, max_orientation]

            newKeyPoints.append(new)

            for h in histogram:
                if h > 0.8*max(histogram):
                    new = [oct_num, scale_num, x, y, h]
                    finalHistogram.append(new)
    

        self.keyPoints = newKeyPoints
        self.descriptor = finalHistogram

    def showKeypointVectors(self):
        plt.figure(figsize=(10, 10))

        for i in range(self.octaves):
            for j in range(self.scales):
                plt.subplot(self.octaves, self.scales, i * self.scales + j + 1)
                plt.imshow(self.gausssiansPyramid[i][j], cmap='gray')

                for k in range(len(self.keyPoints)):
                    if self.keyPoints[k][0] == i and self.keyPoints[k][1] == j:
                        plt.scatter(self.keyPoints[k][3], self.keyPoints[k][2], c='r', s=5)
                        plt.arrow(
                            self.keyPoints[k][3],
                            self.keyPoints[k][2],
                            10000 * math.cos(self.keyPoints[k][4]),
                            10000 * math.sin(self.keyPoints[k][4]),
                            color='r',
                        )

        plt.show()

    def showHistogram(self):
        num_keypoints = min(25, len(self.descriptor))
        num_rows = (num_keypoints // 5) + (1 if num_keypoints % 5 != 0 else 0)
        
        plt.figure(figsize=(15, 3 * num_rows))

        for i in range(num_keypoints):
            plt.subplot(num_rows, 5, i + 1)
            
            # Obtén el descriptor actual
            keypoint_descriptor = self.descriptor[i]

            # Barra horizontal para cada bin del histograma
            plt.bar(range(len(keypoint_descriptor)), keypoint_descriptor, color='blue', alpha=0.7)
            
            # Resalta el bin máximo
            max_bin_index = np.argmax(keypoint_descriptor)
            plt.bar(max_bin_index, keypoint_descriptor[max_bin_index], color='red', alpha=0.7)

            plt.title(f'Keypoint {i + 1}')
        
        plt.tight_layout()
        plt.show()

    
    def calcular_descriptor(self):
        descriptors = []

        for keypoint in self.keyPoints:
            oct_num, scale_num, x, y, orientation = keypoint
            descriptor = self.calcular_descriptor_punto(oct_num, scale_num, x, y, orientation)
            descriptors.append(descriptor)

        self.descriptor = descriptors

    def calcular_descriptor_punto(self, oct_num, scale_num, x, y, orientation):
        descriptor = []

        # Padding para la región
        padding = 8
        img = self.gausssiansPyramid[oct_num][scale_num]
        img = np.pad(img, padding, 'constant', constant_values=0)

        # Obtener la región de la imagen
        region = img[(x+padding)-8:(x+padding)+8, (y+padding)-8:(y+padding)+8]
        rotated_region = np.array(np.rot90(region, int(orientation / (np.pi / 4))))

        # Dividir la región en subregiones 4x4
        subregions = np.split(rotated_region, 4)
        for i in range(len(subregions)):
            subregions[i] = np.split(subregions[i], 4, axis=1)

        # Definir el tamaño del array del histograma de orientaciones
        num_bins = 8
        hist_array_size = 4

        # Inicializar un array para almacenar el histograma de orientaciones
        orientation_histogram = np.zeros((hist_array_size, hist_array_size, num_bins))
        # Calcular el índice del bin del histograma para cada píxel en la subregión
        for i in range(len(subregions)):
            for j in range(len(subregions[i])):
                for pixel in subregions[i][j]:
                    # Asegurarse de que pixel sea un escalar
                    scalar_pixel = np.mean(pixel)  # Aquí uso la media del array, puedes ajustarlo según tu lógica
                    bin_index = int(scalar_pixel * num_bins / 256)
                    orientation_histogram[i, j, bin_index] += 1


        # Aplanar el array del histograma de orientaciones para obtener el vector de características
        descriptor = orientation_histogram.flatten()

        # Normalizar el descriptor
        norm = np.linalg.norm(descriptor)
        descriptor = [val / norm for val in descriptor]

        return descriptor


    def mostrar_histograma(self, descriptor):
        num_bins = len(descriptor)
        bin_indices = range(1, num_bins + 1)

        plt.bar(bin_indices, descriptor, color='blue', alpha=0.7)
        plt.title('Histograma de Orientaciones')
        plt.xlabel('Bin del Histograma')
        plt.ylabel('Frecuencia')
        plt.show()
        plt.close()

    # def bbf_search(query_descriptor, num_neighbors=1, max_candidates=200, kdtree=cKDTree(np.array([[0,0],[0,0]]) )):
    #     heap = []  # Cola de prioridad para almacenar los candidatos
    #     candidates = 0

    #     # Realizar la búsqueda BBF
    #     heapq.heappush(heap, (0, 0))  # Insertar la raíz del k-d tree en la cola de prioridad

    #     while heap and candidates < max_candidates:
    #         _, node_idx = heapq.heappop(heap)  # Obtener el nodo más cercano
    #         node = kdtree.get_arrays()[0][node_idx]  # Obtener el punto del nodo

    #         # Calcular la distancia entre el punto del nodo y el descriptor de la consulta
    #         distance = np.linalg.norm(node - query_descriptor)

    #         # Almacenar el índice del nodo si es un candidato válido
    #         if distance < heapq.nlargest(num_neighbors, heap)[-1][0] or candidates < num_neighbors:
    #             heapq.heappush(heap, (distance, node_idx))
    #             candidates += 1

    #         # Explorar los hijos del nodo actual
    #         if len(kdtree.children[node_idx]) > 0:
    #             for child_idx in kdtree.children[node_idx]:
    #                 child_node = kdtree.get_arrays()[0][child_idx]
    #                 child_distance = np.linalg.norm(child_node - query_descriptor)
    #                 heapq.heappush(heap, (child_distance, child_idx))

    #     # Obtener los índices de los candidatos más cercanos
    #     nearest_neighbors = [heapq.heappop(heap)[1] for _ in range(candidates)]

    #     return nearest_neighbors


    # def matching_descriptores(self, sift_objeto1, sift_objeto2):
    #     # Construir k-d trees para los descriptores
    #     kdtree_objeto1 = cKDTree(sift_objeto1.descriptor)
    #     kdtree_objeto2 = cKDTree(sift_objeto2.descriptor)

    #     # Emparejar descriptores utilizando BBF
    #     matches = []
    #     for query_descriptor in sift_objeto1.descriptor:
    #         nearest_neighbors = self.bbf_search(query_descriptor, kdtree=kdtree_objeto2)
    #         best_match, second_best_match = nearest_neighbors[:2]

    #         # Condición para considerar un emparejamiento válido
    #         if best_match is not None and second_best_match is not None:
    #             ratio = np.linalg.norm(sift_objeto2.descriptor[best_match] - query_descriptor) / np.linalg.norm(sift_objeto2.descriptor[second_best_match] - query_descriptor)
    #             if ratio < 0.8:  # Condición de ratio (como se menciona en la sección citada)
    #                 matches.append((best_match, ratio))

    #     self.matches = matches

    def encontrar_mejor_match(self, descriptor, descriptoresObjeto2):
        #Se usa el metodo de fuerza bruta para encontrar el mejor match
        descriptor = np.array(descriptor)
        mejor_match = None

        for i in range(len(descriptoresObjeto2)):
            descriptor2 = np.array(descriptoresObjeto2[i])
            distancia = np.linalg.norm(descriptor - descriptor2)
            if mejor_match is None or distancia < mejor_match[1]:
                mejor_match = (i, distancia)
        
        return mejor_match
    

    def matching_descriptores(self, sift_objeto1, sift_objeto2):
        #Encuentra k matches a dibujar
        k = 1000

        for i in range(k):

            #Se hace una permutación de los descriptores
            np.random.shuffle(sift_objeto1.descriptor)
            np.random.shuffle(sift_objeto2.descriptor)

            #Se toma el primer descriptor de la permutación
            descriptor = sift_objeto1.descriptor[0]

            #Se encuentra el mejor match para el descriptor
            mejor_match = self.encontrar_mejor_match(descriptor, sift_objeto2.descriptor)

            #Se agrega el mejor match a la lista de matches
            self.matches.append(mejor_match)

            k -= 1

            if k == 0:
                break

    
    def showMatches(self, img1, img2):
        # Ordenar los emparejamientos por el ratio
        self.matches.sort(key=lambda x: x[1])

        # Obtener los índices de los descriptores emparejados
        matches_idx = [match[0] for match in self.matches]

        # Obtener los puntos clave emparejados
        keypoints1 = np.array([self.keyPoints[i][2:4] for i in range(len(self.keyPoints))])
        keypoints2 = np.array([self.keyPoints[i][2:4] for i in range(len(self.keyPoints))])

        # Obtener los puntos clave emparejados
        matched_keypoints1 = keypoints1[matches_idx]
        matched_keypoints2 = keypoints2[matches_idx]

        #Se concatenan las imagenes y 
        img_concat = np.zeros((max(img1.shape[0], img2.shape[0]), max(img1.shape[1], img2.shape[1]) + min(img1.shape[1], img2.shape[1])))
        img_concat = img_concat.astype(np.uint8)
        

        #Se coloca la imagen 1 en la imagen concatenada
        for i in range(img1.shape[0]):
            for j in range(img1.shape[1]):
                img_concat[i,j] = img1[i,j]
        
        #Se coloca la imagen 2 en la imagen concatenada

        for i in range(img2.shape[0]):
            for j in range(img2.shape[1]):
                img_concat[i,j+img1.shape[1]] = img2[i,j]

        
        #Se crea el plot de los matches
        plt.figure(figsize=(20, 20))
        plt.imshow(img_concat, cmap='gray')

        #Se dibuja una linea entre los puntos clave emparejados
        for i in range(len(matched_keypoints1)):
            plt.plot(
                [matched_keypoints1[i][1], matched_keypoints2[i][1] + img1.shape[1]],
                [matched_keypoints1[i][0], matched_keypoints2[i][0]],
                color='r',
                linewidth=0.5,
            )

        plt.show()

        
        
        

    

    



        






