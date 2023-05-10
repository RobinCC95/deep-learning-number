import tensorflow as tf
import keras
import numpy as np
import cv2
###Importar componentes de la red neuronal
from keras.models import Sequential
from keras.layers import InputLayer,Input,Conv2D, MaxPool2D,Reshape,Dense,Flatten
##################################


def cargarDatos(rutaOrigen,numeroCategorias,limite,ancho,alto):
    """Función para cargar las imágenes de entrenamiento y pruebas
    
    Args:
        rutaOrigen (_type_): _description_ 
        numeroCategorias (_type_): _description_
        limite (_type_): _description_
        ancho (_type_): _description_
        alto (_type_): _description_

    Returns:
        _type_: retorrna un arreglo con las imágenes y otro con las probabilidades
    """
    imagenesCargadas=[]
    valorEsperado=[]
    for categoria in range(0,numeroCategorias):
        for idImagen in range(0,limite[categoria]):
            ruta=rutaOrigen+str(categoria)+"/"+str(categoria)+"_"+str(idImagen)+".jpg"
            print(ruta)
            imagen = cv2.imread(ruta) # cargo imagen
            imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY) #convierto a escala de grises
            imagen = cv2.resize(imagen, (ancho, alto)) #redimensiono
            imagen = imagen.flatten() #aplanar de matriz a vector
            imagen = imagen / 255   #normalizar
            imagenesCargadas.append(imagen)
            probabilidades = np.zeros(numeroCategorias) #creo un vector de 10 posiciones que representa las 10 categorias
            probabilidades[categoria] = 1   #asigno la posicion de la categoria que corresponde cero = [1,0,0,0,0,0,0,0,0,0]
            valorEsperado.append(probabilidades)
    imagenesEntrenamiento = np.array(imagenesCargadas)
    valoresEsperados = np.array(valorEsperado)
    return imagenesEntrenamiento, valoresEsperados

#################################
ancho=28
alto=28
pixeles=ancho*alto
#Imagen RGB -->3
numeroCanales=1
formaImagen=(ancho,alto,numeroCanales)
#por ser 10 digitos o 10 clasificacinones
numeroCategorias=10

cantidaDatosEntrenamiento=[60,60,60,60,60,60,60,60,60,60]
cantidaDatosPruebas=[20,20,20,20,20,20,20,20,20,20]

#Cargar las imágenes
imagenes, probabilidades=cargarDatos("dataset/train/",numeroCategorias,cantidaDatosEntrenamiento,ancho,alto)

model=Sequential()
#Capa entrada
model.add(InputLayer(input_shape=(pixeles,)))   #capa de entrada de 784 neuronas
model.add(Reshape(formaImagen)) #convierto de nuevo a matriz

#Capas Ocultas
#Capas convolucionales

#kernerl_size=5,5 --> tamaño de la ventana o filtro
#strides=2,2 --> tamaño del paso
#padding="same" --> relleno al final de la imagen, same -> duplica las ultimas filas y columnas
model.add(Conv2D(kernel_size=5,strides=2,filters=16,padding="same",activation="relu",name="capa_1"))
#pool_size=2,2 --> tamaño de la ventana o filtro reducido que obtiene los datos mas relevantes de la imagen
model.add(MaxPool2D(pool_size=2,strides=2))

model.add(Conv2D(kernel_size=3,strides=1,filters=36,padding="same",activation="relu",name="capa_2"))
model.add(MaxPool2D(pool_size=2,strides=2))

#Aplanamiento
model.add(Flatten())
model.add(Dense(128,activation="relu"))

#Capa de salida
model.add(Dense(numeroCategorias,activation="softmax"))


#Traducir de keras a tensorflow
model.compile(optimizer="adam",loss="categorical_crossentropy", metrics=["accuracy"])
#Entrenamiento
#epochs=30 --> cantidad de iteraciones
#batch_size=60 --> cantidad de datos que se van a procesar en cada iteracion
model.fit(x=imagenes,y=probabilidades,epochs=30,batch_size=60)

# Todo: crear un callback que detenga el entrenamiento cuando se alcance un accuracy de 0.99 o 
#ya no se mejore el accuracy en las n epocas


#Prueba del modelo
imagenesPrueba,probabilidadesPrueba=cargarDatos("dataset/test/",numeroCategorias,cantidaDatosPruebas,ancho,alto)
resultados=model.evaluate(x=imagenesPrueba,y=probabilidadesPrueba)
print("Accuracy=",resultados[1])

# Guardar modelo
ruta="models/modeloA.h5"
model.save(ruta)
# Informe de estructura de la red
model.summary()
