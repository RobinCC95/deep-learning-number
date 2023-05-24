
import cv2
from Prediccion import  Prediccion

clases=["numero 0","numero 1","numero 2","numero 3","numero 4","numero 5","numero 6","numero 7","numero 8","numero 9"]

ancho=28
alto=28

miModeloCNN=Prediccion("models/modeloA.h5",ancho,alto)
imagen=cv2.imread("dataset/test/3/3_3.jpg")

claseResultado=miModeloCNN.predecir(imagen)
print("La imagen cargada es ",clases[claseResultado])

while True:
    cv2.imshow("imagen",imagen)
    k=cv2.waitKey(30) & 0xff
    if k==27:
        break
cv2.destroyAllWindows()
