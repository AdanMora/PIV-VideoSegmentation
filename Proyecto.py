#Instituto Tecnologico de Costa Rica
#Escuela de computacion, Lenguajes de Programacion
#II Sem 2016
#Proyecto Numero 4
#Integrantes: Adrian Barboza Prendas y Adan Mora Fallas


import numpy as np
import cv2
import math
import tensorflow as tf
from xml.etree.ElementTree import *

def Controlador(): #Fue definida por que python no reconoce funciones despues de su llamado
    pass
def Vista(): #Fue definida por que python no reconoce funciones despues de su llamado
    pass
#"ResultadoProyecto.xml"
#"Dissolve1-15.mp4"

class Modelo:
    #Atributos
    Ctr = None
    V = None
    Video=""            #Dirrecion del video
    Histogramas= []     #Almacena los histogramas de las imagenes normalizados
    G= []               #Alamacena la dimisilitud entre cuadros consecutivos
    U= []               #Almacena el resultado de aplciar Tres Sigmas a G
    C= []               #Lista de frames donde se encontraron cortes

    #Metodos
    def __init__(self,C = Controlador(), V = Vista()):
        self.Ctr = C
        self.V = V
        
        
    def setVideo(self,Video):
        self.Video=Video
    def getVideo(self):
        return self.Video
        
    def EtapaVideo(self): 
        video = cv2.VideoCapture(self.Video) #Objeto que contiene el video abierto
        while (video.isOpened()):            #Si el video abrio bien entra
            ret,frame = video.read()         #Lee un frame a la vez
            if ret == False:                 #Si no leyo nada sale del ciclo
                break
            HSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) #Convertir de BGR a HSV
            H = HSV[:,:,0]          #Obteniendo la capa H q se encuentra en la dimension 0
            Max = H.max()           #Numpy Max
            Min = H.min()           #Numpy Min
            #Normalizando la capa H
            H = 255.0 * (H-float(Min))/(Max- Min)
            #Generando Histograma del frame
            hue = H
            with tf.Session() as sess:  #Inicio de sesion de tf
                size = tf.constant(float(H.size),dtype=tf.float32) #TF tamano del Histograma
                nbins = tf.constant(255)                         #nbins que solicita el generador de histogramas de tf
                value_range = tf.constant([0.0,255.0])           #Rango del histograma 
                value = tf.convert_to_tensor( hue, dtype = tf.float32 ) #Convirtiendo el arreglo de numpy en un tensor
                hist = tf.histogram_fixed_width(value,value_range, nbins) #Genera histograma
                normalize = tf.constant(hist.eval(),dtype = tf.float32)   #Convierte los valores a float
                result = tf.div(normalize, size)                          #Normaliza el histograma de con valores entre 0 y 1
                self.Histogramas.append(result.eval())                    #Guardamos histograma en la lista respectiva
                sess.close()                #Cierre de sesion
            del HSV,H,Max,Min,hue,size,nbins,value_range,value,hist,normalize,result
        video.release()         #Libera el video
                
    def CalcularArregloG(self):          #Calculo del arreglo de disimilitud
        for h in range(len(self.Histogramas)-1):
            self.G.append(self.DistanciaBhattacharyya(self.Histogramas[h],self.Histogramas[h+1]))

    def DistanciaBhattacharyya(self,h1,h2,M=255): #Compara dos histogramas 
        with tf.Session() as sess:   #Inicio de sesion de tf
            m1 = tf.segment_mean(h1, tf.zeros([M,], tf.int32))  #Valor medio de h1
            m2 = tf.segment_mean(h2, tf.zeros([M,], tf.int32))  #Valor medio de h2
            M=tf.pow(tf.convert_to_tensor(M,dtype= np.float32),tf.constant(2.0)) #M al cuadrado en tensor para el calculo de alfa
            alfa= tf.div(tf.constant(1.0),tf.sqrt(tf.mul(tf.mul(m1,m2),M))) #Calculo de alfa
            beta= tf.segment_sum(tf.sqrt(tf.mul(tf.convert_to_tensor(h1,dtype= np.float32),tf.convert_to_tensor(h2,dtype= np.float32))),tf.zeros([255,], tf.int32)) #Calculo de beta
            res= tf.sqrt(tf.sub(tf.constant(1.0),tf.mul(alfa,beta))).eval()[0]  #Calculo de la distancia de Battacharyya
            sess.close()    #Cierre de sesion
            del m1,m2,M,alfa,beta
        return res
    
    def Tres_Sigmas(self):
        with tf.Session() as sess:
            med= tf.segment_mean(tf.convert_to_tensor(self.G,dtype= np.float32), tf.zeros([len(self.G),], tf.int32)).eval() #Calculo de la mediana
            sess.close()
        desv= np.std(self.G)    #Calculo de la desviacion estandar gracias a Numpy
        #Implementacionde tres sigmas
        Ab1 = abs(med-desv)
        Ab2 = abs(med+desv)
        del med
        #Guardando resultado en U
        for t in self.G:
            if t >= Ab1 and t>= Ab2:
                self.U.append(1)
            else:
                self.U.append(0)
    def Cortes(self):
        for o in range(len(self.U)):
            if (self.U[o] == 1):
                self.C.append((o+1,o+2))
    def GeneraXML(self):
        nbr = ""
        for j in range(len(self.Video)):
            if self.Video[j] != ".":
                nbr += self.Video[j]
            else:
                break
        top = Element('Cortes')
        top.text = 'Estos son los cortes encontrados en el video'
        for i in range(len(self.C)):
            corte = SubElement(top, 'Corte' + str(i + 1))
            corte.text = 'Se ha encontrado un corte entre el Frame ' + str(self.C[i][0]) + ' y el Frame ' + str(self.C[i][1])
        file = open(nbr+".xml", "wb")
        file.write(tostring(top))
        file.close()

    def AnalizarVideo(self):
        print("Procesando video y generando histogramas ...")
        self.EtapaVideo()
        print("Calculando el arreglo G ... ")
        self.CalcularArregloG()
        self.Tres_Sigmas()
        print("Calculando cortes ...")
        self.Cortes()
        self.GeneraXML()
        self.V.MensajeSucess()
        nbr = ""
        for j in range(len(self.Video)):
            if self.Video[j] != ".":
                nbr += self.Video[j]
            else:
                break
        
class Vista:
    #Atributos
    C = None
    path = ""

    #Metodos
    def __init__ (self, C = Controlador()):
        self.C = C

    def GetPath(self):
        return self.path
        
    def Menu(self):
        print("**************** Bienvenido ****************")
        print("Este programa calcula los cortes de un video mediante el uso de la segmentación automática.")
        print("Creado por: Adrián Barboza y Adán Mora")

    def SetPath(self):
        self.path = input("\nIngrese el path del video:\n")
        
    def ErrorPath(self):
        print("\nEl video no existe, posiblemente el path es incorrecto, intente nuevamente.")
        
    def MensajeSucess(self):
        print("\nAnálisis del video realizado con éxito... Revise la carpeta del código fuente para ver el XML resultante.")
        
class Controlador:
    #Atributos
    M = Modelo()
    V = Vista()

    #Metodos
    def __init__ (self):
        self.V = Vista(self)
        self.M = Modelo(self,self.V)
        self.IniciarPrograma()


    def IniciarPrograma(self):
        self.V.Menu()
        self.V.SetPath()
        while not self.ValidarPath():
            self.V.SetPath()
        self.M.setVideo(self.V.GetPath())
        self.M.AnalizarVideo()

    def ValidarPath(self):
        try:
            path = self.V.GetPath()
            arch = cv2.VideoCapture(path)
            arch.release()
            return True
        except:
            self.V.ErrorPath()
        return False
        
    

#Main
        
contr= Controlador()
