# -*- coding: utf-8 -*-
"""
Created on Sun Dec 11 21:17:57 2022

@author: Duarte
"""


# ISTO CARREGA O MODELO
from tensorflow import keras
model = keras.models.load_model(r'C:\Users\Duarte\Documents\MEGA\03. Vida Académica\03. Mestrado Ciencias Computadores\1 Ano\Semestre 1\Topicos Avancados Inteligencia Artificial\Submissoes\Trabalhos\Projeto\final_model.h5')


import cv2
import os
import tqdm
import numpy as np
# Ler a imagem


os.chdir(r'C:\Users\Duarte\Documents\MEGA\03. Vida Académica\03. Mestrado Ciencias Computadores\1 Ano\Semestre 1\Topicos Avancados Inteligencia Artificial\Submissoes\Trabalhos\Projeto\data\cats_and_dogs\train')


BATCH_SIZE = 256
real_labels = []
ims = []


count = 0
for index, file in tqdm.tqdm(list(enumerate(os.listdir()))):
    count +=1
    predict = 0 if 'cat' in file else 1
    real_labels.append(predict)
    ims.append(cv2.resize(cv2.imread('cat.0.jpg')/255, (224, 224)))
    
    if count == BATCH_SIZE:
        predict = np.round(model.predict(np.array(ims), verbose=0))
        if not (predict == np.array(real_labels)).all():
            break
        
        ims = []
        real_labels = []
        count = 0


for n in range(1, 44):
    pred(fils[i+n])
    shutil.copy(fils[i+n], r'C:\Users\Duarte\Documents\MEGA\03. Vida Académica\03. Mestrado Ciencias Computadores\1 Ano\Semestre 1\Topicos Avancados Inteligencia Artificial\Submissoes\Trabalhos\Projeto\outputs\Failed_Read\\' + fils[i+n])
    
    