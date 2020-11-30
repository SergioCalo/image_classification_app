# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 00:40:50 2020

@author: ASUS
"""

import numpy as np
import streamlit as st
import torch
from PIL import Image, ImageOps
from torchvision import transforms

def import_and_predict(image_data, model):

    transformar = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    size = (600,600)    
    image = ImageOps.fit(image_data, size, Image.ANTIALIAS)

    image = image.convert('RGB')
    image = transformar(image)
    image = image.unsqueeze(0)
    image = image.permute(0,1,2,3)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    output = model(image)
  
    return output

clases = ['Alytes obstetricans', 'Anguis fragilis', 'Chalcides striatus', 'Chioglossa lusitanica', 'Coronella austriaca', 'Coronella girondica', 'Discoglossus galganoi',
              'Emys orbicularis', 'Epidalea calamita', 'Hyla arborea', 'Lacerta schreiberi', 'Lacerta viridis', 'Lissotriton boscai', 'Natrix astreptophora', 'Natrix maura',
              'Pelobates cultripes', 'Pelophylax perezi', 'Podarcis bocagei', 'Podarcis hispanicus', 'Rana iberica', 'Salamandra salamandra', 'Timon lepidus', 
              'Triturus marmoratus', 'Vipera seoanei', 'Zamenis scalaris']
model = torch.load('ResNet50_corrubedo.h5', map_location=lambda storage, loc: storage)

st.write("""
         # Queres saber de que especie se trata?
         """
         )

st.write("Elixe a imaxe do organismo e descúbreo!")

file = st.file_uploader("Escolle aquí a imaxe", type=["jpg", "png"])
#
if file is None:
    st.text("")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    output = import_and_predict(image, model)
    sm = torch.nn.Softmax()
    probabilities = sm(output) 
    prob, prob2=torch.max(probabilities, 1)
    prob=100.*(prob.item())
    prob=round(prob, 2)
    res = torch.topk(probabilities, 4)
    if prob > 70:
      st.write(clases[int(res.indices[0][0])], ', probabilidade:', prob, '%')
      st.text("")
      st.text('Outras posibilidades:')
      st.text("")
      line1 = str(clases[int(res.indices[0][1])]) + ', probabilidade: ' + str(round(float(res.values[0][1]*100)),2) + '%'
      line2 = str(clases[int(res.indices[0][2])]) + ', probabilidade: ' +  str(round(float(res.values[0][2]*100)),2) + '%'
      line3 = str(clases[int(res.indices[0][3])]) + ', probabilidade: ' + str(round(float(res.values[0][3]*100)),2) + '%'
      st.text(line1)
      st.text(line2)
      st.text(line3)
      st.text("")
    else:
      st.text('Non estou seguro, poderías ensinarme outra foto do organismo?')
      st.text('consello: quizais sexa boa idea buscar un ángulo diferente')
