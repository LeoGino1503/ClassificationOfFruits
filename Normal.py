import os
from glob import glob
from PIL import Image
import numpy as np
import pandas as pd
import csv

features = []
Type = ['Eggplant','Ginger','Onion','Potato','Tomato']
for i in range(len(Type)):
  path_input = './TrainData/' + Type[i] + '/'
  # print(path_input) 
  for file in glob(path_input + "*.jpg"):
    print(file)
    image = Image.open(file)
    imgGray = image.convert('L')
    image = imgGray.resize((32, 32))
    image_sequence = image.getdata()
    image_array = []
    image_array = np.array(image_sequence)
    for i in range(len(image_array)):
      image_array[i] = 255 - image_array[i]
    print(image_array.shape)
    features.append(image_array)
    print(len(features))
    # break

with open('Data_train.csv', 'w', newline='') as file:
  mywriter = csv.writer(file, delimiter=',')
  mywriter.writerows(features)
# print(features.shape)
# pd.DataFrame(image_array).to_csv('TestData.csv')

