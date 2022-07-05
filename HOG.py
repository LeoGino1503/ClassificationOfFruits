import os
from glob import glob
from PIL import Image
from skimage.feature import hog
import matplotlib.pyplot as plt
import csv


# reading the image
hog_images = []
hog_features = []
Type = ['Eggplant','Ginger','Onion','Potato','Tomato']
for i in range(len(Type)):
  path_input = './TestData/' + Type[i] + '/'

  # print(path_input) 
  for file in glob(path_input + "*.jpg"):
    print(file)
    images = Image.open(file)
    # images.show()
    width, height = images.size
    print(width, height)
    resized_img = images.resize((64, 128))
    widthre, heightre = resized_img.size
    # resized_img.show()
    print(widthre, heightre)

    fd, hog_image = hog(resized_img, orientations=9, pixels_per_cell=(8, 8),
                  cells_per_block=(2, 2), visualize=True, channel_axis=-1)

    print(fd)

    hog_images.append(hog_image)
    hog_features.append(fd)
    # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)

    # ax1.axis('off')
    # ax1.imshow(resized_img, cmap=plt.cm.gray)
    # ax1.set_title('Input image')

    # ax2.axis('off')
    # ax2.imshow(hog_image, cmap=plt.cm.gray)
    # ax2.set_title('Histogram of Oriented Gradients')
    # plt.show()

    # plt.axis("off")
    # plt.imshow(hog_image, cmap="gray")
  #   break
  # break

import pandas as pd
pd.DataFrame(hog_features).to_csv('TestData.csv')

# with open('TestData.csv', 'w', newline='') as file:
#   mywriter = csv.writer(file, delimiter=',')
#   mywriter.writerows(hog_features)
