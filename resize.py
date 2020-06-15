import pandas as pd
import numpy as np
import os
from skimage import io, transform, color, img_as_ubyte
from torchvision import transforms, utils, datasets
import time
import sys

def store_resized(image_path, new_path, size=(200,200)):
    os.makedirs(new_path, exist_ok=True)
    #get sorteed list of files
    file_names = sorted(os.listdir(image_path))
    for i in range(27429,27434,1):
        image = io.imread(os.path.join(image_path,file_names[i]))
        if len(image.shape) != 3: continue
        if image.shape[2] == 4:
            image = color.rgba2rgb(image)
        
        image = transform.resize(image,size)
        image[np.where(image > 1)] = 1
        if i % 500 == 0: print(i, time.time()) 
        #io.imshow(image)
        io.imsave(os.path.join(new_path,file_names[i]),img_as_ubyte(image))
    print('Resizing finished.')

def main():
    store_resized("train_images","resized_train_images_250",size=(250,250))

if __name__ == '__main__':
    main()
