import torchvision.transforms as transforms
import numpy as np
import torch
import os
import cv2
import time


class DataLoader(object):
    def __init__(self, image_dir,map_dir,batch_size=10):
        self.image_dir = image_dir
        self.batch_size = batch_size
        self.map_dir = map_dir
        self.names = os.listdir(self.image_dir)
        self.size = len(self.names)
        self.cursor = 0
        self.num_batches = int(self.size / self.batch_size)

    def get_batch(self): # Returns batches
        if self.cursor + self.batch_size > self.size:
            self.cursor = 0
            np.random.shuffle(self.names)
            
        img = torch.zeros(self.batch_size, 3, 192, 256)
        smap = torch.zeros(self.batch_size, 1, 192, 256)

        convert_to_tensor = transforms.ToTensor()

        for idx in range(self.batch_size):
            current_tensor = self.names[self.cursor]
            im_path = self.image_dir + current_tensor
            smap_path = self.map_dir + current_tensor
            self.cursor += 1

            image = cv2.imread(im_path)
            img[idx] = convert_to_tensor(image)

            salmap = np.expand_dims(cv2.imread(smap_path,0), axis=2)
            smap[idx] = convert_to_tensor(salmap)
            
        return (img, smap)


        