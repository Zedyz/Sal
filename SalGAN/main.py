import torch
import cv2
import numpy as np
import torch.nn as nn
from tqdm import tqdm
import time

from dataset import DataLoader

from models import Discriminator, Generator
from utils import *
from paths import *

#Empty cache to save GPU Memory
torch.cuda.empty_cache()

#Set Parameters
epochs = 500
batch_size = 10
learning_rate = 0.0003
alpha = 1/20.

#Define device to do computations on
device = torch.device("cuda:0")

#Define models
discriminator = Discriminator().to(device)
generator = Generator().to(device)

#Define optimizers
dis_optimizer = torch.optim.Adagrad(discriminator.parameters(), lr=learning_rate)
gen_optimizer = torch.optim.Adagrad(generator.parameters(), lr=learning_rate)

#BCE Loss
L = nn.BCELoss()

#Load the data
dataset = DataLoader(map_dir = resizedMapsTrain,image_dir = resizedImagesTrain,batch_size = batch_size)

#Example predicted every epoch to see model progress
example = cv2.imread('dsnews.com.png')
example_test = 'example_test/results/'
saved_models = 'during_training/'

#Create paths for example and saving models
createPath(example_test)
createPath(saved_models)

#Start Time
st = time.time()

#Real and Fake Labels for Discriminator
labels_one = torch.ones(batch_size).to(device)
labels_zero = torch.zeros(batch_size).to(device)

#Train loop
for ep in tqdm(range(1,epochs+1)):
    for batch in range(1,dataset.num_batches+1):
        
        #Load data and send to device
        images,maps = dataset.get_batch()
        images = images.to(device)
        maps = maps.to(device)

        #Feed fake maps to Discriminator
        dis_fake_input = torch.cat((images, generator(images)),1)
        dis_fake_output = discriminator(dis_fake_input).squeeze()

        #Train Discriminator and Generator alternate in batches
        #Train Discriminator for even batch number
        if(batch % 2 == 0):

            #Reset gradients before backprop
            dis_optimizer.zero_grad()

            #Feed real maps to Discriminator
            dis_real_input = torch.cat((images,maps),1) #Ground Truths
            dis_real_output = discriminator(dis_real_input).squeeze()

            # D_loss = L(D(I,S),1) + L(D(I,Scap),0)
            dis_loss = L(dis_real_output,labels_one) + L(dis_fake_output, labels_zero)
            
            #Backpropagate the loss
            dis_loss.backward()
            dis_optimizer.step()

        #Train Generator for odd batch number
        else:
            #Train Generator

            #Reset Gradients before backprop
            gen_optimizer.zero_grad()

            #Content Loss
            L_bce = L(generator(images),maps)

            #G_loss = alpha*Lbce + L(D(I,Scap),1)
            gen_loss = alpha * L_bce + L(dis_fake_output, labels_one)


            #Backpropagate the loss
            gen_loss.backward()
            gen_optimizer.step()

        #Delete batch from memory
        del images
        del maps

        #Print Losses every 100 batches
        if (batch)%100 == 0:
            print("Epoch [%d/%d], Batch[%d/%d], d_loss: %.4f, g_loss: %.4f, time: %4.4f"
            % (ep, epochs, batch, dataset.num_batches, dis_loss.data, gen_loss.data,time.time()-st))

    #Save Model Weights after every epoch
    print('Epoch: ', ep, 'Complete')
    print('Saving model in ', saved_models)
    torch.save(generator.state_dict(), saved_models + 'generator_ep'+str(ep)+'.pkl')
    torch.save(discriminator.state_dict(), saved_models + 'discriminator_ep'+str(ep)+'.pkl')
    
    #Predict example map and save it
    predict(generator,example,ep,example_test,device)
    print(f"Saved Example for epoch {ep} in {example_test}")

#Save final model
torch.save(generator.state_dict(), saved_models + 'generatorfinal.pkl')
torch.save(discriminator.state_dict(), saved_models + 'discriminatorfinal.pkl')

print("Trained!")















