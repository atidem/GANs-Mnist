# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 14:16:07 2019

@author: atidem
"""
from keras.layers import Dense,Dropout,Input,ReLU
from keras.models import Model,Sequential
from keras.optimizers import Adam
from keras.datasets import mnist
import numpy as np 
import matplotlib.pyplot as plt

#%% data preprocess

(xTrain,yTrain),(xTest,yTest) = mnist.load_data()
xTrain = (xTrain.astype(np.float32)-127.5)/127.5

xTrain.shape # 28,28 i 28*28 yapmak lazım

xTrain = xTrain.reshape(xTrain.shape[0],xTrain.shape[1]*xTrain.shape[2])
xTrain.shape 

#%% visualize 
plt.imshow(xTest[5])


#%% Generator      input-512-relu-512-relu-1024-relu-784-tanh-output

def createGenerator():
    
    generator = Sequential()
    generator.add(Dense(units=512,input_dim=100))
    generator.add(ReLU())
    
    generator.add(Dense(units=512))
    generator.add(ReLU())
    
    generator.add(Dense(units=1024))
    generator.add(ReLU())

    generator.add(Dense(units=784,activation="tanh"))

    generator.compile(loss="binary_crossentropy",
                      optimizer=Adam(lr=0.0001,beta_1=0.5))

    return generator

g = createGenerator()
g.summary()

#%% Discriminator   input-1024-relu-dropout-512-relu-dropout-256-relu-output-sigmoid

def createDiscriminator():
    
    discriminator = Sequential()
    discriminator.add(Dense(units=1024,input_dim=784))
    discriminator.add(ReLU())
    discriminator.add(Dropout(0.4))

    discriminator.add(Dense(units=512))
    discriminator.add(ReLU())
    discriminator.add(Dropout(0.4))
    
    discriminator.add(Dense(units=256,input_dim=100))
    discriminator.add(ReLU())

    discriminator.add(Dense(units=1,activation="sigmoid"))

    discriminator.compile(loss="binary_crossentropy" ,
                          optimizer=Adam(lr=0.0001,beta_1=0.5))
    
    return discriminator

d = createDiscriminator()
d.summary()

#%% Gan Model Creator   input-g-x-d-outpu

def createGan(dis,gen):
    
    dis.trainable =False
    gan_input = Input(shape=(100,))
    x = gen(gan_input)
    gan_output = dis(x)
    
    gan = Model(inputs=gan_input,outputs=gan_output)
    
    gan.compile(loss="binary_crossentropy",optimizer="adam")
    
    return gan



gan = createGan(d,g)
gan.summary()


#%% train 

epochs = 300
batchSize = 256

for e in range(epochs):
    for _ in range(batchSize):
        
        noise = np.random.normal(0,1,[batchSize,100])
        
        genImg = g.predict(noise)
        
        imgBatch = xTrain[np.random.randint(low=0,high=xTrain.shape[0],size=batchSize)]
        
        x = np.concatenate([imgBatch,genImg])
        
        yDis = np.zeros(batchSize*2)
        yDis[:batchSize] = 1
        
        d.trainable = True 
        d.train_on_batch(x,yDis)
        
        noise = np.random.normal(0,1,[batchSize,100])
        
        yGen = np.ones(batchSize)
        
        d.trainable = False
        
        gan.train_on_batch(noise,yGen)
    
    print("epp",e)

#%% save model
    
g.save_weights("gansGModel300.h5")

#%% visualize 

noise = np.random.normal(loc=0,scale=1,size=[100,100])
genImg = g.predict(noise)
genImg = genImg.reshape(100,28,28)
plt.imshow(genImg[10],interpolation="nearest")
plt.axis("off")
plt.show()
plt.figure()
plt.imshow(genImg[20],interpolation="nearest")
plt.axis("off")
plt.show()
plt.figure()
plt.imshow(genImg[40],interpolation="nearest")
plt.axis("off")
plt.show()
plt.figure()
plt.imshow(genImg[30],interpolation="nearest")
plt.axis("off")
plt.show()
plt.figure()
plt.imshow(genImg[60],interpolation="nearest")
plt.axis("off")
plt.show()
plt.figure()
plt.imshow(genImg[50],interpolation="nearest")
plt.axis("off")
plt.show()






