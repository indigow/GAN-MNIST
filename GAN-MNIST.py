import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
IMAGE_SIZE = 28*28
M = 3000 #number to true images and fake images to use per iteration in the loss function
K = 7  # number of intern iteration for optimizing the descriminator loss per iteration
EPOCHS = 600
tf.enable_eager_execution()

mnist = keras.datasets.mnist
(x_train,y_train) , (x_test,y_test) = mnist.load_data()
x_train=x_train[np.logical_or(y_train==6, y_train==8)]
x_train = x_train/255

generator = keras.Sequential([   #the generator network
     keras.layers.Dense(30,activation=tf.nn.leaky_relu,input_shape=[IMAGE_SIZE,]),
     keras.layers.Dense(IMAGE_SIZE,activation=tf.nn.sigmoid)
    ])

descriminator = keras.Sequential([  #the descriminator network
     keras.layers.Dense(30,activation=tf.nn.leaky_relu,input_shape=(IMAGE_SIZE,)),
     keras.layers.Dense(1,activation=tf.nn.sigmoid)
    ])

def descriminator_loss(descriminator,generator,true_input): #the loss that the descriminator have to optimize
    noise = tf.convert_to_tensor(np.random.uniform(0,1,IMAGE_SIZE*M).reshape(-1,IMAGE_SIZE),dtype=tf.float32)
    l1 = tf.log(1-descriminator(generator(noise)))
    l2 = tf.log(descriminator(tf.convert_to_tensor(true_input.reshape(M,-1),dtype=tf.float32)))
    return tf.reduce_mean(l1+l2)

def generator_loss(descriminator,generator):  #the loss that the generator have to optimize
    noise = tf.convert_to_tensor(np.random.uniform(0,1,IMAGE_SIZE*M).reshape(-1,IMAGE_SIZE),dtype=tf.float32)
    l= tf.log(descriminator(generator(noise)))
    return tf.reduce_mean(l)

def train(descriminator,generator,x_train):
   dloss,gloss ,dtrue,dfake = [],[],[],[]

   doptimizer = tf.train.GradientDescentOptimizer(0.01)
   goptimizer = tf.train.GradientDescentOptimizer(0.01)
   plt.figure()

   for j in range(EPOCHS):
    for i in range(K):
       choices = np.random.choice(range(len(x_train)),M) #selecting M images from the true ones
       with tf.GradientTape() as t:
           dl = descriminator_loss(descriminator,generator,x_train[choices])*(-1) #computing the loss
       g = t.gradient(dl,descriminator.trainable_variables)#computing the gradient of the loss
       doptimizer.apply_gradients(zip(g,descriminator.trainable_variables)) #applying the gradient
    dloss.append(dl*(-1))
    dtrue.append(tf.reduce_mean(descriminator(tf.convert_to_tensor(x_train[choices].reshape(-1,IMAGE_SIZE),dtype=tf.float32))))
    for i in range(1):
     with tf.GradientTape() as t:
        gl = generator_loss(descriminator,generator)*(-1)
     g = t.gradient(gl,generator.trainable_variables)
     goptimizer.apply_gradients(zip(g,generator.trainable_variables)) #same thing but for the generator
    gloss.append(gl*(-1))
    print(j)
    if j%20==0:
        image = tf.reshape(generator(tf.random.uniform((1,IMAGE_SIZE,),0,1))[0],(28,28)).numpy()
        plt.subplot(6,5,j//20+1)
        plt.imshow(image,cmap=plt.get_cmap('gray'))


   plt.figure()
   plt.subplot(3,1,1)
   plt.plot(range(len(dloss)),dloss,'b')
   plt.subplot(3,1,2)
   plt.plot(range(len(gloss)),gloss,'r')
   plt.subplot(3,1,3)
   plt.plot(range(len(dtrue)),dtrue,'g')
   plt.show()
   #print(descriminator(tf.convert_to_tensor(x_train[:M].reshape(M,-1),dtype=tf.float32)))


train(descriminator,generator,x_train)
