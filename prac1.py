#Import all the necessary libraries
import os
import numpy as np
import pandas as pd
from scipy.misc import imread
import pylab
from sklearn.metrics import accuracy_score

import tensorflow as tf
import keras

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Convolution2D, Flatten, MaxPooling2D, Reshape, InputLayer
from keras.models import load_model

#Let’s set a seed value, so that we can control our models randomness
seed=128
rng=np.random.RandomState(seed)

#define vars
input_num_units=784
hidden_num_units1=100
hidden_num_units2=100
hidden_num_units3=100
hidden_num_units4=100
hidden_num_units5=100

output_num_units=10

epochs=30
batch_size=128


#The first step is to set directory paths, for safekeeping!
root_dir = os.path.abspath('../..')
data_dir = os.path.join(root_dir, 'data')
sub_dir = os.path.join(root_dir, 'sub')

#Data Loading and Preprocessing
def preprocess_data():
    train = pd.read_csv(os.path.join(data_dir, 'train.csv'))
    test = pd.read_csv(os.path.join(data_dir, 'Test.csv'))
    sample_submission = pd.read_csv(os.path.join(data_dir, 'Sample_Submission.csv'))
    return train,test,sample_submission

#Let us see what our data looks like! We read our image and display it.
def viewimage(train,dir_name):
    img_name = rng.choice(train.filename)
    filepath = os.path.join(data_dir,dir_name, img_name)
    img = imread(filepath, flatten=True)
    pylab.imshow(img, cmap='gray')
    pylab.axis('off')
    pylab.show()

#For easier data manipulation, let’s store all our images as numpy arrays
def imgtoarray(data,dir_name):
    print("image to array")
    temp = []
    for img_name in data.filename:
        image_path = os.path.join(data_dir,dir_name, img_name)
        img = imread(image_path, flatten=True)
        img = img.astype('float32')
        temp.append(img)
    train_x = np.stack(temp)
    train_x /= 255.0
    train_x = train_x.reshape(-1, 784).astype('float32')
    if(dir_name=='train'):
        train_y = keras.utils.np_utils.to_categorical(train.label.values)
        return train_x,train_y
    else:
        return  train_x

#As this is a typical ML problem, to test the proper functioning of our
# model we create a validation set.
# Let’s take a split size of 70:30 for train set vs validation set
def datasplit(train_x,train_y,train):
    print("data split")
    split_size=int(train_x.shape[0]*0.7)
    train_x,val_x=train_x[:split_size],train_x[split_size:]
    train_y,val_y=train_y[:split_size],train_y[split_size:]
    train.label.ix[split_size:]
    return train_x,val_x,train_y,val_y

def buildmodel(train_x,train_y,val_x,val_y,logger):
    print("#Model Building")
    model=Sequential()
    model.add(Dense(hidden_num_units1,input_dim=input_num_units,activation='relu'))
    #model.add(Dense(hidden_num_units2,activation='relu'))
    #model.add(Dense(hidden_num_units3,activation='relu'))
    #model.add(Dense(hidden_num_units4,activation='relu'))
    #model.add(Dense(hidden_num_units5,activation='relu'))
    model.add(Dense(output_num_units,activation='softmax'))
    print('compile the model with necessary attributes')
    ## compile the model with necessary attributes
    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    print('Its time to train our model')
    #It’s time to train our model
    trained_model=model.fit(train_x,train_y,epochs=epochs,batch_size=batch_size,validation_data=(val_x,val_y),callbacks=[logger])
    return model

def modeleval(model,test_x):
    print('Model Evaluation')
    #Model Evaluation
    #To test our model with our own eyes, let’s visualize its predictions
    pred=model.predict_classes(test_x)
    return pred

def showpred(test,train,pred):
    print('showpred')
    img_name=rng.choice(test.filename)
    filepath=os.path.join(data_dir,'test',img_name)
    img=imread(filepath,flatten=True)
    test_index=int(img_name.split('.')[0]) - train.shape[0]
    print('\nprediction is:', pred[test_index])
    pylab.imshow(img,cmap='gray')
    pylab.axis('off')
    pylab.show()

def save_model(model,name):
	# Save the model to disk
	model.save(name)
	print("Model saved to disk.")

def model_load(model,name):
	model = load_model(name)
	return model

def tensorboard_logger(RUN_NAME,histogram_freq=5,write_graph=True):
	# Create a TensorBoard logger
	logger = keras.callbacks.TensorBoard(
	    log_dir='logs/{}'.format(RUN_NAME),
	    histogram_freq=histogram_freq,
	    write_graph=write_graph )
	return logger


train,test,sample_submission=preprocess_data()

viewimage(train,dir_name='train')

train_x,train_y=imgtoarray(train,dir_name='train')

test_x=imgtoarray(test,dir_name='test')

train_x,val_x,train_y,val_y=datasplit(train_x,train_y,train)

logger=tensorboard_logger(RUN_NAME='model with one layer and epoch{}'.format(epochs))

model=buildmodel(train_x,train_y,val_x,val_y,logger)

save_model(model,'prac2.h5')

pred=modeleval(model,test_x)

showpred(test,train,pred)


#you can visualize the result on tesorboard by using the command on terminal as tensorboard --logdir=logs
#feel free to experiment with the model

