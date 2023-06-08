# -*- coding: utf-8 -*-
"""
Created on Tue Aug 24 16:28:55 2021

@author: koerber
"""



import pickle
import os 
import cv2
import glob
import matplotlib.pyplot as plt
import sys
import tensorflow as tf
from dl.DeepLearning import *

os.environ["CUDA_VISIBLE_DEVICES"]="0" # set GPU if multiple present

def main():


    '''
    This code is developed and run on at least two machines
    Here we setup the default paths for each machine.
    Eventually we must find a more general way to do this.
    '''

    host = os.getenv('HOSTNAME')
    print(host)

    if "dsigpu" in host:
        pklDir = '/home/dsi/aryeh/data/mia/pkl/'
        modelDir = '/home/dsi/aryeh/data/mia/trained_models/'
        modelname = "unetDensnet201_2023-01-24_08-17-37"
        trainingdata = '/home/dsi/aryeh/data/plants/unCropped/resized/'
        predictionFolder = '/home/dsi/aryeh/data/plants/Harvest8Orange5_7Oct17/'
    elif "pop" in host:
        pklDir = '/media/amw/TOSHIBA EXT/alerding/models/pkl/'
        modelDir = '/media/amw/TOSHIBA EXT/alerding/models/gpuModels/'
        modelname = "unetDensnet201_2023-01-24_08-17-37"
        trainingdata =  '/media/amw/TOSHIBA EXT/alerding/annotated/notCroppedLabels2/resized/'
        predictionFolder = '/media/amw/TOSHIBA EXT/alerding/Harvest 8 Orange 5,7 Oct 17.r/vertical/'
    else:
        pklDir = None
        modelDir = None 
        modelname = "unetDensnet201_2023-01-24_08-17-37"
        trainingdata = None
        predictionFolder = None

# the model directory will receive the trained model
    modelDir = input('Enter model output directory or default\n') or modelDir
    
    loadweights = None


# stem, pop, background and unlabeled
# We should have a single cell where all of the model and training parameters are set
    numclasses = 4

# trainingdata is a directory that holds the annotated images. The labels must be in trainingdata/Segmentation_labels/
# and they must be npz files.
    trainingdata = input("Path to folder containing training images, or use default\n") or trainingdata
    validationdata = None

    print("model: ", modelname)

    '''
    Creates the dl object. Default is UNet, resnet152, 100 epochs,
    scaleFactor is 0.5, and learning rate is 0.001.
    All the other parmeters are the defaults of the dl object.
    '''

    print(os.getcwd())


   
    def createDL( epochs=100, scaleFactor=0.5, learning_rate = 0.001):
        dl = DeepLearning()
        dl.epochs = epochs
        dl.ImageScaleFactor = scaleFactor
    #    dl.Mode = Segmentation()
        dl.learning_rate = learning_rate
        dl.Mode.architecture = 'UNet'
        dl.Mode.backbone = 'resnet152'
        return dl

    '''
    create the dl object and verify that it is initialized. 
    '''

#   get number of epochs for this run
    epochs = input('Enter number of epochs, or enter for default (50)\n') or '50'
    dl = createDL( int(epochs), scaleFactor=0.5, learning_rate = 0.002)

# pod, stem, background, and unlabeled (for deleaved soybean plants)
    numClasses =4

    print('init model')
    dl.initModel(numclasses)

    dl.Model = dl.Mode.getModel(numClasses, 3)

    print('load model')
    if loadweights is not None:
        dl.Model.load_weights(loadweights)

    print(dl.initialized)
    
    print('init model')
    
    from pprint import pprint
    pprint(vars(dl))
    dl.data.nchannels = 3
    pprint(vars(dl.data))
    
#    dl.initModel(numclasses)
    policy = tf.keras.mixed_precision.Policy('float32')
    print('after policy')
    
# In the version that reads dl from a pkl file, I use dl.data.NumChannels, and it is 3
# In this version, I found that dl.data.nchannels=1, and I cannot change it to 3 (raises an Atribure error)
# so I have to use dl.data.nchannels, which is set to 3
    print("dl.data.nchannels: ", dl.data.nchannels)
    dl.Model = dl.Mode.getModel(4, dl.data.nchannels)
    print('load model')
    if loadweights is not None:
        dl.Model.load_weights(loadweights)
        
    if dl.initialized:
        print('start training')
        dl.initData_StartTraining(trainingdata, validationdata)
        print('training finished')
    else:
        print('could not initialize model')
            
    
    print('saving weights')
    dl.Model.save_weights(modelDir+modelname + '.h5')
    print('saving training data')
    dl.saveTrainingRecord(modelDir+modelname +'.csv')
#   print('saving model')
#   dl.Model.save(modelDir+'full')
    
    files = glob.glob(trainingdata + '*.jpg')
    count=0
    for i in files:
        img = cv2.imread(i)
        prediction = dl.Mode.PredictImage(img)

        print(type(prediction))
        plt.imshow(prediction)
        plt.show()
        count += 1
        if count>3:
            break
if __name__ == "__main__":
    main()
