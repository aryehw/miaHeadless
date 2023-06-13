# -*- coding: utf-8 -*-
"""
Created on Tue Aug 24 16:28:55 2021

@author: koerber

Modifed by Aryeh Weiss

Last modified: 08 June 2023
"""



import pickle
import os 
import cv2
import glob
import matplotlib.pyplot as plt
import sys
import tensorflow as tf

os.environ["CUDA_VISIBLE_DEVICES"]="1" # set GPU if multiple present


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

# the pkl directory contains the pkl files that hold a DeepLearning structure that defines the DL model
# If the dl object is directly created then the pkl fileis not needed.
    pklDir = input('Enter pkl path or use default\n') or pklDir

# the model directory will receive the trained model
    modelDir = input('Enter model output directory or default\n') or modelDir
    
    loadweights = None

# modelname is the name of the pkl file without its pkl  extension.
# a pkl file holds a data structure or object. IN this case, it is expected to hold a DeepLearning object that was saved
# using File>Save DL Object... in mianalyzer. 
# If the dl object is directly created then this is not needed.
    modelname = input("Enter modelname without extension or use default\n") or modelname

# stem, pop, background and unlabeled
# We should have a single cell where aall of the model and training parameters are set
    numclasses = 4

# trainingdata is a directory that holds the annotated images. The labels must be in trainingdata/Segmentation_labels/
# and they must be npz files.
    trainingdata = input("Path to folder containing training images, or use default\n") or trainingdata
    validationdata = None

    print("model: ", modelname)


        
    print('load settings')
#    filehandler = open(modelname + '.pkl', 'rb')
    filehandler = open(pklDir + modelname + '.pkl', 'rb')

    dl = pickle.load(filehandler)

    print('loaded pkl')
#    sys.exit()

#   get number of epochs for this run
    epochs = input('Enter number of epochs, or enter for default (50)\n') or '50'
    dl.epochs = int(epochs)
    
    print('init model')
    
    from pprint import pprint
    pprint(vars(dl))
    dl.data.nchannels = 3
    pprint(vars(dl.data))
    
#    dl.initModel(numclasses)
    policy = tf.keras.mixed_precision.Policy('float32')
    print('after policy')
    print("dl.data.numChannels: ", dl.data.numChannels)
    dl.Model = dl.Mode.getModel(4, dl.data.numChannels)
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
