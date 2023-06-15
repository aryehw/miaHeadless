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

from pprint import pprint

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

# trainingdata is a directory that holds the annotated images. The labels must be in trainingdata/Segmentation_labels/
# and they must be npz files.
    trainingdata = input("Path to folder containing training images, or use default\n") or trainingdata
    validationdata = None
    
    BackBone = input("input backbone or enter for default\n") or 'densenet201'

    strEpochs = input("input number of epochs or enter to use the predefined value in a pkl file\n") or '0'
    epochs = int(strEpochs)



    print("model: ", modelname)
        
    print('load settings')
#    filehandler = open(modelname + '.pkl', 'rb')
    filehandler = open(pklDir + modelname + '.pkl', 'rb')

    dl = pickle.load(filehandler)

    print('loaded pkl')
#    sys.exit()


####################################
# Set up various training paramteres

# stem, pop, background and unlabeled
    numClasses = 4
    
# we can change the number of epochs and the backbone of the loaded dl object
    print(dl.epochs)
    if epochs != 0:
        dl.epochs = epochs
    if dl.Mode.backbone != BackBone:
        dl.Mode.backbone = BackBone
    pprint(vars(dl.Mode))
#####################################    
#####################################      
    '''
    This code saves the dl object as a pkl file. 
    It is saved in teh pkl directory, but mabe I should save it in the trained model directory.
    '''

    pklFileHandler = open(pklDir+modelname+'_ep'+str(dl.epochs) + '.pkl','wb')
    #import sys
    #print(sys.getrecursionlimit())
    #sys.setrecursionlimit(4*sys.getrecursionlimit())
    #print(sys.getrecursionlimit())

    hed = dl.hed
    dl.hed = None
    model = dl.Model
    dl.Model = None
    pickle.dump(dl, pklFileHandler)
    dl.hed = hed
    dl.Model = model

######################################     
        
    print('init model')
    
    pprint(vars(dl))
    dl.data.nchannels = 3
    pprint(vars(dl.data))
    
    dl.initModel(numClasses)
    
    dl.Model = dl.Mode.getModel(numClasses, 3)

    '''
    policy = tf.keras.mixed_precision.Policy('float32')
    print('after policy')
    print("dl.data.numChannels: ", dl.data.numChannels)
    dl.Model = dl.Mode.getModel(4, dl.data.numChannels)
    
    
    '''
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
    modelname=dl.Mode.architecture+'_'+dl.Mode.backbone
    dl.Model.save_weights(modelDir+modelname+'_ep'+str(dl.epochs) + '.h5')
    print('saving training data')
    dl.saveTrainingRecord(modelDir+modelname+'_ep'+str(dl.epochs) +'.csv')
#   print('saving model')
#   dl.Model.save(modelDir+'full')
    


# This code does a prediction on a directory of jpg images (could be any type, but that is what I have).
# predictionFolder contains the images which ill be predicted, and a subdirectory with the modelname will be created
# to hold the predicted segmentation.

    predictionFolder = input('Enter folder for prediction or use default\n')  or predictionFolder
    outputPath = predictionFolder + modelname+'_ep'+str(dl.epochs)+'/'
    try:
        os.mkdir(outputPath)
    except FileExistsError:
        pass
    files = glob.glob(predictionFolder + '*.jpg')
    print(len(files))
    
    
    count=0
    for i in files:
        img = cv2.imread(i)
        prediction = dl.Mode.PredictImage(img)

        print(type(prediction))
        plt.imshow(prediction)
        plt.show()
    # use the following code to limit the number of predicted files, when doing quick tests.
        count += 1
        if count>3:
            break
if __name__ == "__main__":
    main()
