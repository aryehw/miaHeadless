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

os.environ["CUDA_VISIBLE_DEVICES"]="0" # set GPU if multiple present


def main():

    pklDir = input('Enter pkl path or use default\n') or '/media/amw/TOSHIBA EXT/alerding/models/pkl/'
    modelDir = input('Enter model output directory or default\n') or '/media/amw/TOSHIBA EXT/alerding/models/gpuModels'
    
    loadweights = None
    # modelname = 'model_2022-10-28_11-28-35' # saved object without extension
    modelname = input("Enter modelname without extension or use default\n") or "unetDensnet201_2023-01-24_08-17-37"
    numclasses = 4
    trainingdata = '/media/amw/TOSHIBA EXT/alerding/annotated/notCroppedLabels2/resized/'
    validationdata = None

        
    print('load settings')
#    filehandler = open(modelname + '.pkl', 'rb')
    filehandler = open(pklDir + modelname + '.pkl', 'rb')

    dl = pickle.load(filehandler)

    print('loaded pkl')
#    sys.exit()
    
    dl.epochs=10
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
    for i in files:
        img = cv2.imread(i)
        prediction = dl.Mode.PredictImage(img)

        print(type(prediction))
        plt.imshow(prediction)
        plt.show()
#       for count > 3:
#           break
if __name__ == "__main__":
    main()
