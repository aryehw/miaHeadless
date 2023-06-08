# miaHeadless
Code to use mianalyzer headless 

## trainHeadless_noPkl.ipynb 
A jupyter script that can either load a dl object from a pkl file, or set up the dl object without the pkl file.

## train_headless.py 
A python script that reads the dl structure
from a pkl file. The number of epochs can be entered. The other 
dl parameters are as defined in the pl file, created by mianalyzer
with "Save dl object..." action.

## train_headless_noPkl.py 
A python script that creates a dl object
without reading it from a pkl file. Again, the user is prompted for
the number of epochs. Other parameters are currently hard-coded.

