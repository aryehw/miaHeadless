# miaHeadless
Code to use mianalyzer headless. The jupyter scripts are more up to date.
When I get a chance, I update the .py versions.
Currently, menuTrainHeadless.ipynb is that most developed script. 
I am having some trouble with the tkinter resources, but it is usable.

mianalyzer is a open source standalone deep learning (DL) application with a
simple GUI interface for building DL networks of various
architectures and backbones.
mianalyzer was writen by Nils Koerber (nils.koerber@bfr.bund.de)
At present there is no defined API for mianalyzer. This project was
done with Nils Koerber's help (he wrote the first python script for headless
training and answered numerous questions), and some reverse enginereing of the 
source code on my part. (@aryehw)

The codes in this project are meant to enable mianalyzer to be run headless.
This allows mianalyzer to be run on remote GPU servers over networks that
cannot support GUI update in reasonable time.

## menuTrainHeadless.ipynb
This version of the trainHeadless_noPkl.ipynb uses tkinter to open directory
and file chooser menus. This is more conventient, but the choosers may open
slowly when run over remote servers.
The option to not use a pkl file to initalize the dl object was removed
in thei script, because I was unable to get that option to work properly.

## trainHeadless_noPkl.ipynb 
A jupyter script that can either load a dl object from a pkl file,
or set up the dl object without the pkl file.
Currently, setting up the dl object without reading it from a
saved pkl file does not produce accurate results. 
However, after the pkl file is loaded, the backbone and number of epochs
can be changed.

## train_headless.py 
A python script that reads the dl structure
from a pkl file. The number of epochs can be entered. The other 
dl parameters are as defined in the pkl file, created by mianalyzer
with "Save dl object..." action.

## train_headless_noPkl.py 
A python script that creates a dl object
without reading it from a pkl file. Again, the user is prompted for
the number of epochs. Other parameters are currently hard-coded.

These scripts require the presence of mianalyzer modules in the import path.
The required directories are mia/dl and mia/util, which can be found on the
MIAnalyzer github page:
https://github.com/MIAnalyzer/MIA/
