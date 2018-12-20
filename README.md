# dictionary_segmentation

----
## Interactive Dictionary Segmentation Method

Segmentation method based to some extent on the article [Dictionary Based Image Segmentation](http://orbit.dtu.dk/en/publications/dictionary-based-image-segmentation(3b08c76e-7c7a-4bf9-80e6-c4146513a7f6).html) by Anders Bjorholm Dahl & Vedrana Andersen Dahl. The method is ideal for microscopic images with repeated patterns. The information two sources of information is combined; structure in the image and user provided labels. Image patches of the image are captured and clustered to create a dictionary with the main features of the image. User labels are then propagated from image grid to the dictionary and back to the image grid .

A graphical user interface makes the method interactive such that the user can make the segmentation better with feedback.

![](media/demo.gif)

## How to run
You can use the method by itself (see example.py) or run my_gui.py to run the user interface (PyQt 5.11.3 is required). 

*I recommended creating a Python environment with the exact versions as below*

### Requirements 
Python 3.5.2

PyQt5==5.11.3
PyQt5-sip==4.19.13
numpy==1.13.3
qimage2ndarray==1.7
scipy==1.0.0
scikit-learn==0.19.1
Pillow==4.3.0

[1]: http://orbit.dtu.dk/en/publications/dictionary-based-image-segmentation(3b08c76e-7c7a-4bf9-80e6-c4146513a7f6).html