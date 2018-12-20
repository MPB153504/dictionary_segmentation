import numpy as np
from PIL import Image
from dictionary_segmentation import dict_seg
from matplotlib import pyplot as plt

# Load and preprocess the image
dsm = dict_seg.dictionarySegmentationModel()
dsm.load_image('example_images/dummy.png')
dsm.preprocess()

# The number of distinct colors in label image will result in the 
# number of classes
label_im = np.asarray(Image.open('example_images/dummy_label.png'))

# label image is read as RGBA we only use RGB for labelling
dsm.prepare_labels(label_im[:, :, 0:3])

# Propagate labels to dictionary grid and back to image grid
dsm.iterate_dictionary()

seg_im = dsm.segmentation_image
prob_im = dsm.probability_image

plt.imshow(seg_im)
plt.show()
