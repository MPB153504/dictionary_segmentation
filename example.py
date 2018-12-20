import numpy as np
from PIL import Image
from dictionary_segmentation import dict_seg
from matplotlib import pyplot as plt


dsm = dict_seg.dictionarySegmentationModel()
dsm.load_image('example_images/dummy.png')
dsm.preprocess()

label_im = np.asarray(Image.open('example_images/dummy_label.png'))

dsm.prepare_labels(label_im[:,:,0:3])
dsm.iterate_dictionary()

seg_im = dsm.segmentation_image
prob_im = dsm.probability_image

plt.imshow(seg_im)
plt.show()
