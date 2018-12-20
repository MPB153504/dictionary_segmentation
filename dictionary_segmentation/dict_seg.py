#!/usr/bin/env python
# coding: utf-8

# ### Special Project @ DTU
# # Dictionary Segmentation Model

# Student: Ari Páll Ísberg (s134713@student.dtu.dk)
# Supervisors: Anders Dahl & Vedrana Dahl

import numpy as np

from PIL import Image
from sklearn.feature_extraction import image as skimage
from sklearn.neighbors import KDTree
from sklearn.cluster import MiniBatchKMeans
from scipy import sparse


class dictionarySegmentationModel:
    def __init__(self):

        """
        Options:
        When using two diffusion steps, we use the resulted probability image
        and run the iteration again. We can apply binarisation of the labels
        between the two diffusion steps. For binarisation we identify the class
        of the highest probability for each pixel, and apply {0, 1} labelling.
        The operation of overwriting imposes the original user-provided
        labelling to all labelled pixels in between the two diffusion steps

        Parameters:
        n_patches is the number of random patches used in the clustering of
        features. n_clusters is the number of clusters, The number of clusters
        should be large, measured in hundreds or thousands, and is roughly
        reflecting the variability in the image. patch_size is thes ize of
        patches. The size of the patches should reflect the scale of the
        distinctive image features and could, for example, be 9 pixels. For
        simplicity, we always assume that the size of the image patches M is
        odd and patches are centred around the central pixel.
        """

        # Options
        self.two_step_diffusion = False
        self.overwrite = False
        self.binarisation = False

        # Define parameters
        self.n_patches = 5000
        self.n_clusters = 500
        self.patch_size = 7 # has to be odd number, min is 3

        self.probability_image = None
        self.segmentation_image = None

    def load_image(self, image_path):
        """ Given a path to image it load the image to the method

        Parameters
        ----------

        image_path : string,
            A path to the image of interest

        Returns
        -------
        """

        self.image_path = image_path

        # Color
        self.im = np.asarray(Image.open(self.image_path))

        # Gray
        # self.im = np.asarray(Image.open(self.image_path).convert("L"))

    def preprocess(self):
        """ Starts the preprocessing of the image
        """

        self.row, self.col = self.im.shape[0:2]
        intensity_dictionary = self.cluster_patches_kmeans_batches(self.im, self.patch_size, self.n_patches, self.n_clusters)
        A, A_vector = self.create_assignments(intensity_dictionary, self.im, self.patch_size)
        B = self.construct_biadjacency_mat(A_vector, self.patch_size, self.n_clusters, self.row, self.col)
        self.T1, self.T2 = self.get_transition_mat(B)

    def prepare_labels(self, label_im_color):
        """ Creates a layered label image, where L(x,y,c) = 1
        if the user indicated that pixel (x, y) belongs to class c,
        and 0 otherwise. The layered label image sums up to 1 in every
        pixel position. All pixels that are don't have any label will get
        the average value (1/n_labels)

        Parameters
        ----------

        label_im_color : numpy array, shape = (image_height, image_width, 3)
            A label image in color, each color in the label image will be a class

        Returns
        -------

        labels : numpy array, shape = (image_height, image_width, n_labels)
             The final layered label image
        """
        self.label_im_color = label_im_color
        self.label_im = np.dot(self.label_im_color[..., :3], [0.299, 0.587, 0.114])

        # Created to hold information on what class is what color
        self.color_reference = {}

        row, col = self.label_im.shape
        unique = np.unique(self.label_im)
        unique_labels = np.delete(unique, np.argmin(unique))

        self.n_labels = unique_labels.shape[0]

        labels = np.ones((row, col, self.n_labels)) * (1 / self.n_labels)

        # Assumes that the max value of the image is the label
        for i, label_val in enumerate(unique_labels):

            labels[:, :, :][self.label_im == label_val] = 0  # All other classes are 0
            labels[:, :, i][self.label_im == label_val] = 1

            # Put in color to the class
            pos_idx = (labels[:, :, i] == 1)
            self.color_reference[i] = self.label_im_color[pos_idx][0]

        self.labels = labels

    def iterate_dictionary(self):
        """ After preprocessing and loading a label image, user can iterate through
        the dictionary. User chooses if he wants to iterate once or twice and
        if he wants to binarize the image between the iterations. Additionally
        he can choose if he wants to overwrite his labels onto the image
        between diffusion steps (iterations) Finally the segmentation is
        colored according to the original colors of the labels and returned as
        a image. Probability images are also available if user wants.
        """

        if self.two_step_diffusion:
            n_loops = 2
        else:
            n_loops = 1

        Pt = self.labels

        for i in range(n_loops):

            # Reshape label image into vector
            L = Pt.reshape((-1, self.n_labels))
            L_sparse = sparse.csr_matrix(L)

            # Compute the propagation
            P_sparse = self.T2 * (self.T1 * L_sparse)

            # Make dense to visualize and reshape into image
            P = np.array(P_sparse.todense())
            P = P.reshape(self.labels.shape)

            Pt = P

            if self.binarisation:
                Pt = self.threshold_max(Pt, self.labels)

            if self.overwrite:
                Pt = self.overwrite_user_labels(Pt, self.labels)

        result = np.argmax(Pt, axis=2)

        result_color = self.create_color_im(result)

        self.probability_image = P
        self.segmentation_image = result_color

    def create_color_im(self, result_im):
        """ As the segmented image is in layers for each class we want to color
        the final image segmentation accordingly to the labels that the user
        provided.
        """

        row, col = result_im.shape
        unique_labels = np.unique(result_im)
        n_labels = unique_labels.shape[0]

        # We want to create RGB image
        rgb = np.zeros((row, col, 3))

        # Give each class in label im a rgb value
        for i in range(n_labels):
            rgb[:, :, :][result_im[:, :] == i] = self.color_reference[i]

        return rgb

    def cluster_patches_kmeans_batches(self, im, patch_size, n_patches, n_clusters):
        """ Computes clusters from random samples of the image and returns
        the centers of each cluster as a matrix (intensity dictionary)

        Parameters
        ----------

        im : numpy array, shape = (image_height, image_width)
            The input image

        patch_size: int
            The size of the patch (one side)

        n_patches: int
            The number of random patches to use in the clustering

        n_clusters: int
            The number of cluster centers to find

        Returns
        -------

        intensity_dictionary : numpy array, shape = (n_clusters, n_patches^2)
             The centers of the clusters as a matrix where each row is basically a patch
        """

        # Get random patches and reshape into matrix
        rand_patches = skimage.extract_patches_2d(
            im, (patch_size, patch_size),
            max_patches=n_patches)

        rand_patches_mat = np.reshape(
            rand_patches, (n_patches, -1), order='C')

        # Cluster random patches (kmeans) and get cluster centers into matrix.
        kmeans = MiniBatchKMeans(
            n_clusters=n_clusters,
            random_state=123,
            compute_labels=False,
            init_size=3 * n_clusters).fit(rand_patches_mat)
        intensity_dictionary = kmeans.cluster_centers_

        return intensity_dictionary

    def create_assignments(self, intensity_dictionary, im, patch_size):
        """ Construct the assignment image "A" where we go through all patches
        in the image and find the corresponding closest cluster in the intensity
        dictionary. Boundary pixels are set to -1.

        Parameters
        ----------

        intensity_dictionary : numpy array, shape = (n_clusters, n_patches^2)
             The centers of the clusters as a matrix where each row is basically a patch

        im : numpy array, shape = (image_height, image_width)
            The input image

        patch_size: int
            The size of the patch (one side)

        Returns
        -------

        A : numpy array, shape = (image_height, image_width)
             The assignment image where each pixel is a integer that corresponds
             to a row in the intensity dictionary. Boundary pixels are set to -1.
        ind: numpy array, shape = ((image_height - (patch_size - 1))*image_width-(patch_size - 1),1)
            it is the assignments as vector instead of matrix
        """
        # Get size/shape
        row, col = im.shape[0:2]

        # Creating a KD tree to match patches from image to the intensity dictionary.
        tree = KDTree(intensity_dictionary)

        # Extract all patches from image and reshape into matrix
        all_patches = skimage.extract_patches_2d(
            im, (patch_size, patch_size), max_patches=None, random_state=None)
        all_patches_mat = all_patches.reshape((len(all_patches), -1))

        # Get the indices in the intensity dictionary
        dist, A_vector = tree.query(all_patches_mat, k=1)

        # Create the Assignment Image A
        pad = int((patch_size - 1) / 2)  # Need to pad because of boundary pixels.
        A_nopad = A_vector.reshape((row - (patch_size - 1), col - (patch_size - 1)))
        A = np.pad(A_nopad, pad_width=pad, mode='constant', constant_values=-1)

        return A, np.array(A_vector)

    def construct_biadjacency_mat(self, A_vector, patch_size, n_clusters, row, col):
        """ Compute the sparse biadjacency matrix "B" that is a linear index between
        the each pixel in the image and the clusters in the dictionary

        Parameters
        ----------

        A_vector : numpy array, shape = (image_height*image_width - (patch_size-1)^2)
             The assignment image where each pixel is a integer that corresponds
             to a row in the intensity dictionary.

        patch_size: int
            The size of the patch (one side)

        n_clusters: int
            The number of cluster centers to find

        row: int
            image height of origincal image

        col: int
            image width of original image

        Returns
        -------

        B : scipy.sparse csr matrix, shape = (image_height*image_width, patch_size*patch_size*n_clusters)
            B[i,j] = 1 if pixel "i" in image matches to pixel "j" in the intensity_dictionary.
        """
        def cartesian_product(*arrays):
            la = len(arrays)
            dtype = np.result_type(*arrays)
            arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
            for i, a in enumerate(np.ix_(*arrays)):
                arr[...,i] = a
            return arr.reshape(-1, la)

        # Rename variables to match equations
        X = row
        Y = col
        M = patch_size
        M2 = M*M
        K = n_clusters
        s = int((M-1)/2)
        XY = X*Y
        MMK = M*M*K
        maxlen = (X-M+1)*(Y-M+1)*M*M

        data = np.ones(maxlen)

        XYdxdy = cartesian_product(np.arange(s, X-s), np.arange(s, Y-s), np.arange(-s, s+1), np.arange(-s, s+1))

        x = XYdxdy[:, 0]
        y = XYdxdy[:, 1]
        dx = XYdxdy[:, 2]
        dy = XYdxdy[:, 3]

        A_repeat = np.repeat(A_vector[:, 0], M2)

        I = (y + dy) + ((x + dx))*Y
        J = (dx + s) + (dy + s)*M + A_repeat*M2

        B = sparse.csr_matrix((data, (I, J)), shape=(XY, MMK))

        return B

    def get_transition_mat(self, B_sparse):
        """ Compute the T1 and T2 transition matrices. T1 -> maps image to dictionary,
        T2 -> maps dictionary to image. Then propagation will be P = T2*T1*L.

        Parameters
        ----------

        B : scipy.sparse csr matrix, shape = (image_height*image_width, patch_size*patch_size*n_clusters)
            B[i,j] = 1 if pixel "i" in image matches to pixel "j" in the intensity_dictionary.

        Returns
        -------

        T1 : scipy.sparse csr matrix, shape = (patch_size*patch_size*n_clusters,image_height*image_width)
            Is a mapping from the image to the dictionary.

        T2 : scipy.sparse csr matrix, shape = (image_height*image_width, patch_size*patch_size*n_clusters)
            Is a mapping from the dictionary to the image.
        """

        XY, MMK = B_sparse.shape  # Get shape
        eps = np.spacing(1)
        # T1
        B_sparse_sum = B_sparse.sum(
            axis=0)  # Sum upp one axis corresponding to B^T * l_{nx1}
        inv_diags_B_sum = sparse.spdiags(
            1 / (B_sparse_sum + eps), 0, MMK, MMK, format='csr')
        T1_sparse = inv_diags_B_sum * B_sparse.T
        # T2
        B_sparse_sum_2 = B_sparse.sum(
            axis=1).T  # Sum upp other axis corresponding to B * l_{mx1}
        inv_diags_B_sum_2 = sparse.spdiags(
            1 / (B_sparse_sum_2 + eps), 0, XY, XY, format='csr')
        T2_sparse = inv_diags_B_sum_2 * B_sparse

        return T1_sparse, T2_sparse

    def threshold_max(self, P, labels):
        """ Thresholds the propability image P by finding the max value over all classes. Furthermore
        it overwrite the user ("correct") labels on top.

        Parameters
        ----------

        P : numpy array, shape = (image_height,image_width,n_labels)
            Probability image where each pixel sums up to 1.

        labels : numpy array, shape = (image_height, image_width, n_labels)
            The final layered user label image

        Returns
        -------

        Pt : numpy array, shape = (image_height,image_width,n_labels)
            Thresholded probability image where each pixel is either 1 or 0 but still sums to 1 in every pixel.
        """

        # Threshold - make the highest propability of the classes 1 and rest 0 '
        arg_max = np.argmax(P, axis=2)

        Pt = np.zeros(labels.shape)

        for i in range(labels.shape[2]):

            Pt[:, :, i][arg_max == i] = 1

        return Pt

    def overwrite_user_labels(self, Pt, labels):
        """ Function overwrites the user ("correct") labels on top.

        Parameters
        ----------

        P : numpy array, shape = (image_height,image_width,n_labels)
            Probability image where each pixel sums up to 1.

        labels : numpy array, shape = (image_height, image_width, n_labels)
            The final layered user label image

        Returns
        -------

        Pt : numpy array, shape = (image_height,image_width,n_labels)
            Input image with user labels overwritten on top
        """

        # Threshold - make the highest propability of the classes 1 and rest 0

        for i in range(labels.shape[2]):

            # Overwrite current threshold with the correct label information
            Pt[:, :, i][labels[:, :, i] == 0] = 0
            Pt[:, :, i][labels[:, :, i] == 1] = 1

        return Pt
