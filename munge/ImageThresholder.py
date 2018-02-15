"""Class to threshold an image and plot necessary figures related to thresholding"""

from munge.utils import image as image_utils, misc

from sklearn.mixture import GaussianMixture
import numpy as np
from scipy.stats import norm
import SimpleITK as sitk
import matplotlib.pyplot as plt
import matplotlib as mpl
from skimage import morphology, filters

class ImageThresholder(object):
    """
    ImageThresholder class can be instantiated with the following args

    - **parameters**, **types**, **return** and **return types**::
    :param data_element: Instance of ``DataElement`` class
    :param n_components: Number of components to the model fit
    :type method: Method to use for model fit. Currently only GMM is supported
    :type postprocess: Boolean to specify whether to do morphological postprocessing after thresholding
    """
    def __init__(self, data_element, n_components=2, method='gmm', postprocess=False):
        self.data_element = data_element
        self.image = self.data_element.image
        self.method = method
        self.postprocess = postprocess

        if not self.data_element.ocontour_path:
            raise ValueError('DataElement does not have an outer contour')

        self.mask = self.data_element.ocontour_mask
        self.masked_image = self.image[self.mask]

        self.n_components = n_components
        self.model = None

        if method != 'gmm':
            raise ValueError('Currently only GMM thresholding is supported')

    def get_thresholded_contour_mask(self):
        """
        Thresholds the o-contour region and returns a mask

        :return: Boolean mask containing the thresholded image
        """

        uniq = np.unique(self.masked_image, return_counts=True)
        data_points = [point for point in zip(*uniq)]

        # define a gaussian mixture model and fit it to the masked image
        gmm = GaussianMixture(n_components=self.n_components, covariance_type='full').fit(self.masked_image.reshape(-1, 1))
        self.model = gmm

        # take the cluster_intensities which are the means of the gaussians
        cluster_intensities = [p[0] for p in gmm.means_]

        # if bi-modal then take the average otherwise take average of low and medium intensity to get the threshold
        if self.n_components == 2:
            threshold = np.mean(cluster_intensities)
        else:
            cluster_intensities.sort()
            threshold = np.mean(cluster_intensities[:2])

        # generate the gaussian curves for plotting purpose using scipy.stats.norm.pdf
        x = np.arange(0, self.masked_image.max())
        std_devs = np.sqrt(gmm.covariances_.flatten())
        gaussians = np.array([p * norm.pdf(x, mu, std_dev)
                                    for mu, std_dev, p in zip(gmm.means_.flatten(), std_devs, gmm.weights_)])

        self.fits = gaussians
        self.threshold = threshold

        img = self.image.copy()

        #first make all the pixels other than the o-contour mask as 0. we are not interested anything outside o-contour
        img[~self.mask] = 0

        thresholded_img = (img > threshold)

        if self.postprocess:
            postprocess_pipeline = [self.dilate]

            processed_img = thresholded_img

            for fn in postprocess_pipeline:
                processed_img = fn(processed_img)

            return processed_img

        return thresholded_img


    def dilate(self, thresholded_img):
        """
        Performs binary dilation on the given image using a disk-shaped structural element of arbitrary radius 3.

        :param thresholded_img: thresholded image
        :return: dilated image
        """
        return morphology.binary_dilation(thresholded_img, morphology.disk(radius=3))

    def get_jaccard_coeff(self):
        """
        Gets the jaccard similarity co-efficient of the reference and the detected region

        :return: jaccard coefficient
        """
        contour_mask = self.get_thresholded_contour_mask()

        image_copy = self.image.copy()
        image_copy[contour_mask] = 255
        image_copy[~contour_mask] = 0

        ref_img = self.image.copy()
        ref_img[self.data_element.target] = 255
        ref_img[~self.data_element.target] = 0

        label_measures = sitk.LabelOverlapMeasuresImageFilter()
        label_measures.Execute(sitk.GetImageFromArray(ref_img), sitk.GetImageFromArray(image_copy))

        return label_measures.GetJaccardCoefficient()

    def plot_model_fit(self, filename=None):
        """
        Plots the histogram of the o-contour region with the gaussians that were used to fit the model and the selected
        threshold value

        :param filename: File path to save the plot
        """
        if not self.model:
            self.get_thresholded_contour_mask()

        plt.hist(self.masked_image, bins=256, normed=True)

        for i in range(len(self.fits)):
            plot_label = '{} Fit {}'.format(self.method, i+1)
            plt.plot(np.arange(self.masked_image.max()), self.fits[i], label=plot_label, linewidth=3.0)

        plt.axvline(x=self.threshold, label='Threshold value', color='k')

        plt.title('Histogram of LV region with model fits')
        plt.xlabel('Intensity')
        plt.ylabel('Relative count')
        plt.legend()

        if filename:
            plt.savefig(filename)
        else:
            plt.show()

    def plot_thresholding_result(self, filename=None):
        """
        Plots the thresholded region along with the ground truth region overlaid in different colors

        :param filename: File path to save the plot
        """
        min_x, max_x, min_y, max_y = misc.get_bounding_box_coords(self.data_element.ocontour)

        rgb_img = image_utils.grayscale_to_rgb(self.image)
        plt.imshow(self.image[min_x:max_x, min_y:max_y], interpolation='nearest', cmap='gray')

        ground_truth_icontour = rgb_img.copy()
        ground_truth_icontour[self.data_element.target] = [0, 255, 0]
        plt.imshow(ground_truth_icontour[min_x:max_x, min_y:max_y], interpolation='bilinear', alpha=0.5)

        thresholded_mask = self.get_thresholded_contour_mask()
        predicted_contour = rgb_img.copy()
        predicted_contour[thresholded_mask] = [255, 0, 0]
        plt.imshow(predicted_contour[min_x:max_x, min_y:max_y], interpolation='bilinear', alpha=0.4)

        plt.xticks([])
        plt.yticks([])

        plt.suptitle('Overlay of ground truth and predicted i-contour')
        plt.title('Jaccard Coefficient = {:.3f}. Green - Ground Truth, Red - Predicted (Yellow is the common region)'.format(
            self.get_jaccard_coeff()
        ), fontsize=8)

        if filename:
            plt.savefig(filename)
        else:
            plt.show()
