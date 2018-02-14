"""Class to represent a data point in the dataset with relevant features and methods"""
import numpy as np

from .utils import contour, image, misc

class DataElement(object):
    """
    DataElement class can be instantiated with the following args

    - **parameters**, **types**, **return** and **return types**::
    :param dicom_path: full path of the DICOM image
    :param contour_path: full path of the corresponding contour file
    :type dicom_path: string
    :type contour_path: string
    """

    def __init__(self, dicom_path, icontour_path):
        self.id = misc.get_uuid()

        self.dcm_path = dicom_path
        self.icontour_path = icontour_path
        #self.ocontour_path = ocontour_path

        self.dcm_num = contour.get_dcm_num_for_contour(self.icontour_path)

        self.dcm_image = image.parse_dicom_file(self.dcm_path)
        self.image = self.dcm_image['pixel_data']

        self.icontour = contour.parse_contour_file(self.icontour_path)

        self.target = contour.poly_to_mask(self.icontour, self.dcm_image['width'], self.dcm_image['height'])

    def asarray(self):
        """
        Returns the DataElement in the form of (data, label)

        :return: array of data and labels
        """
        return [image.grayscale_to_rgb(self.image), self.target]

    def get_image_contour_overlay(self, window=30, patch_color=[255, 0, 0]):
        """
        Gets a bounding box around the contour with and without the contour overlaid (horizontally stacked).
        This will be useful for manual verification of the annotation

        :param window: size of bounding box required around the marked contour
        :param patch_color: [r, g, b] value of the color in which the patch should be overlaid
        :return: horizontally stacked array with left image being original and the right with the patch drawn
        """
        width, height = [self.dcm_image['width'], self.dcm_image['height']]
        data_rgb = image.grayscale_to_rgb(self.image)

        x_values = [c[0] for c in self.icontour]
        y_values = [c[1] for c in self.icontour]

        min_x = int(min(x_values)) - window
        max_x = int(max(x_values)) + window
        min_y = int(min(y_values)) - window
        max_y = int(max(y_values)) + window

        data_copy = np.array(data_rgb, copy=True)
        data_copy[self.target] = patch_color

        data_cutout = data_copy[min_x:max_x, min_y:max_y]
        data_original = data_rgb[min_x:max_x, min_y:max_y]
        return np.hstack((data_original,data_cutout))

    def get_roi_avg_relative_intensity(self):
        """
        Gets the relative intensity (%) of the ROI. Relative intensity is w.r.t the maximum intensity of the image

        :return: average intensity in percentage
        """
        avg_absolute_intensity = np.mean(self.image[self.target])
        avg_relative_intensity = (avg_absolute_intensity/self.image.max()) * 100
        return avg_relative_intensity

    def get_area_in_sqmm(self):
        """
        Gets the area of the contour in sq.mm. The conversion is done using the ``PixelSpacing`` tag of the DICOM image.

        :return: area in sq.mm
        """
        area_in_pixels = len(self.icontour)

        res_x, res_y = self.dcm_image['resolution']
        conversion_factor = (res_x * res_y)

        return (area_in_pixels * conversion_factor)
