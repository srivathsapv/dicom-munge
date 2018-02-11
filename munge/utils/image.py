"""Image related util functions"""

import dicom
from dicom.errors import InvalidDicomError

import numpy as np

def parse_dicom_file(filename):
    """Parse the given DICOM filename

    :param filename: filepath to the DICOM file to parse
    :return: dictionary with DICOM image data
    """

    try:
        dcm = dicom.read_file(filename)
        dcm_image = dcm.pixel_array

        try:
            intercept = dcm.RescaleIntercept
        except AttributeError:
            intercept = 0.0
        try:
            slope = dcm.RescaleSlope
        except AttributeError:
            slope = 0.0

        if intercept != 0.0 and slope != 0.0:
            dcm_image = dcm_image*slope + intercept
        dcm_dict = {
          'pixel_data' : dcm_image,
          'width': dcm_image.shape[0],
          'height': dcm_image.shape[1],
          'resolution': get_dcm_resolution(dcm)
        }
        return dcm_dict
    except InvalidDicomError:
        return None

def get_dcm_resolution(dcm_img):
    """
    Gets the resolution of the DICOM image

    :param dcm_img: pydicom instance of DICOM image
    :return: Resolution of the DICOM image i.e equivalent spacing of 1 pixel in millimeters
    """
    pixel_spacing = dcm_img.data_element('PixelSpacing').value
    return [float(pixel_spacing[0]), float(pixel_spacing[1])]

def grayscale_to_rgb(img):
    img.resize((img.shape[0], img.shape[1], 1))
    return np.repeat(img.astype(np.uint8), 3, 2)
