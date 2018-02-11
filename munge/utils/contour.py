"""Contour related util functions"""

import numpy as np
from PIL import Image, ImageDraw

def parse_contour_file(filename):
    """Parse the given contour filename

    :param filename: filepath to the contourfile to parse
    :return: list of tuples holding x, y coordinates of the contour
    """

    coords_lst = []

    with open(filename, 'r') as infile:
        for line in infile:
            coords = line.strip().split()

            x_coord = float(coords[0])
            y_coord = float(coords[1])
            coords_lst.append((x_coord, y_coord))

    return coords_lst


def poly_to_mask(polygon, width, height):
    """Convert polygon to mask

    :param polygon: list of pairs of x, y coords [(x1, y1), (x2, y2), ...]
     in units of pixels
    :param width: scalar image width
    :param height: scalar image height
    :return: Boolean mask of shape (height, width)
    """

    # http://stackoverflow.com/a/3732128/1410871
    img = Image.new(mode='L', size=(width, height), color=0)
    ImageDraw.Draw(img).polygon(xy=polygon, outline=0, fill=1)
    mask = np.array(img).astype(bool)
    return mask

def get_dcm_num_for_contour(contour_file_name):
    """Gets the DICOM series number for a given contour file name or full file path
    
    Ex: For 'data/contourfiles/SC-HF-I-1/i-contours/IM-0001-0048-icontour-manual.txt' the return value will be 48

    :param contour_file_name: name of the contour file
    :return: Integer corresponding to the DICOM series number
    """
    if '/' in contour_file_name:
        contour_file_name = contour_file_name.split('/')[-1]

    parts = contour_file_name.split('-')
    return int(parts[2])
