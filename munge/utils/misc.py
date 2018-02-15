"""Miscellaneous util functions"""
import json
import uuid
import csv
import os

import numpy as np
from sklearn.mixture import GaussianMixture
from scipy.stats import norm

def get_app_config(config_file):
    """
    Gets the application configuration as a dict from the given file

    :param config_file: path to the configuration file
    :return: Dict containing the application configuration
    """

    return json.load(open(config_file))

def get_uuid():
    """
    Generates and returns a random GUID

    :return: Random GUID V4
    """

    return str(uuid.uuid4())

def csv2dict(csv_file):
    """
    Converts and returns a CSV file to Dict

    :param csv_file: Path to the CSV file
    :return: Dict representation of the CSV file
    """

    rows = [r for r in csv.reader(open(csv_file))]
    return {row[0]:row[1] for row in rows[1:]}

def get_ocontour_for_icontour(icontour_file, ocontour_dir):
    """
    Gets the ocontour file corresponding to the given icontour_path. If the ocontour file does not exist, `None` is returned.

    :param icontour_path: Full path to the icontour file
    :return: Path of the corresponding ocontour file, if exists or `None`
    """

    ocontour_path = ocontour_dir + icontour_file.replace('icontour', 'ocontour')
    return ocontour_path if os.path.exists(ocontour_path) else None

def get_bounding_box_coords(contour, window=30):
    """
    Given a contour and window, get the min and max co-ordinates of a bounding box around that window

    :param contour: Array of co-ordinates defining the contour
    :param window: The window size of the bounding box
    :return: min_x, max_x, min_y and max_y of the bounding box
    """
    x_values = [c[0] for c in contour]
    y_values = [c[1] for c in contour]

    min_x = int(min(x_values)) - window
    max_x = int(max(x_values)) + window
    min_y = int(min(y_values)) - window
    max_y = int(max(y_values)) + window

    return [min_x, max_x, min_y, max_y]
