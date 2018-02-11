"""Class to represent a dataset as a whole or for each study"""
import os

import matplotlib.pyplot as plt
import numpy as np

from .utils import contour, image, misc
from .DataElement import DataElement

class Dataset(object):
    """
    Dataset class can be instantiated with the following args

    - **parameters**, **types**, **return** and **return types**::
    :param config_file: full path of the application config file
    :type config_file: string
    """
    def __init__(self, config_file='config.json'):
        self.config = misc.get_app_config(config_file)
        self.current_dataset = None

    def get_all(self):
        """
        Maps the images with the contours and returns a generator with data points

        :return: generator of instances of ``DataElement`` having the corresponding image and contour
        """
        dataset = []

        for mapping in list(self._get_all_mapping(self.config['link_file_path'])):
            print(mapping)
            dicom_path = mapping['dicom_path']
            contour_path = mapping['contour_path']

            dataset.append(DataElement(dicom_path, contour_path))

        self.current_dataset = dataset
        yield from dataset

    def _get_all_mapping(self, link_file):
        """
        Combines the result of ``_get_mapping_by_study`` and returns a generator of the mappings
        """
        link = misc.csv2dict(link_file)
        all_mapping = []

        for patient_id, original_id in link.items():
            mapping_for_study = self._get_mapping_by_study(patient_id, original_id)
            #all_mapping.append([mapping for mapping in mapping_for_study])
            yield mapping_for_study

        # flattened = [mapping for study_mapping in all_mapping for mapping in study_mapping]
        # yield from flattened

    def get_by_study(self, patient_id):
        """
        Maps the images with contours and returns a generator with data points, for the given study

        :param patient_id: unique ID of the study
        :return: generator of instances of ``DataElement`` having the corresponding image and contour, for the given study
        """
        link = misc.csv2dict(self.config['link_file_path'])
        original_id = link[patient_id]

        dataset = []

        for mapping in self._get_mapping_by_study(patient_id, original_id):
            dicom_path = mapping['dicom_path']
            contour_path = mapping['contour_path']

            yield DataElement(dicom_path, contour_path)

        #self.current_dataset = dataset
        #yield from dataset

    def _get_mapping_by_study(self, patient_id, original_id):
        """
        For a given study, finds the mapping between the images and the contours
        """
        contour_dir = self.config['icontour_dir_template'].format(original_id)

        mapping = []
        for contour_file in os.listdir(contour_dir):
            dcm_num = contour.get_dcm_num_for_contour(contour_file)

            yield {
                'dicom_path': self.config['dicom_path_template'].format(patient_id, dcm_num),
                'contour_path': contour_dir + contour_file
            }

        #yield from mapping

    def plot_verification_for_study(self, patiend_id, filename=None, rows=5, columns=5):
        """
        Plots a series of images with the corresponding contour patches for the given study

        :param patient_id: unique ID of the study
        :param filename: filename to save the plot in
        :param rows: number of rows in the plot
        :param columns: number of columns in the plot
        """
        fig = plt.figure(figsize=(15, 15))

        study_elements = [overlay for overlay in self.get_by_study(patiend_id)]
        study_elements.sort(key=lambda element: element.dcm_num)

        for i, data_element in enumerate(study_elements):
            overlay = data_element.get_image_contour_overlay()

            ax = fig.add_subplot(rows, columns, i + 1)
            ax.set_title('{}.dcm'.format(data_element.dcm_num))
            ax.set_xticks([])
            ax.set_yticks([])
            plt.imshow(overlay, interpolation='none')

        ax = fig.add_subplot(rows, columns, len(study_elements) + 2)
        ax.set_title('Relative Avg. Intensity (RAI)')
        ax.set_xlabel('Image')
        ax.set_ylabel('RAI')
        intensities = [element.get_roi_avg_relative_intensity() for element in study_elements]
        plt.plot(np.arange(len(intensities)), intensities)

        ax = fig.add_subplot(rows, columns, len(study_elements) + 3)
        ax.set_title('Area of ROI in sq.mm')
        ax.set_xlabel('Image')
        ax.set_ylabel('Area')
        areas = [element.get_area_in_sqmm() for element in study_elements]
        plt.plot(np.arange(len(areas)), areas)

        fig.suptitle('Study {}'.format(patiend_id))

        if filename:
            plt.savefig(filename)
        else:
            plt.show()

    def asarray(self):
        """
        Returns the array representation of all the data points in this dataset

        :return: array of data and labels of this dataset
        """
        elements = self.get_all() if not self.current_dataset else self.current_dataset

        elements_array = [e.asarray() for e in elements]
        data = [e[0] for e in elements_array]
        label = [e[1] for e in elements_array]

        return [np.asarray(data), np.asarray(label)]

    def to_dict(self, patient_id=None):
        """
        Returns the Dict representation of the dataset

        :return: Dict having id, dcm_path and contour_path attributes of the data points in this dataset
        """
        if patient_id:
            elements = self.get_by_study(patient_id)
        else:
            elements = self.get_all()

        return [{'id': e.id, 'dcm_path': e.dcm_path, 'contour_path': e.contour_path} for e in elements]
