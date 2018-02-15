from pathlib import Path

import numpy as np

from munge.Dataset import Dataset
from munge.DataElement import DataElement
from munge.utils import *

dataset = Dataset('config.json')
all_data = [data for data in dataset.get_all()]

def test_element_instanceof():
    assert isinstance(all_data[0], DataElement)

def test_dcm_contour_mapping():
    contour_files = [
        'data/contourfiles/SC-HF-I-1/i-contours/IM-0001-0128-icontour-manual.txt',
        'data/contourfiles/SC-HF-I-4/i-contours/IM-0001-0027-icontour-manual.txt',
        'data/contourfiles/SC-HF-I-6/i-contours/IM-0001-0189-icontour-manual.txt'
    ]

    for cfile in contour_files:
        data_element = [element for element in all_data if element.icontour_path == cfile][0]

        contour_content = contour.parse_contour_file(cfile)
        mask = contour.poly_to_mask(contour_content, data_element.dcm_image['width'], data_element.dcm_image['height'])

        assert np.array_equal(mask, data_element.target)
        assert contour.get_dcm_num_for_contour(cfile) == data_element.dcm_num

def test_asarray():
    np.random.shuffle(all_data)
    element = all_data[0]

    img, _ = element.asarray()
    raw_img = image.grayscale_to_rgb(image.parse_dicom_file(element.dcm_path)['pixel_data'])
    assert np.array_equal(img, raw_img)

def test_get_by_study():
    study_data = [e for e in dataset.get_by_study('SCD0000101')]
    assert len(study_data) == 18

def test_plot_for_study():
    plot_path = 'tests/tmp/plot.png'
    dataset.plot_verification_for_study('SCD0000501', plot_path)
    assert Path(plot_path).is_file()
