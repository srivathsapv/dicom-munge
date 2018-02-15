import numpy as np

from munge.DataElement import DataElement

def test_ocontour_mapping():
    element = DataElement('data/dicoms/SCD0000101/48.dcm',
                          'data/contourfiles/SC-HF-I-1/i-contours/IM-0001-0048-icontour-manual.txt',
                          'data/contourfiles/SC-HF-I-1/o-contours/IM-0001-0048-ocontour-manual.txt')
    try:
        element.overlay_contours()
    except AttributeError:
        assert True
    assert len(element.get_image_ocontour_overlay()) == 0

def test_ocontour_overlay():
    element = DataElement('data/dicoms/SCD0000101/59.dcm',
                          'data/contourfiles/SC-HF-I-1/i-contours/IM-0001-0059-icontour-manual.txt',
                          'data/contourfiles/SC-HF-I-1/o-contours/IM-0001-0059-ocontour-manual.txt')
    assert len(element.get_image_ocontour_overlay()) != 0

def test_overlay_contours():
    element = DataElement('data/dicoms/SCD0000101/59.dcm',
                          'data/contourfiles/SC-HF-I-1/i-contours/IM-0001-0059-icontour-manual.txt',
                          'data/contourfiles/SC-HF-I-1/o-contours/IM-0001-0059-ocontour-manual.txt')
    contours = element.overlay_contours()
    assert contours.shape[-1] == 3
