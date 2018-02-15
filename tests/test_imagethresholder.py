from pathlib import Path

from munge.DataElement import DataElement
from munge.ImageThresholder import ImageThresholder

def test_imagethresholder():
    element = DataElement('data/dicoms/SCD0000101/139.dcm',
                          'data/contourfiles/SC-HF-I-1/i-contours/IM-0001-0139-icontour-manual.txt',
                          'data/contourfiles/SC-HF-I-1/o-contours/IM-0001-0139-ocontour-manual.txt')

    thresholder = ImageThresholder(element, n_components=2, method='gmm', postprocess=True)

    model_fit_path = 'tests/tmp/model_fit.png'
    thresholder.plot_model_fit(model_fit_path)
    assert Path(model_fit_path).is_file()

    thresholding_result_path = 'tests/tmp/thresholding_result.png'
    thresholder.plot_thresholding_result(thresholding_result_path)
    assert Path(thresholding_result_path).is_file()
