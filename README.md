# Munge
![travis](https://travis-ci.org/srivathsapv/dicom-munge.svg?branch=master)
![coverage](https://github.com/srivathsapv/dicom-munge/blob/master/coverage.svg)
![docs](https://readthedocs.org/projects/dicom-munge/badge/?version=latest)

## Module for DICOM data preprocessing and loading

### General Setup

1. Install package dependencies
```
$ conda install --yes --file requirements.txt
```

2. **config.json**

Download the data and put it in a directory and make changes to `config.json` accordingly. Currently I have my data in the `data`
(git ignored) folder and hence the content of the file is:

```
{
    "dicom_path_template": "data/dicoms/{}/{}.dcm",
    "icontour_dir_template": "data/contourfiles/{}/i-contours/",
    "ocontour_dir_template": "data/contourfiles/{}/o-contours/",
    "link_file_path": "data/link.csv"
}
```

### Jupyter Notebook

The usage of the package is easily illustrated in [this](https://github.com/srivathsapv/dicom-munge/blob/master/Usage.ipynb)
Jupyter Notebook. Detailed explanation is also given here in [Phase 1
Solutions](https://github.com/srivathsapv/dicom-munge/wiki/Phase-1-Answers).


### Part 1: Parse the _o-contours_

* Modified `Dataset` class `get_all` and `get_by_study` methods to get the `o-contour` for each image if it exists
* Modified `DataElement` class to accept the optional argument of `o-contour` while constructing
* Added `overlay_contours` method to `DataElement` to view both contours at the same time
* Did not make any change to `DataLoader` class, assuming that the primary label for the ML training pipeline will be the
i-contours

### Part 2: Heuristic LV Segmentation approaches

This [jupyter notebook](https://github.com/srivathsapv/dicom-munge/blob/master/Thresholding.ipynb) has all the code samples that
were used to generate the below figures and to do the operations described below.

#### Analysis of the O-Contour Histogram

![image_hist1](https://user-images.githubusercontent.com/1017519/36243640-87670886-11f0-11e8-9132-dd6abd6e07c3.png =250x)

Looking at the above histogram of the o-contour of the image (SCD0000101/59.dcm) we can see that it is a bi-modal distribution
with took peaks corresponding to low intensity and high intensity regions respectively. Looking at the data, it is obvious that
we are interested only in the high intensity region. So we should automatically choose a threshold that will retain only the high
intensity pixels.

![image_hist2](https://user-images.githubusercontent.com/1017519/36243645-8b12e82e-11f0-11e8-8ecc-fa26a66b7bfa.png)

It is not always the case that the o-contour histogram is bi-modal in nature. Some times the image has low intensity (black),
medium intensity (gray) and high intensity (white) pixels thus having a tri-modal nature like the one shown above
(SCD0000201/80.dcm). In this case we should choose a threshold that will retain the medium and the high intensity pixels because
in most of the cases the i-contour region is comprised of medium and high intensity pixels and the region between o-contour and
i-contour mostly has low intensity pixels.

#### Fitting a Gaussian Mixture Model to the histogram

Looking at the above histograms, it is obvious that they are distributions comprising of a mixture of gaussians. Hence we can
fit a Gaussian Mixture Model (GMM) to the histogram to automatically choose the optimum threshold value for that image. This
threshold is chosen as,

* The average of the gaussians' means in case of a bi-modal distribution (this is the point where the two curves cross -
  refer figure below)
* The average of the first two gaussians' means in case of a tri-modal distribution

The below figures show the histograms along with the gaussians that were used to fit and the chosen threshold value.

**Bi-modal GMM Fit**
![gmm1](https://user-images.githubusercontent.com/1017519/36243997-5f6dfeaa-11f2-11e8-81b5-689f0f61eac3.png)

**Tri-modal GMM Fit**
![gmm2](https://user-images.githubusercontent.com/1017519/36244001-60a4a896-11f2-11e8-9e07-72eb9d4e0e3f.png)

#### Finding the i-contour by simple thresholding

Once the threshold has been automatically found out by fitting a GMM model, we do simple thresholding on the o-contour region i.e
set all pixels greater than the threshold value to `white` and all other pixels `black`. Then we get the co-ordinates
of these `white` pixels which is nothing but the detected `i-contour`.

#### Qualitative evaluation - Visualization

Once we have the detected `i-contour`, we can overlay this with the ground truth `i-contour` and look at the image to
qualitatively assess the performance of the segmentation. One such visualization is shown below.

![overlay](https://user-images.githubusercontent.com/1017519/36244323-c398cd46-11f3-11e8-922b-1d0ab7156384.png)

#### Quantitative evaluation - Jaccard Coefficient

[Jaccard Coefficient or Jaccard Index](https://en.wikipedia.org/wiki/Jaccard_index) is a widely used metric to measure the
performance of segmentation. Below is the overlay visualization along with the jaccard coefficient metric. For the first
image, the metric is 0.908, which shows that it is a very good prediction. For the Second image, the metric is 0.728, which
shows that it is not a great prediction and there is room for improvement.

![jaccard1](https://user-images.githubusercontent.com/1017519/36244692-44306bde-11f5-11e8-881c-f3702242334c.png)
![jaccard2](https://user-images.githubusercontent.com/1017519/36245170-d7cc3736-11f7-11e8-9bc2-9f0d0f0da7fc.png)

#### Final comments on simple thresholding

From the above results and discussions, it is evident that a simple thresholding scheme is very much possible to do
segmentation but that alone would not suffice when the images have _salt and pepper_ noise and poorly defined boundaries and
intensity differences. For example, the performance of the above scheme purely depends on how well separated the region
intensities are, which might not be the case always.

#### Alternate methods

1. **Morphological Operations and Connected Components**:
In the first image below, we see tiny speckles in the predicted contour and also minor cracks in the boundary of the detected
contour region. The tiny speckles can be removed by the morphological operation `erosion` and the cracks can be filled
by the morphological operation `dilation`.

From the below two images we can see that dilation has improved the jaccard score from 0.728 to 0.826. Here I have dilated
using a disk shaped structural element with an arbitrary radius of 3.

**Before Dilation**:
![dilation1](https://user-images.githubusercontent.com/1017519/36245170-d7cc3736-11f7-11e8-9bc2-9f0d0f0da7fc.png)

**After Dilation**:
![dilation2](https://user-images.githubusercontent.com/1017519/36245360-be1bef88-11f8-11e8-86f7-4fc5dbf57bce.png)

After dilation we can further erode the image to get rid of tiny speckles and finally do [connected
components](http://scikit-image.org/docs/dev/api/skimage.measure.html#skimage.measure.label) to find the `i-contour` region.

2. **Scale Invariant Feature Transform (SIFT):**
Apart from using raw intensities as heuristic, we can do smarter by detecting local features like edges and disks using SIFT
operations. Using SIFT we can also do key point detection and description to localize the `i-contour` region.

### Auto generated documentation using Sphinx
Documentation can be found [here](http://dicom-munge.readthedocs.io/en/latest/).

### Future Work
* Improve GMM by automatically finding the modality (bi-modal or tri-modal) of the image either by using Bayesian Information
Criterion(BIC) or Akaike Information Criterion (AIC) or some other metric.
* The threshold is currently chosen by taking average of the two gaussians. This will be incorrect when the two gaussians have
different standard deviations. So instead take weighted average to compute the threshold.
* Currently dilation is being done using a disk-shaped structural element of arbitrary radius 3. This radius can be decided
based on the radius of the annotated `o-contour` region.
