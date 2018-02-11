# Munge

## Module for DICOM data preprocessing and loading

### General Setup

1. Install package dependencies
```
$ conda install --yes --file requirements.txt
```

2. `config.json`
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

### Part 1: Parsing the DICOM images and Contour Files

#### Usage
```
from munge.Dataset import Dataset

dataset = Dataset('config.json')

all_data = dataset.get_all() # contains all images with its corresponding contours
study_data = dataset.get_by_study('SCD0000101') # contains the images and corresponding contours for the given study

# plots the verification for the given study
dataset.plot_verification_for_study('SCD0000101')
```

#### Explanation

1. How did you verify that you are parsing the contours correctly?

**Verification by Unit Tests:**

I have written unit tests which can be found in `tests/test_dataset.py`. Here in the function `test_dcm_contour_mapping()`, I have
asserted whether, for randomly chosen contours, the right DICOM images are picked or not.

**Manual verification by visualization:**

Even though unit tests make sure that the function is doing what it is supposed to, qualitative verification is important in
image annotations. For this purpose I have written a function in the `Dataset` class which will plot the images along with
the contours as patches overlaid on them. For trained sonographers, this gives a quick overview of whether the labelling is
correct or not.

Apart from the images, I have also included two plots
1. Relative average intensity: This will give a rough indicator of whether the region marked is correct or not assuming that
in this case, the average intensity of the myocardium area should lie in some range. If the average intensity does not lie in
this assumed range, we can suspect that the annotater has marked it incorrectly.
2. Area of contour in sq.mm: As each study is a time series, contraction/dilation of the valve can be confirmed with a
sinusoidal/semi-sinusoidal wave. If the marked area is wrong, we can find out using this plot.

(The above plots are included just as a sample to make a point that, some medical information like these can be incorporated
to verify the annotation and to quickly identify mistakes if any)

<insert image>

2. What changes did you make to the code, if any, in order to integrate it into our production code base?

* Modified the given `parse_dicom_file` function to include `width`, `height` and `resolution` in the return value
* Moved the `parse_dicom_file` to `utils/image.py` for better organization
* Moved the given `parse_contour_file` and `poly_to_mask` functions to `utils/contour.py` for better organization
* Wrote a `DataElement` class to abstract each data point in the dataset (for more details refer documentation)
* Wrote a `Dataset` class to load the dataset with the dicom->contour mapping (for more details refer documentation)

### Part 2: Model training pipeline

#### Usage
```
# continuation of the above snippet

from munge.DataLoader import DataLoader

data_loader = DataLoader(dataset)
train_data = data_loader.load_train_data(epochs=10, batch_size=8)
# train_data contains the dataset split into batch_size for each epoch

DataLoader.plot_random_epoch(train_data)
```

#### Explanation

1. Did you change anything from the pipelines built in Parts 1 to better streamline the pipeline built in Part 2? If so, what? If
not, is there anything that you can imagine changing in the future?

* Added `DataLoader` class to load the dataset according to epochs and batch size (for more details refer documentation)
* Refer <enhancements>

2. How do you/did you verify that the pipeline was working correctly?

**Verification by Unit Tests:**

I have written unit tests which can be found in `tests/test_dataloader.py`. Here in the function `test_load_train_data()`, I
have asserted whether, for the given epoch and batch size, the data is split correctly or not.

**Manual verification of randomness by visualization:**

For this purpose I have written a function in the `DataLoader` class which will randomly select an epoch and plot the batch wise
images that were split. This will give a visual indicator that the data split is indeed random.

<insert image>

**Manual verification of randomness by log file:**

The call to the function `load_train_data` will write to the log file (`data_loader.log`), the UUID of the `DataElement`
instance in each epoch, in each iteration. By making sure that the UUIDs are different, we can ensure that the training data
is random enough to be fed into a network.

3. Given the pipeline you have built, can you see any deficiencies that you would change if you had more time? If not, can you
think of any improvements/enhancements to the pipeline that you could build in?
    * Could have used open source code for generating the splits. Tried `keras.preprocessing.image.ImageDataGenerator` but in our
    case the data points are instances of `DataElement` class which the class could not handle
    * Even though I have used `yield` wherever possible, for huge datasets, need to refactor the code such that it works in a
    parallel manner
    * Refer <enhancements>

### Package testing
```
$ pytest --cov=munge --cov-report=html tests/
```
HTML coverage report can be found in `htmlcov/index.html`. Currently the code is 93% covered.

### Auto generated documentation using Sphinx
Documentation can be found in `build/index.html`

### Future Work
* More exception handling
* Data cleaning - Adaptive Histogram Equalization/Mean Normalization
* Integration with LogDNA for better log monitoring
