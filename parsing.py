from munge.Dataset import Dataset
from munge.DataLoader import DataLoader
import matplotlib.pyplot as plt
import numpy as np

dataset = Dataset('config.json')
all_data = dataset.get_by_study('SCD0000101')

elements = [e for e in all_data]
print(elements[0].contour_path)

# data_loader = DataLoader(dataset)
# train_data = data_loader.load_train_data()
# DataLoader.plot_random_epoch(train_data)
#sdata[slabel] = [255, 0, 0]
# plt.imshow(sdata, interpolation='none', vmin=0, vmax=255)
# plt.show()


# for data in all_data:
#     print(data.get_area_in_sqmm())

#dataset.plot_verification_for_study('SCD0000101')
