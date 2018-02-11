import numpy as np
from pathlib import Path

from munge.Dataset import Dataset
from munge.DataElement import DataElement
from munge.DataLoader import DataLoader

dataset = Dataset('config.json')
all_data = [data for data in dataset.get_all()]
data_loader = DataLoader(dataset)

def test_load_train_data():
    epochs = 20
    batch = 4

    train_data = data_loader.load_train_data(epochs=epochs, batch_size=4)

    epoch_length = len(train_data)
    iteration_size = len(train_data[0][0])

    assert epoch_length == epochs
    assert iteration_size == 24 # total data size = 96. batch size = 4. so iteration size is 96/4=24

def test_plot_random_epoch():
    plot_path = 'tests/tmp/plot_epoch.png'
    train_data = data_loader.load_train_data()
    data_loader.plot_random_epoch(train_data, epoch_size=10, filename=plot_path)
    assert Path(plot_path).is_file()
