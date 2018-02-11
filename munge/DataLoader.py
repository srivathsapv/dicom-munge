"""Class to load data in the second stage of the pipeline"""
import numpy as np
import matplotlib.pyplot as plt
import sys

class DataLoader(object):
    """
    DataLoader class can be instantiated with the following args

    - **parameters**, **types**, **return** and **return types**::
    :param dataset: instance of ``Dataset`` class
    :type Dataset: string
    """
    def __init__(self, dataset):
        self.dataset = dataset

    def load_train_data(self, epochs=10, batch_size=8, log_file='data_loader.log'):
        """
        Returns an array of ``DataElement``s split into batches and epochs

        :param epochs: number of epochs needed
        :param batch_size: number of images to be used per batch
        :param log_file: path to the log_file
        :return: array of dimension epochs x (data_size/batch_size) x batch_size containing instances of ``DataElement``
        """
        log_file_handler = open(log_file, "w+")
        log_file_handler.write('Printing image UUIDs in each iteration to confirm randomness!')

        dataset = [element for element in self.dataset.get_all()]

        train_data = []
        for e in range(epochs):
            epoch_data = []
            batches = 0

            log_file_handler.write('\nEpoch #{}\n'.format(e))
            log_file_handler.write(('*' * 50) + '\n')

            dataset_copy = np.array(dataset, copy=True)

            # shuffle and split the data in chunks of batch_size
            np.random.shuffle(dataset_copy)
            data_batches = np.split(dataset_copy, batch_size)

            for batch in data_batches:
                batches += 1
                log_file_handler.write(str([img.id for img in batch]) + '\n')
                epoch_data.append(batch)

            train_data.append(epoch_data)

        log_file_handler.close()
        return train_data

    @staticmethod
    def plot_random_epoch(data, epoch_size=10, filename=None):
        """
        Method to plot the images from a randomly selected epoch

        :param data: return value of ``load_train_data`` function
        :param epoch_size: size of epoch
        :param filename: file to which the plot should be saved
        """
        epoch_data = data[np.random.randint(0, epoch_size)]

        imgs = []
        
        for iter_images in epoch_data:
            x_batch = [iter_img for iter_img in iter_images]
            clips = [i.image for i in x_batch]
            imgs.append(np.hstack(clips))

        final_img = np.vstack(imgs)
        plt.imshow(final_img, cmap=plt.cm.gray, vmin=0, vmax=255)

        if filename:
            plt.savefig(filename)
        else:
            plt.show()
