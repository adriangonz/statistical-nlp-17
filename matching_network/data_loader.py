# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Created by: BoyuanJiang
# College of Information Science & Electronic Engineering,ZheJiang University
# Email: ginger188@gmail.com
# Copyright (c) 2017

# @Time    :17-8-27 10:46
# @FILE    :data_loader.py
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import numpy as np


class OmniglotNShotDataset():
    def __init__(self, batch_size, classes_per_set=20, samples_per_class=1, seed=2017, shuffle=True, use_cache=True):
        """
        Construct N-shot dataset
        :param batch_size:  Experiment batch_size
        :param classes_per_set: Integer indicating the number of classes per set
        :param samples_per_class: Integer indicating samples per class
        :param seed: seed for random function
        :param shuffle: if shuffle the dataset
        :param use_cache: if true,cache dataset to memory.It can speedup the train but require larger memory
        """
        np.random.seed(seed)
        # We need to load in CSV files and convert them to something, maybe np.array of dtype=object
        # https://stackoverflow.com/questions/14639496/python-numpy-array-of-arbitrary-length-strings
        self.x = np.load('data/data.npy') 
        #print(self.x.shape)
        # Reshaping from (1623, 20, 28, 28) to (1623, 20, 28, 28, 1)
        # We won't need to do this, but note how the data is organized by
        # Number of classes, number of instances of each class, shape of data
        # We still want something like that
        self.x = np.reshape(self.x, newshape=(self.x.shape[0], self.x.shape[1], 28, 28, 1))
        #print(self.x.shape)
        # Shuffles within the classes, not between them
        if shuffle:
            np.random.shuffle(self.x)
        # Need to modify this for our own splits
        self.x_train, self.x_val, self.x_test = self.x[:1200], self.x[1200:1411], self.x[1411:]
        # self.mean = np.mean(list(self.x_train) + list(self.x_val))
        # We won't need to do this either for sentences eg normalize them here
        self.x_train = self.processes_batch(self.x_train, np.mean(self.x_train), np.std(self.x_train))
        self.x_test = self.processes_batch(self.x_test, np.mean(self.x_test), np.std(self.x_test))
        self.x_val = self.processes_batch(self.x_val, np.mean(self.x_val), np.std(self.x_val))
        # self.std = np.std(list(self.x_train) + list(self.x_val))
        self.batch_size = batch_size
        self.n_classes = self.x.shape[0]
        self.classes_per_set = classes_per_set
        self.samples_per_class = samples_per_class
        # Next line used below for _get_batch_from_cache(self, dataset_name)
        # Not sure where this line is applied
        self.indexes = {"train": 0, "val": 0, "test": 0}
        self.datatset = {"train": self.x_train, "val": self.x_val, "test": self.x_test}
        self.use_cache = use_cache
        if self.use_cache:
            self.cached_datatset = {"train": self.load_data_cache(self.x_train),
                                    "val": self.load_data_cache(self.x_val),
                                    "test": self.load_data_cache(self.x_test)}

    def processes_batch(self, x_batch, mean, std):
        """
        Normalizes a batch images
        :param x_batch: a batch images
        :return: normalized images
        """
        return (x_batch - mean) / std

    def _sample_new_batch(self, data_pack):
        """
        Collect 1000 batches data for N-shot learning
        :param data_pack: one of(train,test,val) dataset shape[classes_num,20,28,28,1]
        :return: A list with [support_set_x,support_set_y,target_x,target_y] ready to be fed to our networks
        """
        # Shape of data_pack = (1200,20,28,28,1), which is Number of (classes in training, num examples per class, image dim)
        #print(data_pack.shape)
        # Will need to change the last two dims here (data_pack.shape[3], data_pack.shape[4]) which are the image dims
        # Similarly where these crop up below
        support_set_x = np.zeros((self.batch_size, self.classes_per_set, self.samples_per_class, data_pack.shape[2],
                                  data_pack.shape[3], data_pack.shape[4]), np.float32)

        support_set_y = np.zeros((self.batch_size, self.classes_per_set, self.samples_per_class), np.int32)
        target_x = np.zeros((self.batch_size, data_pack.shape[2], data_pack.shape[3], data_pack.shape[4]), np.float32)
        target_y = np.zeros((self.batch_size, 1), np.int32)
        #print(support_set_x.shape)
        #print(support_set_y.shape)
        #print(target_x.shape)
        #print(target_y.shape)

        # This should not need to be changed
        for i in range(self.batch_size):
            classes_idx = np.arange(data_pack.shape[0])
            samples_idx = np.arange(data_pack.shape[1])
            choose_classes = np.random.choice(classes_idx, size=self.classes_per_set, replace=False)
            choose_label = np.random.choice(self.classes_per_set, size=1)
            choose_samples = np.random.choice(samples_idx, size=self.samples_per_class + 1, replace=False)

            x_temp = data_pack[choose_classes]
            x_temp = x_temp[:, choose_samples]
            y_temp = np.arange(self.classes_per_set)
            support_set_x[i] = x_temp[:, :-1]
            support_set_y[i] = np.expand_dims(y_temp[:], axis=1)
            target_x[i] = x_temp[choose_label, -1]
            target_y[i] = y_temp[choose_label]

        return support_set_x, support_set_y, target_x, target_y

    # We don't need to do this
    def _rotate_data(self, image, k):
        """
        Rotates one image by self.k * 90 degrees counter-clockwise
        :param image: Image to rotate
        :return: Rotated Image
        """
        return np.rot90(image, k)

    #We don't need to do this
    def _rotate_batch(self, batch_images, k):
        """
        Rotates a whole image batch
        :param batch_images: A batch of images
        :param k: integer degree of rotation counter-clockwise
        :return: The rotated batch of images
        """
        batch_size = batch_images.shape[0]
        for i in np.arange(batch_size):
            batch_images[i] = self._rotate_data(batch_images[i], k)
        return batch_images

    def _get_batch(self, dataset_name, augment=False):
        """
        Get next batch from the dataset with name.
        :param dataset_name: The name of dataset(one of "train","val","test")
        :param augment: if rotate the images
        :return: a batch images
        """
        if self.use_cache:
            support_set_x, support_set_y, target_x, target_y = self._get_batch_from_cache(dataset_name)
        else:
            support_set_x, support_set_y, target_x, target_y = self._sample_new_batch(self.datatset[dataset_name])
        if augment:
            k = np.random.randint(0, 4, size=(self.batch_size, self.classes_per_set))
            a_support_set_x = []
            a_target_x = []
            for b in range(self.batch_size):
                temp_class_set = []
                for c in range(self.classes_per_set):
                    temp_class_set_x = self._rotate_batch(support_set_x[b, c], k=k[b, c])
                    if target_y[b] == support_set_y[b, c, 0]:
                        temp_target_x = self._rotate_data(target_x[b], k=k[b, c])
                    temp_class_set.append(temp_class_set_x)
                a_support_set_x.append(temp_class_set)
                a_target_x.append(temp_target_x)
            support_set_x = np.array(a_support_set_x)
            target_x = np.array(a_target_x)
        # Need to change this and next line. Reshaping not necessery
        support_set_x = support_set_x.reshape((support_set_x.shape[0], support_set_x.shape[1] * support_set_x.shape[2],
                                               support_set_x.shape[3], support_set_x.shape[4], support_set_x.shape[5]))
        support_set_y = support_set_y.reshape(support_set_y.shape[0], support_set_y.shape[1] * support_set_y.shape[2])
        return support_set_x, support_set_y, target_x, target_y

    # Next three functions should not need to be changed
    def get_train_batch(self, augment=False):
        return self._get_batch("train", augment)

    def get_val_batch(self, augment=False):
        return self._get_batch("val", augment)

    def get_test_batch(self, augment=False):
        return self._get_batch("test", augment)

    
    def load_data_cache(self, data_pack, argument=True):
        """
        cache the dataset in memory
        :param data_pack: shape[classes_num,20,28,28,1]
        :return:
        """
        cached_dataset = []
        classes_idx = np.arange(data_pack.shape[0])
        samples_idx = np.arange(data_pack.shape[1])
        for _ in range(1000):
            # This and next line needs to eb changed
            support_set_x = np.zeros((self.batch_size, self.classes_per_set, self.samples_per_class, data_pack.shape[2],
                                      data_pack.shape[3], data_pack.shape[4]), np.float32)

            support_set_y = np.zeros((self.batch_size, self.classes_per_set, self.samples_per_class), np.int32)
            target_x = np.zeros((self.batch_size, data_pack.shape[2], data_pack.shape[3], data_pack.shape[4]),
                                np.float32)
            target_y = np.zeros((self.batch_size, 1), np.int32)
            for i in range(self.batch_size):
                choose_classes = np.random.choice(classes_idx, size=self.classes_per_set, replace=False)
                choose_label = np.random.choice(self.classes_per_set, size=1)
                choose_samples = np.random.choice(samples_idx, size=self.samples_per_class + 1, replace=False)

                # Some changes in these lines
                x_temp = data_pack[choose_classes]
                x_temp = x_temp[:, choose_samples]
                y_temp = np.arange(self.classes_per_set)
                support_set_x[i] = x_temp[:, :-1]
                support_set_y[i] = np.expand_dims(y_temp[:], axis=1)
                target_x[i] = x_temp[choose_label, -1]
                target_y[i] = y_temp[choose_label]
            cached_dataset.append([support_set_x, support_set_y, target_x, target_y])
        return cached_dataset

    # I think this is fine
    def _get_batch_from_cache(self, dataset_name):
        """
        :param dataset_name:
        :return:
        """
        if self.indexes[dataset_name] >= len(self.cached_datatset[dataset_name]):
            self.indexes[dataset_name] = 0
            self.cached_datatset[dataset_name] = self.load_data_cache(self.datatset[dataset_name])
        next_batch = self.cached_datatset[dataset_name][self.indexes[dataset_name]]
        self.indexes[dataset_name] += 1
        x_support_set, y_support_set, x_target, y_target = next_batch
        return x_support_set, y_support_set, x_target, y_target