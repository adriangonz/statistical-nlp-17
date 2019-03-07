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

        # WE NEED TO LOAD IN CSV FILES AND CONVERT THEM TO SOMETHING, MAYBE NP.ARRAY OF DTYPE=OBJECT
        self.x = np.load('data/x_train.csv')
        self.x_test = np.load('data/x_test.csv')

        # RESHAPING FROM (1623, 20, 28, 28) TO (1623, 20, 28, 28, 1)
        self.x = np.reshape(self.x, newshape=(self.x.shape[0], self.x.shape[1], self.x.shape[2], self.x.shape[3], 1))

        # SHUFFLES WITHIN THE CLASSES, NOT BETWEEN THEM
        if shuffle:
            np.random.shuffle(self.x)

        self.x_train, self.x_val = self.x[:7000], self.x[7000:]

        self.batch_size = batch_size
        self.n_classes = self.x.shape[0]
        self.classes_per_set = classes_per_set
        self.samples_per_class = samples_per_class
        self.indexes = {"train": 0, "val": 0, "test": 0}
        self.datatset = {"train": self.x_train, "val": self.x_val, "test": self.x_test}
        self.use_cache = use_cache
        if self.use_cache:
            self.cached_datatset = {"train": self.load_data_cache(self.x_train),
        "val": self.load_data_cache(self.x_val),
        "test": self.load_data_cache(self.x_test)}

        def _sample_new_batch(self, data_pack):
            """
        Collect 1000 batches data for N-shot learning
        :param data_pack: one of(train,test,val) dataset shape[classes_num,20,28,28,1]
        :return: A list with [support_set_x,support_set_y,target_x,target_y] ready to be fed to our networks
        """

            # data_pack = (# of class in training, num of examples per class, sentence dim[0],sentence dim[1], 1 )
            support_set_x = np.zeros((self.batch_size, self.classes_per_set, self.samples_per_class, data_pack.shape[2],
                                      data_pack.shape[3], data_pack.shape[4]), np.float32)

            support_set_y = np.zeros((self.batch_size, self.classes_per_set, self.samples_per_class), np.int32)
            target_x = np.zeros((self.batch_size, data_pack.shape[2], data_pack.shape[3], data_pack.shape[4]),
                                np.float32)
            target_y = np.zeros((self.batch_size, 1), np.int32)

            # THIS WILL NEED TO BE CHANGED
            # THERE ARE NO EXPLICIT LABELS FOR THE OMNIGLOT DATA
            # THEY SIMPLY KNOW THAT DATA IN THE SAME 0TH INDEX OF THE INPUT DATA
            # EG IN THE SAME data_pack.shape[0] ARE OFF THE SAME CLASS AND SO
            # WILL HAVE THE SAME LABEL, AND THIS CLASS IS GIVEN A NUMBER FOR A
            # LABEL HERE
            # QUESTION, HOW DO WE AVOID CLASSES BEING GIVEN THE SAME NUMBER FOR A LABEL?
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
            # NEED TO CHANGE THIS AND NEXT LINE. RESHAPING NOT NECESSERY
            support_set_x = support_set_x.reshape(
                (support_set_x.shape[0], support_set_x.shape[1] * support_set_x.shape[2],
                 support_set_x.shape[3], support_set_x.shape[4], support_set_x.shape[5]))
            support_set_y = support_set_y.reshape(support_set_y.shape[0],
                                                  support_set_y.shape[1] * support_set_y.shape[2])
            return support_set_x, support_set_y, target_x, target_y

        # NEXT THREE FUNCTIONS SHOULD NOT NEED TO BE CHANGED
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
                # THIS AND NEXT LINE NEEDS TO BE CHANGED
                support_set_x = np.zeros(
                    (self.batch_size, self.classes_per_set, self.samples_per_class, data_pack.shape[2],
                     data_pack.shape[3], data_pack.shape[4]), np.float32)

                support_set_y = np.zeros((self.batch_size, self.classes_per_set, self.samples_per_class), np.int32)
                target_x = np.zeros((self.batch_size, data_pack.shape[2], data_pack.shape[3], data_pack.shape[4]),
                                    np.float32)
                target_y = np.zeros((self.batch_size, 1), np.int32)
                for i in range(self.batch_size):
                    choose_classes = np.random.choice(classes_idx, size=self.classes_per_set, replace=False)
                    choose_label = np.random.choice(self.classes_per_set, size=1)
                    choose_samples = np.random.choice(samples_idx, size=self.samples_per_class + 1, replace=False)

                    # SOME CHANGES IN THESE LINES
                    x_temp = data_pack[choose_classes]
                    x_temp = x_temp[:, choose_samples]
                    y_temp = np.arange(self.classes_per_set)
                    support_set_x[i] = x_temp[:, :-1]
                    support_set_y[i] = np.expand_dims(y_temp[:], axis=1)
                    target_x[i] = x_temp[choose_label, -1]
                    target_y[i] = y_temp[choose_label]
                cached_dataset.append([support_set_x, support_set_y, target_x, target_y])
            return cached_dataset

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