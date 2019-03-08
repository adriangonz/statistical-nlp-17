# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Created by: BoyuanJiang
# College of Information Science & Electronic Engineering,ZheJiang University
# Email: ginger188@gmail.com
# Copyright (c) 2017

# @Time    :17-8-29 22:26
# @FILE    :mainOmniglot.py
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
from typing import Any, Union

from data_loader import OmniglotNShotDataset
from OmniglotBuilder import OmniglotBuilder
import tqdm
import numpy as np

# Experiment setup
batch_size = 10
fce = True
classes_per_set = 10
samples_per_class = 1
channels = 1
# Training setup
total_epochs = 1
total_train_batches = 10 # The number of batches of size 20
total_val_batches = 10
total_test_batches = 10
best_val_acc = 0.0

data = OmniglotNShotDataset(batch_size=batch_size, classes_per_set=classes_per_set,
                            samples_per_class=samples_per_class, seed=2017, shuffle=True, use_cache=False)

# Vocab
vocab_length = 27449
# print(type(data))
# vocab_length = int(np.amax(data))+1  # type: Union[int, Any]
# print(vocab_length)

obj_oneShotBuilder = OmniglotBuilder(data)
obj_oneShotBuilder.build_experiment(batch_size=batch_size, num_channels=1, lr=1e-3, vocab_length=vocab_length, classes_per_set=20,
                                    samples_per_class=1, keep_prob=0.0, fce=True, optim="adam", weight_decay=0,
                                    use_cuda=True)

with tqdm.tqdm(total=total_train_batches) as pbar_e:
    for e in range(total_epochs):
        total_c_loss, total_accuracy = obj_oneShotBuilder.run_training_epoch(total_train_batches)
        print("Epoch {}: train_loss:{} train_accuracy:{}".format(e, total_c_loss, total_accuracy))
        total_val_c_loss, total_val_accuracy = obj_oneShotBuilder.run_val_epoch(total_val_batches)
        print("Epoch {}: val_loss:{} val_accuracy:{}".format(e, total_val_c_loss, total_val_accuracy))
        if total_val_accuracy>best_val_acc:
            best_val_acc = total_val_accuracy
            total_test_c_loss, total_test_accuracy = obj_oneShotBuilder.run_test_epoch(total_test_batches)
            print("Epoch {}: test_loss:{} test_accuracy:{}".format(e, total_test_c_loss, total_test_accuracy))
        pbar_e.update(1)