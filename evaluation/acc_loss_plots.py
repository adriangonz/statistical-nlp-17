import numpy as np
import pandas as pd
import matplotlib.pylab as plt
from matching_network import matching_networks
import torch
import torch.nn as nn
import math
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable

# I obviously have some problems with importing torch (DLL load failed - I will check it tomorrow)
# Also, will I be able to obtain these variables by following invocation?:
acc_5way_1shot, c_loss_5way_1shot = matching_networks.MatchingNetwork()
acc_5way_5shot, c_loss_5way_5shot = matching_networks.MatchingNetwork()
acc_20way_1shot, c_loss_20way_1shot = matching_networks.MatchingNetwork()
acc_20way_5shot, c_loss_20way_5shot = matching_networks.MatchingNetwork()
# I suppose we want to see the difference in train and val set?
#accuracy_train, crossentropy_loss_train = matching_networks.MatchingNetwork()
#accuracy_val, crossentropy_loss_val = matching_networks.MatchingNetwork()


plt.figure(0)
plt.plot(acc_5way_1shot,'r')
plt.plot(acc_5way_5shot,'g')
plt.plot(acc_20way_1shot,'b')
plt.plot(acc_20way_5shot,'y')
# I haven't found info about epochs in matching_networks.py but in omniglot, not sure how it will be connected
plt.xticks(np.arange(0, total_epochs+1, total_epochs/5))
plt.rcParams['figure.figsize'] = (8, 6)
plt.xlabel("Number of Epochs")
plt.ylabel("Accuracy")
plt.title("Training vs Validation Accuracy ") 
plt.legend(['train', 'valid'])
 
plt.figure(1)
plt.plot(c_loss_5way_1shot,'r')
plt.plot(c_loss_5way_5shot,'g')
plt.plot(c_loss_20way_1shot,'b')
plt.plot(c_loss_20way_5shot,'y')
plt.xticks(np.arange(0, total_epochs+1, total_epochs/5))
plt.rcParams['figure.figsize'] = (8, 6)
plt.xlabel("Number of Epochs")
plt.ylabel("Loss ") 
plt.title("Crossentropy Loss ") 
plt.legend(['5-way 1-shot', '5-way 5-shot','20-way 1-shot','20-way 5-shot'])
plt.show()


'''
In case of comparison between train and val sets:

plt.figure(0)
plt.plot(accuracy_train,'r')
plt.plot(accuracy_val,'g')
plt.xticks(np.arange(0, total_epochs+1, total_epochs/5))
plt.rcParams['figure.figsize'] = (8, 6)
plt.xlabel("Number of Epochs")
plt.ylabel("Accuracy")
plt.title("Training vs Validation Accuracy ") 
plt.legend(['train', 'valid'])
 
plt.figure(1)
plt.plot(crossentropy_loss_train,'r')
plt.plot(crossentropy_loss_val,'g')
plt.xticks(np.arange(0, total_epochs+1, total_epochs/5))
plt.rcParams['figure.figsize'] = (8, 6)
plt.xlabel("Number of Epochs")
plt.ylabel("Training vs Validation Loss ") 
plt.legend(['train', 'valid'])
plt.show()

'''