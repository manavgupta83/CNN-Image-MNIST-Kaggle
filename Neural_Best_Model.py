
# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np
from sklearn import ensemble, feature_extraction, preprocessing, svm
from sklearn.cross_validation import train_test_split,cross_val_score
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')

#import lasagna and Nolearn 
import lasagne
from lasagne import layers
from lasagne.updates import nesterov_momentum,sgd
from nolearn.lasagne import NeuralNet
import theano

#import pickle to download the pickilized dataset
import sys
import os
import gzip
import pickle
import numpy
from sklearn.cross_validation import train_test_split
from sklearn.utils import shuffle

# import data
train = pd.read_csv("train_mnist.csv")
test = pd.read_csv("test_mnist.csv")


# In[45]:



list_ind = [
1182
]




for i in list_ind:

     ### steps 1 : take a row with 728 pixels

    data = test.ix[i,:]
    print i
#     print data

     ### steps 2 : convert it into 28*28 image matrix

    def transfer(digit_pixel):
        img_matrix = np.zeros((28,28))
        for i in range(0,27):
            for j in range (0,27):
                index = i * 28 + j
                img_matrix[i][j] =digit_pixel[index+1]
        return img_matrix

    img = transfer(data)

#     print img

### steps 3 : matplotlib imshow can plot an image matrix.. 

    plt.imshow(img,cmap = plt.get_cmap('gray'))
    plt.axis('off')
    plt.show()



# In[56]:




# In[18]:


train_x_final = np.asarray(train.ix[:,1:], dtype = np.float32)
train_y_final = np.asarray(train.ix[:,0], dtype = np.int32)

# train_x_final, train_y_final = shuffle(train_x_final, train_y_final, random_state = 21)

train_x_final = train_x_final.reshape(-1, 1, 28, 28)
train_x_final, train_y_final = shuffle(train_x_final, train_y_final, random_state = 21)
       

class AdjustVariable(object):
    def __init__(self, name, start=0.03, stop=0.001):
        self.name = name
        self.start, self.stop = start, stop
        self.ls = None

    def __call__(self, nn, train_history):
        if self.ls is None:
            self.ls = np.linspace(self.start, self.stop, nn.max_epochs)

        epoch = train_history[-1]['epoch']
        new_value = float32(self.ls[epoch - 1])
        getattr(nn, self.name).set_value(new_value)
        

def float32(k):
    return np.cast['float32'](k)
       

#Create a Nueral Network with 3 layers - Input , Hidden and Output 
net1 = NeuralNet(
    layers=[('input', layers.InputLayer),
            ('conv1', layers.Conv2DLayer),
            ('pool1', layers.MaxPool2DLayer),
            ('conv2', layers.Conv2DLayer),
            ('pool2', layers.MaxPool2DLayer),
            ('conv3', layers.Conv2DLayer),
            ('pool3', layers.MaxPool2DLayer),
            ('hidden1', layers.DenseLayer),
            ('dropout1', layers.DropoutLayer),
            ('hidden2', layers.DenseLayer),
            ('dropout2', layers.DropoutLayer),
            ('hidden3', layers.DenseLayer),
            ('dropout3', layers.DropoutLayer),
            ('output', layers.DenseLayer),
            ],
    # layer parameters:
    input_shape=(None,1,28,28), # The shape of input data or training/test data - 784 dimensions . So 784 inputs
    
    conv1_num_filters=32, conv1_filter_size=(3, 3), pool1_pool_size=(2, 2),
    conv2_num_filters=64, conv2_filter_size=(2, 2), pool2_pool_size=(2, 2),
    conv3_num_filters=128, conv3_filter_size=(2, 2), pool3_pool_size=(2, 2),
    
    
    hidden1_num_units=200,  # number of units in 'hidden' layer
#     hidden1_nonlinearity = lasagne.nonlinearities.sigmoid,
    dropout1_p = 0.2, # dropout probability
    
    hidden2_num_units = 100,
#     hidden2_nonlinearity = lasagne.nonlinearities.sigmoid,
    dropout2_p = 0.1,
    
    hidden3_num_units = 50,
    hidden3_nonlinearity = lasagne.nonlinearities.sigmoid,
    dropout3_p = 0.05,
    
    output_nonlinearity=lasagne.nonlinearities.softmax,  # The activation function 
    
    output_num_units=10,  # 10 target values for the digits 0, 1, 2, ..., 9

    # optimization method:
    update=nesterov_momentum,
#     update = sgd,
    update_learning_rate=0.001, # Keep higher for faster learning (inaccurate) and lower for slow convergence ( more accurate)
    update_momentum=0.9, # Momentem Parameter

    #NEW ENTRY
#     objective_loss_function = lasagne.objectives.multiclass_hinge_loss,
    
#      on_epoch_finished=[
#         AdjustVariable('update_learning_rate', start=0.03, stop=0.0001),
#         AdjustVariable('update_momentum', start=0.9, stop=0.999),
#         ],
    
    eval_size = 0.1,
    regression = False,
    max_epochs = 2, # Number of epochs the training will be done. Higher the value better trained but also risk of overfitting
    verbose=1,
    )

# Train the network on the training data - Similar to Scikit-learn fit function
net1.fit(train_x_final,train_y_final)



# In[8]:

# test_check = np.asarray(test, dtype = np.float32)
# test_check = test_check.reshape(-1,1,28,28)
# test_check
predicted = net1.predict_proba(test_check)
predicted_nn = pd.DataFrame(predicted, index = test.index, columns = list('0123456789'))
predicted_nn.head(10)
predicted_nn.to_csv("predicted_nn_iter2_probab.csv")


# In[20]:

from lasagne.objectives import multiclass_hinge_loss


# In[ ]:


# scaler = preprocessing.StandardScaler()
# train_final_new = scaler.fit_transform(train_final.ix[:,1:])
# train_x = pd.DataFrame(train_final_new, index = train_final.index, columns = zero_var_list)

# C_range = np.logspace(-2, 10, 13)
# gamma_range = np.logspace(-9, 3, 13)
# param_grid = dict(gamma=gamma_range, C=C_range)
# cv = StratifiedShuffleSplit(train_final.ix[:4000,0], n_iter=5, test_size=0.2, random_state=42)
# grid = GridSearchCV(SVC(), param_grid=param_grid, cv=cv)
grid.fit(train_x, train_final.ix[:,0])

# print("The best parameters are %s with a score of %0.2f"
#       % (grid.best_params_, grid.best_score_))

