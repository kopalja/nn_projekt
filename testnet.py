
from Network3 import *
from DataLoader import load_data
from LayersWrapper import *
import numpy as np
import pickle
import matplotlib.pyplot as plt
#### Virat
### Visor
### http://clickdamage.com/sourcecode/cv_datasets.php
#test


training_data, mnist_validation_data, test_data = load_data()
#expanded_data, _, _ = load_data("data/mnist_expanded.pkl.gz")


with open("data/dogs_and_snakes.pkl", 'rb') as f:
    training_data, validation_data = pickle.load(f, encoding='latin1')

t1 = training_data[0]
t2 = training_data[1]
t1 = [i[0:150*200] for i in t1]

print(len(t1))
training_data = [t1[:2000], t2[:2000]]

t1 = validation_data[0]
t2 = validation_data[1]
t1 = [i[0:150*200] for i in t1]
validation_data = [t1[:300], t2[:300]]

ar = t1[10]
ar = np.reshape(ar, newshape=(150, 200))
plt.imshow(ar, cmap="gray")
plt.show()



net = Batch_Network(
    layers = LayersWrapper(
        input_shape = (1, 150, 200),
        layers_description = (
            ConvLayer_w(output_images = 20, kernel_size = 5, activation_fn = T.nnet.relu),
            PoolLayer_w(shape = (2, 2)),
            ConvLayer_w(output_images = 40, kernel_size = 5, activation_fn = T.nnet.relu),
            PoolLayer_w(shape = (2, 2)),
            #ConvLayer_w(output_images = 60, kernel_size = 5, activation_fn = T.nnet.relu),
            #ConvLayer_w(output_images = 80, kernel_size = 5, activation_fn = T.nnet.relu),
            # ConvLayer_w(output_images = 60, kernel_size = 5, activation_fn = T.nnet.relu),
            # PoolLayer_w(shape = (2, 2)),
            # ConvLayer_w(output_images = 40, kernel_size = 5, activation_fn = T.nnet.relu),
            # ConvLayer_w(output_images = 40, kernel_size = 5, activation_fn = T.nnet.relu),
            # PoolLayer_w(shape = (2, 2)),
            # ConvLayer_w(output_images = 40, kernel_size = 5, activation_fn = T.nnet.relu),
            #PoolLayer_w(shape = (2, 2)),
            FullyConectedLayer_w(size = 100, activation_fn = T.nnet.relu),    
            FullyConectedLayer_w(size = 100, activation_fn = T.nnet.sigmoid),   
            FullyConectedLayer_w(size = 2, activation_fn = T.nnet.softmax),      
        )
    ),
    training_data = training_data,
    validation_data = validation_data, 
    cost_fn = Cf.probability,
    learning_rate = 0.03, 
    minibatch_size = 20, 
    dropout = 0.5,
    l2_regulation = 0.1,
    update_type = Update.Sgd,
    normalize_data = False,
    load_from_file = None,
    save_to_file = None
)

net.train(epoch = 150)