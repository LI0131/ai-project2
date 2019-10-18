# ai-project2
Feed Forward Neural Network for the MNIST dataset using MSE, Sigmoid Units, and Stochastic Gradient Descent

## Git Repository
https://github.com/LI0131/ai-project2

## Sources
https://www.geeksforgeeks.org/ml-stochastic-gradient-descent-sgd/
https://www.geeksforgeeks.org/reading-images-in-python/
https://medium.com/tebs-lab/how-to-classify-mnist-digits-with-different-neural-network-architectures-39c75a0f03e3https://towardsdatascience.com/a-simple-starter-guide-to-build-a-neural-network-3c2cf07b8d7c
https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/
https://medium.com/machine-learning-for-li/explain-feedforward-and-backpropagation-b8cdd25dcc2f
http://www.faqs.org/faqs/ai-faq/neural-nets/part2/section-10.html
https://www.inf.ed.ac.uk/teaching/courses/mlp/2016/mlp02-sln.pdf
https://towardsdatascience.com/gradient-descent-demystified-bc30b26e432a
https://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.shape.html
https://docs.scipy.org/doc/numpy-1.14.0/reference/generated/numpy.random.normal.html
https://medium.com/@14prakash/back-propagation-is-very-simple-who-made-it-complicated-97b794c97e5c

## Dev Setup
There are two options for running this network:
1. `pip3 install keras tensorflow matplotlib numpy` `python3 app.py`
2. `pip3 install pipenv` `pipenv install` `pipenv run app`

## Implementation
My implementation of a Feed Forward Neural Network is given in the `/model` package within the main directory of this project. I chose to create the model as a stand alone package for two reasons: 1. It is highly reusable and adaptable to other projects 2. It allows for quick adaptations witin model class to expand the number of hidden layers. 

### Class Structure
Each layer is built against the abstract class `Layer` in the `layer.py` file. The abstract class defines the key get and set function for the model and defines the class variables that each layer will need -- that is a list of nodes and an error matrix. The input layer requires additional methods for the inserting and reseting images as the values of the input nodes. The output layer require more specialized methods for computing the error matrix after the full pass of the front propagation algorithm. The hidden layer, however, is fairly consistent with the methods defined in the abstract class. 

The weight matrices are each an instance of the `WeightMatrix` class found in the `weight_matrix.py` file. The weight matrix is responsible for corrolating two adjacent layers and providing the methodology for both backward and forward propagation throughout the network. This class is highly dependent upon the functions defined in the `utils.py` file, which contains my implementation of stochastic gradient descent and the sigmoid activation function.

The `Model` class itself is defined within the `__init__.py` file for the `/models` directory. I chose place this class here in order to pair down the number of files that would be necessary. It also makes logical sense to define the `Model` class within the initialization pathway for the model package. The `Model` class itself is used to define the layers that are necessary for the model to perform the forward and backward propagation steps for training (and for greater model use). I define a `__call__` function within the class so that the model object can be passed an image to classify if called upon.

The configuration of hyperparameters is dealt with in the `config.py` file. This file will look at variables defined in the local environment in order to define the values of the constants before using the preset defaults in the file.

### Data Flow
Images are first pulled from the Keras API. The images are complied into dictionary of images and associated labels, which can be subdivided using the splice operators in python. The testing and training sets are derived in such a fashion. The model is then instantiated, and the dictionary of testing images is passed to the `train()` method defined in the model. This method iterates over each image, sets the expected output in the output layer based on the image's label, and the image is set within the input layer. The input layer takes the two dimensional matrix and creates a single vector by iterating over each row in the matrix and using python's `extend()` method. The derived vector is assigned as the value for `self.nodes`. It then calls the forward propagation method in the Model class, which iterates over each weight matrix. As I said in the class structure section, each weight matrix is responsible for both forward and backward propagation. Hence, the Model class's forward propagation method simply calls upon each weight matrix's `propagate_forward()` method. It iterates successively over the input layer, first hidden layer, second hidden layer, and finally to the output layer.

The `propagate_forward()` method in the `WeightMatrix` class is one of the two most central functions in my feed forward neural network implementation. By taking the dot product between the inlayer (the layer from which we are propagating) with the weight matrix, we are able to derive the node vector for the subsequent layer. Since we initialize the weight matrices based on the size of the two layers that it joins, it is always possible to take a dot product between the input layer and the weight matrices since the dimension y dimension of the inlayer must be equivalent to the x dimension of the weight matrix in all cases. After we determine the output vector using the dot product, the subsequent layer's node values are set using the `set_node_values()` function defined in the Layer class. However, before we set the next layer's node values we apply the sigmoid activation function to normalize the output of the dot product such that each value in the output vector will be between 0 and 1. This range is due to the nature of the sigmoid function -- my implementation of this function can be found in the `utils.py` file in the main directory.

Once the forward propagation algorithm has run from the input layer to the output layer, the `train()` function in the Model calls upon the Model class's `_propagate_backward()` method. This function behaves very similarly to the `_propagate_forward()` method with one difference. During the backpropagation process, the model will need to utilize the error matrix of each Layer class implementation. Therefore, the OutputLayer class needs to set its error_matrix before the backpropagation process can begin. 
 
The `set_error_matrix()` method defined in the OutputLayer class is used to perform this functionality. This method takes the `self.node` array and iterates over each index in the array. This array is necessarily of size 10 due to the nature of the problem this model is designed to solve (the MNIST dataset contains images of the ten digits and we are trying to associate the correct label with a given image). Therefore, we can associate each index of the array with a digit value. If the index corrolates to the image label we are training on, the value associated with that index should be 1 if not it should be 0. The difference between the true value in the array and the corresponding 0 or 1 value is defined as the error. This function is defined within the `utils.py` file. The error matrix will, therefore, be a vector of length ten containing the output of the `error()` function for each index value. 

Once the error matrix in the output layer is defined the model iterates over each weight matrix, as it did in the `_propagate_forward()` method, except this time it iterates over the weight matrices in the reverse direction. Like the `propagate_forward()` method in the WeightMatrix, the `propagate_backward()` method is extremely important. It works in conjunction with the `stochastic_gradient_descent()` function defined in the `utils.py` file (I will talk more about the inner workings of this function in the Stochastic Gradient Descent section of this README). In my backpropagation the hidden error is computed first. The error matrix from the output layer (denoted outlayer in the WeightMatrix class) is dot producted with the transpose of the weights. This associates the weights with the error for the output node to which they are connected. Taking the dot product of the weight with the error value gives the amount of the error for which that particular weight is responsible. Using this strategy, the error will be propagated back in the correct proportion to each node in the previous layer (denoted inlayer). The model then uses computes the new weight values using the `stochastic_gradient_descent()` function. This process is repeated for each weight matrix.

The training process continues until each image in the training set has been used to perform both the forward propagation and backward propagation steps.

After the training process is fully completed -- all images have been iterated over -- the testing process can begin. This process is similar to the training process, however, in this case we keep a counter for the number of correctly associated images and labels and we only need complete forward propagation. The testing set is composed of 1/7 of the total datapoints. Like in the training set, the set of images passed to the `test()` method are iterated over one by one. After each weight matrix is iterated through in the `_propagate_forward()` method, the model calls the OutputLayer method `get_classification()`. This method takes the array of node values and determines which index has the maximum value. Given the correlated between the digits and the indices of the output array of node values, we can simply return the index of the array with the maximum value. The value given by the `get_classification()` method is compared against the label for the current image. If the two values are equivalent, we increment the counter for the number of correct associations. After all the images in the set are iterated over, we return the percentage of correct associations given by `num_correct/len(images) * 100`.



