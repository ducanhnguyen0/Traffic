# Traffic: A Neural Network detects traffic signs

Harvard CS50AI Project

## Description:

A neural network to classify road signs based on an image of those signs using TensorFlow, OpenCV, Numpy.

## Tech Stack:

* Python
* TensorFlow
* OpenCV
* Numpy

## Project Specification:

### load_data
Used the OpenCV-Python module (cv2) to read each image as a numpy.ndarray (a numpy multidimensional array) and resize each image to have width IMG_WIDTH and height IMG_HEIGHT to pass these images into a neural network. The function then return a tuple (images, labels) where images should be a list of all of the images in the data set, where each image is represented as a numpy.ndarray of the appropriate size and labels should be a list of integers, representing the category number for each of the corresponding images in the images list.

### get_model
The function return a compiled neural network model using TensorFlow Sequential model included three convolutional layers, the first two layers with 32 filters and the third layer with 64 filters and all three layers apply 3x3 kernel, two max-pooling layers with 2x2 pool size, one hidden layer with 128 units and dropout rate is 0.5 to prevent overfit and one last output layer with number of units are the number of categories we need to classify. Neural Network model is compiled with optimizer 'adam', the function loss 'categorical_crossentropy' and metrics are accuracy.

## How to run

1. Clone this project
2. Run command line in terminal: python traffic.py 'dataset file'

## Experiment
I have performed a lot of experiments in the neural network model with several attempts to:

1. different numbers of convolutional and pooling layers
2. different numbers and sizes of filters for convolutional layers
3. different pool sizes for pooling layers
4. different numbers and sizes of hidden layers
5. dropout

## Conclusion
I have concluded, from my oppions, based on several attempts I have made:

1. After trying different numbers of convolutional and pooling layers, I realized that I can only use upto 3 max-pooling layers before it results in negative dimensions since the size of input is relatively small and using max-pooling layers can reduce pixels which results in faster processing of models. Also, the more number of convolutional layers, the more accuracy the model is since it helps filter to detect the edge of object but at the same time, it can cause the overfitting for models that match too close to the training set. I noticed that the model I made with first two convolutional layers with 32 filters and 3x3 kernel and two max-pooling layers with 2x2 pool size and the third convolutional layers with 64 filters and 3x3 kernel perform the best and most stable.

2. After trying different numbers and sizes of filters for convolutional layers, I realized that the more filter it learns, the more accuracy the model performance will be but it also decrease the processing speed but if the number of filters are relatively large then it will overfit the trainning set since the input is relatively small so it really depends on the input size. And also, more filters help models learn quicker so that's why the accuracy rate increase dramatically. And also, I realized that the filters rate in convolutional layers may affect the result if the hidden layers have number of inputs relatively equal that the accuracy is really low compare the model which have number of inputs in hidden layers more than filters of convolutional layers.

3. After trying different pool sizes for pooling layers, I realized that the pool size will reflect on how the pixels are left for model to process and it also important that different size of pool affect the result and the process time of model.

4. After trying different numbers and sizes of hidden layers, I realized that more hidden layers will make model more generalization since it adds more weights then it need to change the weights to optimize the ouput and also it helps minimize the loss. That's why we can see the accuracy will increase slightly for each epoch not increasing dramatically like we add more convolutional layers.

5. After trying to modify dropout rate, I think the dropout rate can help model more generalization and prevent overfitting. I think dropout rate at 0.5 is relatively safe and useful for most of the cases.

