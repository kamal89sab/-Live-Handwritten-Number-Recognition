# Live Handwritten Digit Recognition Using CNN Model Trained on MNIST Dataset

## Training Dataset

The MNIST database (Modified National Institute of Standards and Technology database) is a large database of handwritten digits that is commonly used for training various image processing systems. The MNIST database of handwritten digits, has a training set of 60,000 28x28 grayscale images of the 10 digits, and a test set of 10,000 examples

## CNN model

The CNN model consists of two sets of 2D convolution and maxpooling layers, followed by a dropout layer to prevent overfitting. The 2D result, after being flattened, goes through a dense layer of 128 neurons with reLU activation followed by another dropout layer. The output is fed into a dense layer of 10 neurons with softmax activation. 

The model is trained for 10 epochs.

## Live Predicion

Uses an openCV window where the user can draw a digit see the predtion in real-time.

### Using the live predictor

draw with LMB down
'enter' key - Perform preprocessing and display result
'c' key - clear the window
'esc' key - exit the program