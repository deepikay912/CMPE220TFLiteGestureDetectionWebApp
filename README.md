# CMPE220 TFLite Gesture Detection WebApp

# Overview

This is a web app used to predict gesture from a webcam using transfer learning. Used a pretrained MobileNet model and train another model using the internal MobileNet activation to predict upto 8 different classes from the webcam defined by the user.


# Technolgies used :

HTML,JavaScript,tensorflow, tensorflow.js, Keras

WebCam is must to run train the model


# Features & Steps:

User can train model by adding multiple gestures
User can see loss and accuracy in web application
User can test the model by giving some test images and our trained model will show predicted gesture on app.
User can download trained model and weights from application.
Converted downloaded model to TFLite model using TensorFlow converter
Tested converted model using same web application.

![UI](https://github.com/deepikay912/CMPE220TFLiteGestureDetectionWebApp/blob/master/UIScreenshots/UI.png)



# Implementation Details 

First, I have trained the model using web application by collecting all gestures from web cam, calculated accuracy and loss for the model. For this I have used two models. One model will be the pretrained MobileNet model that is truncated to output the internal activations. This model does not get trained after being loaded into the browser.

The second model will take as input the output of the internal activation of the truncated MobileNet model and will predict probabilities for each of the selected output classes which can be up, down, left, right, left click, right click, scroll up and scroll down. This is the model we'll actually train in the browser.

By using the internal activation of MobileNet, we can reuse the features that MobileNet has already learned to predict the 1000 classes of ImageNet with a relatively small amount of retraining.

The base model being used here is MobileNet with a width of .25 and input image size of 224 X 224 and used intermediate depth wise convolutional layer such as conv_pw_13_relu by calling getLayer('conv_pw_13_relu'). we can choose other layers to see how accuracy changes.

# Convert to Tensorflow Lite 

Once we download trained model,converted that model to TFLite model using tensorflow converters.Copied TFlite model to assets folder to use that model in testing gestures from browser

 Steps to convert to TFLite model :

Deserializing the Keras model from TensorFlow.js
To begin with, we have to convert the sequential block trained with TensorFlow.js to Keras H5. This is achieved by employing tfjs-converter's load_keras_model method which is going to deserialize the Keras objects.

Merging the base model and the classification block
This step involves loading the aforementioned custom classification block that was just generated and then passing the output of the base model's intermediate Depthwise Convolutional Layer's activation as input to the top classification block.

Generating the TensorFlow Lite model
After obtaining the Keras H5 model, the session is cleared of all the global variables and reloaded again from the saved Keras model. Finally, the TensorFlow Lite Optimizing Converter or TOCO is used to convert the model from Keras to TFLite FlatBuffers.

# How to run app  

 Open intex.html in browser 
 Click on train to train data, test to test data, download to download model and weights

# Additional Information

- Entire code is attached for data training is attached in WebApplication folder
- Code to convert TensorFlow to TensorFlowLite model is in TFtoTFLite folder
- UI screenshots are in UIScreenshots folder
- output files generated like TFLite model, model weights bin are in output files folder

# References :

https://medium.com/tensorflow/introducing-tensorflow-js-machine-learning-in-javascript-bf3eab376db

https://www.tensorflow.org/lite/examples







