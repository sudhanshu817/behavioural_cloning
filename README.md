
**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/center_2021_09_10_20_26_07_764.jpg "Centre data"
[image2]: ./examples/center_2021_09_10_20_31_54_815.jpg "Recovery Image"
[image3]: ./examples/center_2021_09_10_20_31_54_887.jpg "Recovery Image"
[image4]: ./examples/center_2021_09_10_20_31_54_955.jpg "Recovery Image"


#### 1. This repo includes all required files to train a model and run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Running the code:
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```
where model.h5 is the saved model.

#### 3. Code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model is a slight modification of the AlexNet architecture, where I removed a few layers and changed the number of dense units in fully connected layers.
The model includes RELU layers to introduce nonlinearity, and the data is normalized in the model using a Keras lambda layer. 

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting. 
The model was trained and validated on different data sets to ensure that the model was not overfitting by creating a training and validation data generator. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road, driving the car in reverse direction.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to first try a simple network with convolution and dense blocks, however this model seemed too simple to learn the appropriatye behaviour.

So in the next iteration I used AlexNet architecture and made few modifications to it. But the model was underfitting and I atrributed this to vanishing gradients.

So I reduced the network size by removing a few convolutional and dense blocks. I also reduced the number of units in fully connected layers.

This time the loss went really low and the model wasn't underfitting or overfitting, so I collected some training data to improve the model at instances where it was failing. After incorporating these changes to my model, it worked fine and the car was able to complete a lap without going off the road.

Note: The model was taking a lot of time to load on my local gpu, so I trained it on cpu.

#### 2. Final Model Architecture

The final model architecture (model.py) consisted of a convolution neural network with the following layers and layer sizes ...

Conv2D filters=96 kernel_size=(11,11) strides=(4,4) padding='same'
BatchNormalization
Activation relu
MaxPooling2D pool_size=(2,2) strides=(2,2) padding='same'
Conv2D filters=256 kernel_size=(5, 5) strides=(1,1) padding='same'
BatchNormalization
Activation relu
MaxPooling2D pool_size=(2,2) strides=(2,2) padding='same'
Conv2D filters=384 kernel_size=(3,3) strides=(1,1) padding='same'
BatchNormalization
Activation relu
Conv2D filters=384 kernel_size=(3,3) strides=(1,1) padding='same'
BatchNormalization
Activation relu
Flatten
Dense 64
BatchNormalization
Activation relu
Dense 1

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded 1 laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image1]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to come back to centre of the road from side. These images show what a recovery looks like starting from ... :

![alt text][image2]
![alt text][image3]
![alt text][image4]

Then I recorded a lap by driving the car in opposite direction because straight lap seemed to be made up of all left turns giving very less oppurtunity for the model to learn a right turn, but driving the car in reverse order helped in providing training instances for the same.

Even after doing this the model seemed to fail when it entered the bridge by colliding with the right edge head-on and getting stuck there. This made me collect some data where I made sure drive very close to the bridge edge and then making a shart turn to the centre.

I also used the data provided by Udacity for training.

I didn't do any pre-processing or data augmentation.

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 10. I used an adam optimizer so that manually training the learning rate wasn't necessary.

Here's my output:
[video_gif](https://giphy.com/embed/eRH1equM8wfUnS9fBL)
