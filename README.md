#**Behavioral Cloning Project** 

##Introduction##
Udacity created a game that lets you drive around a track and save the video feed and associated steering angles. Using this data, the goal is to train a deep neural network that correctly predicts what steering angles will keep the a car on the road. 

The goals of this project are the following:
* Use the simulator to collect data of good driving behavior.
* Build, a convolution neural network in Keras that predicts steering angles from images.
* Train and validate the model with a training and validation set.
* Test that the model successfully drives around track one without leaving the road.
* Summarize the results with a written report.


##Background and files##

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* This report (writeup_report.md) summarizing the process and results


Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```


##Exploratory analysis##

To start with, I looked at the type of data I was dealing with. To summarize, there are:
* 8036 observations (images and associated steering angles)
* Images which each have the dimension 160*320*3 (160*320 height and width + 3 colors channels)

I then plotted a series of images using matplotlib to get a sense of data quality and data diversity. I found that all the images were informative - although there was a bias towards images with left turns.


##Preprocessing stage##
In the preprocessing stage, I did 2 things. First, I applied a slight Gaussian blur using the standard cv2 library to remove the effects of noise and reduce overfitting. 

Second, I normalized the image by dividing each pixel value by 255. This will help with numerical stability.


##Data splitting##
Next, I shuffled the data and split it training and validation sets, where the validation set made up 10% of the overall size. 


##Solution design##
I started with a very simple two-layer fully connected network to make sure that I could make predictions. I tested on a  small number of observations, and didn't worry about overfitting. From there, I slowly added layers of complexity to make the model fit better - ensuring that loss decreased as I made changes. 

The first layer I added was a Cropping2D layer which removed 50 pixels from the top of each image and 20 pixels from the bottom. I did this because I noticed that those areas of the image didn't provide any value (they were mostly trees and the sky). Then I added multiple Convolution2D layers with increasing depth and a frame of 5x5. This was similar to the Nvidia model so I figured it would be good. After the Convolution layers I also used relu activation layers so that my model could become better at making non-linear predictions. Lastly, I added a dropout layer with dropout probability 0.2 to improve the robustness of the model and reduce overfitting.

I used mean squared error as a loss function because this is generally accepted as a good starting point. And I used the Adam optimizer so that I didn't have to worry about tuning several hyperparameters.

For training, I played aroudn with a number of batch size and epoch combinations to see what worked best. I found that after 10 epochs, improvements generally wore off. Eventually I also found that batch sizes of 64 generally did well.


##Testing##
After training the final model, I ran the simulator in autonomous mode and was able to go around the entire track. 


##Closing thoughts##
* I didn't spend a lot of time training or tuning hyperparameters because I was doing the computations on my local machine. For additional accuracy, I would use Amazon EC2 and do training there. 
There are a number of things I could have done to further improve the model. These include
* Collecting more data or simulating new data. There were more images with left turns, so an easy way to generate additional data would've been to reflect images horizontally and invert steering angles. 
* The car turns seemd rigid so one approach to improve that would've been to use some type of smoothing that keeps a rolling average of steering angles and adjusts them towards the mean. 
* I only used the center images but could have also used the left and right images for additional accuracy. 
* The trainign data I used mostly consisted of good driving. However this introduces a bias and makes the model less good at recovery. For improved accuracy, I would add additional data specifically focused on recover (starting from the edges of a line and returning to the center).






