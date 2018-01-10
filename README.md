# Keras_practice<br>

Neural Networks using Keras (with Image recognition case study)<br>

Problem Statement:<br>
Here, we need to identify the digit in given images. We have total 70,000 images, out of which 49,000 are part of train images with the label of digit and rest 21,000 images are unlabeled (known as test images). Now, We need to identify the digit for test images. Public and Private split for test images are 40:60 and evaluation metric of this challenge is accuracy.

About Data:<br>
The data set used for this problem is from the populat MNIST data set. Developed by Yann LeCun, Corina Cortes and Christopher Burger for evaluating machine learning model on the handwritten digit classification problem. It is a widely used data set in the machine learning community.<br>


More information about available data:<br>

Train link has "train.csv" and digit images for train and test data.<br>
Test link has "test.csv" having images name, need to predict the image label<br>

download dataset from : https://datahack.analyticsvidhya.com/contest/practice-problem-identify-the-digits/

I have build a structure approach in the python file.<br>
You can:<br>
Save the model<br>
load the model<br>
generate logs <br>
view performance on Tensorboard<br>

I have uploaded my saved  model as well.<br>

I am able to achive 97.2% accuaracy on validation data by tuning the parameters.

