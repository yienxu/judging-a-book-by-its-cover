# Judging a Book by its Cover: A Modern Approach

Authors:

Yien Xu (yien@cs.wisc.edu)

Boyang Wei (bwei9@wisc.edu)

Jiongyi Cao (jcao56@wisc.edu)

## Abstract

In this work, we employ three machine learning techniques to study the relationship between book covers and their popularity.
We first train a classification model using a Convolutional Neural Network (CNN) to predict the popularity level given a resized and randomly cropped 120 x 120 image input of the book cover. Second we employ an interpretation algorithm called LIME to extract patches in the image to explain the prediction result. Finally, we explore GAN to generate the most popular cover by machine itself. We find that our classification model suffers from over fitting, and thus, we in general fail to conclude a clear relationship between the cover and popularity. However, given a specific instance, LIME extract reasonable features to explain the result. And finally, our GAN is able to provide some general tips of what popular book covers have.

## Code Walk-Through

### Scripts

This [folder](scripts) contains the scripts that we use to generate our dataset - one for CNN and the other for GAN. To get started, please down the dataset from [Kaggle](https://www.kaggle.com/meetnaren/goodreads-best-books) and modify the various paths in the scripts. The scripts will generate `.csv` files with three columns: `title`, `filename`, and `label`. Note that the scripts will try to open the images to detect corruption, so please make sure to download the dataset.

### CNN

This [folder](cnn) contains the script to train a CNN model. Details of this CNN architecture are written in our report [here](Report.pdf).

### LIME

This [folder](lime) contains the script to run LIME given a pre-trained CNN model. Please make sure to have a pre-trained model ready before running this script. We are currently still improving the style of this script - so please wait and see.

### GAN

This [folder](gan) contains the script to train a GAN model. Details of this GAN architecture are written in our report [here](Report.pdf).
