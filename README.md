This is the main branch for our project. In this branch we try to copy, as best we can, the model architechture and data used in the paper that we attempted to emulate. That being said, our strongest results appear in our two other branches where we were able to incorporate more data and different model architechtures to achieve much stronger results.

Branch Summaries:

main: direct replica of paper without any substantial changes. Model trained on around 300 images and achieved a maximum testing accuracy of around 92%
preprocessing_with_7X7X2048: The major difference in this branch from main is that we incorporated much more data(10 times as much). To further increase accuracy we didn't do any max pooling in the preprocessing and therefore, for each image, the preprocessing returned a tensor of shape(7,7,2048).
no_resnet50: In this branch, our preprocessing doesn't have any resnet and instead simply returns a 224 by 224 tensor representing the image itself. To then extract relevant features we used a series of convolutional networks and were able to achieve 95% accuracy with no fine tuning given that it took a very long time to run.

Code Organization:
The code is split up into three main files. main.py, model.py, and preprocess.py. The purpose of main is to call the preprocessing of the data, and then run, train, and test the model. The preprocessing is done with a series of different functions. First, we are able to match every image to a label of 1 or 0 for whether they had a tumor or not, randomize the order, and then extract the features using resnet50.
Once preprocessed, we can pass the 2048-D Tensor of features to the model. The model consists of a convolutional layyer, batch normalization, dropout, and then a couple of fully connected layers before using sigmoid activation to scale the result between 0 and 1. We use the adam optimizer, binary cross-entropy as the loss function, and a learning rate of 0.0001. That being said, we were able to see 92.16% accuracy which is still pretty impressive given how little data we were using.
