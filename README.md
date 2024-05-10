This branch contains the code for our own conv2d layers instead of the resnet50 for extracting features. Understandably this was much more computationally expensive than the models in the other branches, and therefore we were only able to fully run this once as it won't even run on my computer and we had to use a much better gpu. That being said we were able to acheive an accuracy of about 95% accuracy which we found surprising as we weren't able to experiment with any of the hyperparameters.

The structure of main.py is exactly the same except we have to format the features accordingly to have save (224,224,1). The only difference in preprocess.py is that we don't pass the images through resnet. For the model itself, in model.py, we added a series of 2 conv2d layers before flattening and passing the code through some dense layers. We believe that if we had more computing power and time to experiment with some hyperparameters, we would be able to see an accuracy very similar to that using RESNET50.

Accuracy: 95%
