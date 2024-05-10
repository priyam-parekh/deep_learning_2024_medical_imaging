import argparse
import tensorflow as tf
import pickle
import numpy as np

from model import TumorClassifierModel
'''
The parse arguments function is used to create default values for some of the different
hyperparameters used in our model.
'''
def parseArguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--data", default= "C:\dl\deep_learning_2024_medical_imaging\medical_data\data.p")
    parser.add_argument("--num_epochs", type=int, default=6)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    args = parser.parse_args()
    return args
def train(model,train_inputs,train_labels,args):
    '''
    This function trains our entire model given the train_inputs and labels. We
    use adam optimizer and binary_crosentropy as the loss.
    '''
    model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
    history = model.fit(train_inputs,train_labels,epochs=args.num_epochs,batch_size=args.batch_size)
    return history
def test(model,test_inputs,test_labels):
    '''
    The test function evaluates our model given the test_inputs and test_labels
    '''
    loss,accuracy = model.evaluate(test_inputs,test_labels)
    print(f'Test Loss: {loss:.4f}')
    print(f'Test Accuracy: {accuracy:.4f}')
    return loss,accuracy
def main(args):
    with open(args.data,'rb') as data_file:
        data_dict = pickle.load(data_file)
    train_labels = np.array(data_dict['train_labels'])
    print(train_labels.shape)
    train_features = np.squeeze(np.array(data_dict['train_images']))
    print(train_features.shape)
    test_labels = np.array(data_dict['test_labels'])
    test_features = np.squeeze(np.array(data_dict['test_images']))
    model =TumorClassifierModel()#instantiate model and then train and test
    train(model,train_inputs=train_features,train_labels=train_labels,args=args)
    test(model,test_inputs=test_features,test_labels=test_labels)

if __name__ == '__main__':
    main(parseArguments())
