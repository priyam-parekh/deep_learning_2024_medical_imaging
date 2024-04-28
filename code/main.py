import argparse
import tensorflow as tf
import pickle
import numpy as np

from model import TumorClassifierModel

def parseArguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=12)
    parser.add_argument("--data", default= "C:\dl\deep_learning_2024_medical_imaging\medical_data\data.p")
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    args = parser.parse_args()
    return args
def train(model,train_inputs,train_labels,args):
    model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
    # Define a callback function to print accuracy after each batch
    def print_batch_accuracy(epoch, logs):
        print(f'Epoch {epoch + 1} Accuracy: {logs["accuracy"]:.4f}')
    batch_accuracy_callback = tf.keras.callbacks.LambdaCallback(on_batch_end=print_batch_accuracy)

    history = model.fit(train_inputs,train_labels,epochs=args.num_epochs,batch_size=args.batch_size,callbacks=[batch_accuracy_callback])
    return history
def test(model,test_inputs,test_labels):
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
    model =TumorClassifierModel()
    train(model,train_inputs=train_features,train_labels=train_labels,args=args)
    test(model,test_inputs=test_features,test_labels=test_labels)
if __name__ == '__main__':
    main(parseArguments())
