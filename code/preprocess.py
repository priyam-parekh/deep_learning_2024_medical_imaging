import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split

def load_and_process_images(directory, target_size=(224, 224)):
    """
    Load images from the directory, resize them, and normalize the pixel values.
    :param directory: Path to the directory containing subdirectories for each class
    :param target_size: Tuple specifying the image dimensions
    :return: Tuple of numpy arrays (inputs, labels)
    """
    classes = os.listdir(directory)
    images = []
    labels = []

    for idx, label in enumerate(classes):
        class_dir = os.path.join(directory, label)
        for filename in os.listdir(class_dir):
            img_path = os.path.join(class_dir, filename)
            img = load_img(img_path, target_size=target_size)
            img = img_to_array(img)
            img = img / 255.0  
            images.append(img)
            labels.append(idx)
    
    return np.array(images), np.array(labels)

def get_data(data_dir):
    """
    Prepares training and test datasets.
    :param data_dir: Base directory of the dataset
    :return: Training and test datasets
    """
    images, labels = load_and_process_images(data_dir)
    train_images, test_images, train_labels, test_labels = train_test_split(
        images, labels, test_size=0.2, random_state=42
    )
    return train_images, test_images, train_labels, test_labels

if __name__ == '__main__':
    data_dir = 'medical_data' 
    train_images, test_images, train_labels, test_labels = get_data(data_dir)
    print("Data Loaded and Preprocessed")
    print("Train Images: ", train_images.shape)
    print("Test Images: ", test_images.shape)
    print("Train Labels: ", train_labels.shape)
    print("Test Labels: ", test_labels.shape)
