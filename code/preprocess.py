import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split
import pickle
from tqdm import tqdm
from PIL import Image

def extract_features(names,data_folder):
    '''
    Method used to extract the features from the images in the dataset using ResNet50
    '''
    image_features = []
    resnet = tf.keras.applications.ResNet50(False)  ## Produces Bx7x7x2048
    pbar = tqdm(names)
    for i, image_name in enumerate(pbar):
        img_path = f'{data_folder}/{image_name}'
        pbar.set_description(f"[({i+1}/{len(names)})] Processing '{img_path}' into 2048-D ResNet GAP Vector")
        with Image.open(img_path) as img:
            img = img.convert('L')
            img_array = np.array(img.resize((224,224)))
        img_in = tf.keras.applications.resnet50.preprocess_input(img_array)[np.newaxis, :]
        img_in = np.expand_dims(img_in,axis=-1)
        img_in = np.concatenate((img_in,img_in,img_in),axis=-1)
        image_features += [resnet(img_in)]
    print()
    return np.array(image_features) # returns an np array of number of images by features extracted.

def load_and_process_images(directory, target_size=(224, 224)):
    """
    Load images from the directory, resize them, and normalize the pixel values.
    :param directory: Path to the directory containing subdirectories for each class
    :param target_size: Tuple specifying the image dimensions
    :return: Tuple of numpy arrays (inputs, labels)
    """
    image_features = []
    labels = []
    no_directory = f'{directory}/no'
    no_image_names = os.listdir(no_directory)
    for __ in range(len(no_image_names)):
        labels.append(0)
    image_features = extract_features(no_image_names,no_directory)
    yes_directory = f'{directory}/yes'
    yes_image_names = os.listdir(yes_directory)
    for _ in range(len(yes_image_names)):
        labels.append(1)
    image_features = np.concatenate((image_features,extract_features(yes_image_names,yes_directory)))

    # add all the images in no to the images and add the corresponding label.
    labels = np.array(labels)
    indexes = np.arange(len(image_features))
    np.random.shuffle(indexes)
    image_features = image_features[indexes]
    labels = labels[indexes]
    print(len(labels))
    print(len(image_features))
    return np.array(image_features), np.array(labels)

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
    return dict(train_images = train_images, test_images = test_images, train_labels = train_labels, test_labels = test_labels)

def create_pickle(data_folder):
    with open(f'{data_folder}/data.p', 'wb') as pickle_file:
        pickle.dump(get_data(data_folder), pickle_file)
    print(f'Data has been dumped into {data_folder}/data.p!')
if __name__ == '__main__':
    # data_dir = '../medical_data' 
    data_dir = "C:\dl\deep_learning_2024_medical_imaging\medical_data"
    create_pickle(data_dir)
