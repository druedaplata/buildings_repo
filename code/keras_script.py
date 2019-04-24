import os
import pathlib
import random
import numpy as np
import pandas as pd
import configparser

from glob import glob
import tensorflow as tf
from keras import backend as K
from keras.models import Model
from keras.preprocessing import image as krs_image
from keras.layers import GlobalAveragePooling2D, Input
from keras.layers.merge import concatenate, add, multiply, average

import imgaug as ia
from imgaug import augmenters as iaa

from keras.utils import multi_gpu_model
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.core import Dense, Dropout
from keras.optimizers import Adam, SGD, Adadelta
from keras.applications import InceptionV3, VGG16, VGG19, Xception, ResNet50
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, ReduceLROnPlateau
from sklearn.utils import class_weight
import keras_metrics as km


def setup_dirs(models_dir, logs_dir, networks_list):
    for net in networks_list:
        os.makedirs(f'{models_dir}/{net}', exist_ok=True)

    os.makedirs(f'{logs_dir}', exist_ok=True)


def sometimes(aug): return iaa.Sometimes(0.5, aug)


# Create preprocessing pipeline
preprocessing = iaa.Sequential(
    [
        # Apply the following to most images
        iaa.Fliplr(0.6),
        iaa.GaussianBlur(0.7),
        # Crop images by -5% and 10% of height/width
        sometimes(iaa.Crop(percent=0.15)),

        iaa.ChannelShuffle(0.5, [1, 0, 1]),

        iaa.SomeOf((1, 3),
                   [
            iaa.Affine(rotate=(-15, 15)),
            #iaa.CoarseSaltAndPepper(p=0.1, size_percent=0.07),
            iaa.Affine(shear=(-15, 15))

        ],
            random_order=True),
    ]
)


def get_image_generator(images_dir, split, *args):
    img_width, img_height, batch_size = args

    image_file_list = glob(f'{images_dir}/{split}/**/*.JPG', recursive=True)
    dirs = sorted(os.listdir(f'{images_dir}/{split}'))
    num_classes = len(dirs)
    num_images = len(image_file_list)
    classes = {v: i for i, v in enumerate(dirs)}
    random.shuffle(image_file_list)

    datagen = ImageDataGenerator()

    def image_generator(images_list, batch_size):
        i = 0
        while True:
            batch = {'images': [], 'labels': []}
            for b in range(batch_size):
                if i == len(images_list):
                    i = 0
                    random.shuffle(images_list)
                # Load image
                image_path = images_list[i]
                image = krs_image.load_img(
                    image_path, target_size=(img_height, img_width))
                image = krs_image.img_to_array(image)
                # Get label from path
                label = classes[image_path.split('/')[-2]]

                i += 1
                batch['images'].append(image)
                batch['labels'].append(label)

            # Convert images to batch form
            batch['images'] = np.array(batch['images'], dtype=np.uint8)
            # Standardize images in batch
            batch['images'] = datagen.standardize(batch['images'])
            # Apply preprocessing to image batch
            batch['images'] = preprocessing.augment_images(batch['images'])

            batch['labels'] = np.eye(len(dirs))[batch['labels']]

            yield batch['images'], batch['labels']

    generator = image_generator(image_file_list, batch_size)

    return num_images, num_classes, generator


def get_combined_generator(images_dir, csv_dir, csv_data, split, *args):

    img_width, img_height, batch_size = args

    image_file_list = glob(f'{images_dir}/{split}/**/*.JPG', recursive=True)
    dirs = sorted(os.listdir(f'{images_dir}/{split}'))
    num_classes = len(dirs)
    num_images = len(image_file_list)
    classes = {v: i for i, v in enumerate(dirs)}
    random.shuffle(image_file_list)

    df = pd.read_csv(f'{csv_dir}/{split}.csv', index_col=csv_data[0])

    datagen = ImageDataGenerator()

    def my_generator(images_list, dataframe, batch_size):
        i = 0
        while True:
            batch = {'images': [], 'csv': [], 'labels': []}
            for b in range(batch_size):
                if i == len(images_list):
                    i = 0
                    random.shuffle(images_list)

                # Load image
                image_path = images_list[i]
                # Remove last .JPG from image, generated images have two.
                image_name = os.path.basename(image_path)[:-4]
                image = krs_image.load_img(
                    image_path, target_size=(img_height, img_width))
                image = krs_image.img_to_array(image)

                i += 1
                # Get label from CSV
                csv_attrs = dataframe.loc[image_name, :]
                label = classes[csv_attrs['clase']]
                csv_attrs = csv_attrs.drop(labels='clase')

                batch['images'].append(image)
                batch['csv'].append(csv_attrs)
                batch['labels'].append(label)

            # Convert images to batch form
            batch['images'] = np.array(batch['images'], dtype=np.uint8)
            # Standardize images in batch
            batch['images'] = datagen.standardize(batch['images'])
            # Apply preprocessing to image batch
            batch['images'] = preprocessing.augment_images(batch['images'])

            batch['csv'] = np.array(batch['csv'])
            batch['labels'] = np.eye(len(dirs))[batch['labels']]

            yield [batch['images'], batch['csv']], batch['labels']

    csv_generator = my_generator(image_file_list, df, batch_size)

    _, features = df.shape
    # Minus the index column
    features -= 1

    return num_images, num_classes, features, csv_generator


def get_cnn_model(network, input_shape, main_input, *args):
    """
    Returns a convolutional neural network model with imagenet weights.

    Arguments:

    network : string
              Name of a predefined network must be implemented.

    input_shape : tuple
                  Three values with image width, height, and channels
                  (img_width, img_height, channels)

    main_input : Input
                 Input object using the input shape defined, redundancy for a bug in keras.

    """
    models = {
        'inceptionV3':
        InceptionV3(
            input_shape=input_shape,
            weights='imagenet',
            include_top=False,
            input_tensor=main_input),
        'vgg16':
        VGG16(
            input_shape=input_shape,
            weights='imagenet',
            include_top=False,
            input_tensor=main_input),
        'vgg19':
        VGG16(
            input_shape=input_shape,
            weights='imagenet',
            include_top=False,
            input_tensor=main_input),
        'xception':
        Xception(
            input_shape=input_shape,
            weights='imagenet',
            include_top=False,
            input_tensor=main_input),
        'resnet50':
        ResNet50(
            input_shape=input_shape,
            weights='imagenet',
            include_top=False,
            input_tensor=main_input)
    }

    base_model = models[network]
    if network == 'inceptionV3':
        return base_model, 249
    else:
        return base_model, len(base_model.layers)


def get_image_model(network, num_classes, img_width, img_height):
    """Returns a model that uses only images as input.

    Arguments:
        network {string} -- Name of the network trained on imagenet
        num_classes {int} -- Number of classes in the dataset
        img_width {int} -- Image width
        img_height {int} -- Image height

    Returns:
        Keras Model -- Keras Model ready to compile.
    """
    input_shape = (img_width, img_height, 3)
    image_input = Input(shape=input_shape)
    base_model, last_layer_number = get_cnn_model(
        network, input_shape, image_input)

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    predictions = Dense(num_classes, activation='softmax')(x)

    return Model(base_model.input, predictions), last_layer_number


def get_csv_plus_image_model(network, num_classes, features, img_width, img_height, merge_type='concat'):
    input_shape = (img_width, img_height, 3)
    image_input = Input(shape=input_shape)
    base_model, last_layer_number = get_cnn_model(
        network, input_shape, image_input)

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    # Use a Dense layer after GAP to make sure shape is the same on merge, no matter which network is running.
    x = Dense(512, activation='relu')(x)

    # Create MLP using features from csv files
    aux_input = Input(shape=(features, ))
    # 256, 128, 64
    aux = Dense(512, activation='relu')(aux_input)
    aux = Dropout(0.3)(aux)
    aux = Dense(512, activation='relu')(aux)
    aux = Dropout(0.3)(aux)
    aux = Dense(512, activation='relu')(aux)

    # Merge both inputs
    if merge_type == 'concat':
        merge = concatenate([x, aux])
    elif merge_type == 'add':
        merge = add([x, aux])
    elif merge_type == 'mul':
        merge = multiply([x, aux])
    elif merge_type == 'avg':
        merge = average([x, aux])
    predictions = Dense(num_classes, activation='softmax')(merge)

    return Model(inputs=[image_input, aux_input], output=predictions), last_layer_number


def get_callback_list(network, path, models_dir, logs_dir):
    """
    Returns a list of parameters for training in keras.

    Arguments
        network : string
            Name of an implemented network
        path : string
            Filename to store the logs and models while training
        models_dir : string
            Path to folder where models are saved.
        logs_dir : string
            Path to folder where logs are saved.
    """
    callback_list = [
        ModelCheckpoint(
            f'{models_dir}/{network}/{path}.h5',
            monitor='val_loss',
            verbose=1,
            save_best_only=True,
            mode='min'),
        EarlyStopping(monitor='val_loss', patience=50, verbose=1),
        #ReduceLROnPlateau(monitor='val_loss', patience=4, verbose=1),
        TensorBoard(log_dir=f'{logs_dir}/{network}/{path}')
    ]
    return callback_list


def train_on_images(network, images_dir, *args):
    """
    Trains a convolutional neural network on images from images_dir.

    Arguments:
        network : string
            Name of an implemented CNN on current keras version.
        images_dir : string
            Path to a directory with subdirs for each image class.
    """
    # Extract parameters from args
    img_width, img_height, batch_size, lr_rate, epochs, models_dir, logs_dir, gpu_number = args

    # Get image generators
    num_images_train, num_classes_train, train_gen = get_image_generator(
        images_dir, 'train', img_width, img_height, batch_size)

    num_images_val, num_classes_val, val_gen = get_image_generator(
        images_dir, 'val', img_width, img_height, batch_size)

    # Make sure train/val have the same number of classes
    assert num_classes_train == num_classes_val

    # Create class weights, useful for imbalanced datasets
    # class_weights = class_weight.compute_class_weight(
    #    'balanced', np.unique(train_gen.classes), train_gen.classes)

    if num_classes_train == 8:
        class_weights = {0: 50, 1: 1, 2: 1, 3: 50, 4: 50, 5: 50, 6: 50, 7: 1}
    if num_classes_train == 7:
        class_weights = {0: 50, 1: 1, 2: 1, 3: 50, 4: 50, 5: 50, 6: 1}

    # Get image model
    model, last_layer_number = get_image_model(
        network, num_classes_train, img_width, img_height)

    # Create path to save training models and logs
    top_weights_path = f'A_{network}'

    # Use a multi-gpu model if available and configured
    if gpu_number > 1:
        model = multi_gpu_model(model, gpus=gpu_number)

    # Compile model and set learning rate
    model.compile(
        loss='categorical_crossentropy',
        optimizer=Adadelta(lr=lr_rate),
        metrics=[
            'accuracy',
            km.categorical_f1_score()
        ])

    # Get list of training parameters in keras
    callback_list = get_callback_list(
        network,
        top_weights_path,
        models_dir,
        logs_dir,
    )

    # Train the model on train split, for half the epochs
    model.fit_generator(
        train_gen,
        steps_per_epoch=num_images_train // batch_size,
        epochs=epochs // 2,
        validation_data=val_gen,
        validation_steps=num_images_val // batch_size,
        class_weight=class_weights,
        callbacks=callback_list,
        use_multiprocessing=True)

    # Load the best model from previous training phase
    model.load_weights(f'{models_dir}/{network}/{top_weights_path}.h5')

    # After training for a few epochs, freeze the bottom layers, and train only the last ones.
    for layer in model.layers[:last_layer_number]:
        layer.trainable = False
    for layer in model.layers[last_layer_number:]:
        layer.trainable = True

    # Compile model with frozen layers, and set learning rate
    model.compile(
        loss='categorical_crossentropy',
        optimizer=Adadelta(lr=lr_rate),
        metrics=[
            'accuracy',
            km.categorical_f1_score()
        ])

    # Train the model on train split, for the second half epochs
    model.fit_generator(
        train_gen,
        steps_per_epoch=num_images_train // batch_size,
        epochs=epochs // 2,
        validation_data=val_gen,
        validation_steps=num_images_val // batch_size,
        class_weight=class_weights,
        callbacks=callback_list,
        use_multiprocessing=True)

    # Create path to save the model
    # model.save(f'{models_dir}/{network}/{top_weights_path}.h5')


def train_combined(network, images_dir, csv_dir, csv_data, merge_type, *args):
    """
    Trains a network combining a convolutional network and a multilayer perceptron
    on images and csv data.

    Arguments:
        network : string
            Name of an implemented CNN on current keras version.
        images_dir : string
            Path to a directory with subdirs for each image class.
        csv_dir : string
            Path to a directory that containts train/val csv files.
    """
    # Extract parameters from args
    img_width, img_height, batch_size, lr_rate, epochs, models_dir, logs_dir, gpu_number = args

    # Get combined and image generators, and number of features in csv files.
    num_images_train, num_classes_train, features, multi_train_gen = get_combined_generator(
        images_dir, csv_dir, csv_data, 'train', img_width, img_height, batch_size)

    num_images_val, num_classes_val, features, multi_val_gen = get_combined_generator(
        images_dir, csv_dir, csv_data, 'val', img_width, img_height, batch_size
    )

    # Make sure train/val have the same number of classes
    assert num_classes_train == num_classes_val

    # Create class weights, useful for imbalanced datasets
    # class_weights = class_weight.compute_class_weight(
    #    'balanced', np.unique(train_gen.classes), train_gen.classes)

    if num_classes_train == 8:
        class_weights = {0: 50, 1: 1, 2: 1, 3: 50, 4: 50, 5: 50, 6: 50, 7: 1}
    if num_classes_train == 7:
        class_weights = {0: 50, 1: 1, 2: 1, 3: 50, 4: 50, 5: 50, 6: 1}

    # Create model object in keras for both types of inputs
    model, last_layer_number = get_csv_plus_image_model(
        network, num_classes_train, features, img_height, img_height, merge_type)

    # Use a multi-gpu model if available and configured
    if gpu_number > 1:
        model = multi_gpu_model(model, gpus=gpu_number)

    # Create path to save training models and logs
    top_weights_path = f'B_{merge_type}_{network}'

    # Compile model and set learning rate
    model.compile(
        loss='categorical_crossentropy',
        optimizer=Adadelta(lr=lr_rate),
        metrics=[
            'accuracy',
            km.categorical_f1_score()
        ])

    # Get list of training parameters in keras
    callback_list = get_callback_list(
        network,
        top_weights_path,
        models_dir,
        logs_dir,
    )

    # Train the model on train split, for half the epochs
    model.fit_generator(
        multi_train_gen,
        steps_per_epoch=num_images_train // batch_size,
        epochs=epochs // 2,
        validation_data=multi_val_gen,
        validation_steps=num_images_val // batch_size,
        class_weight=class_weights,
        callbacks=callback_list,
        use_multiprocessing=True)

    # Load the best model from previous training phase
    model.load_weights(f'{models_dir}/{network}/{top_weights_path}.h5')

    # After training for a few epochs, freeze the bottom layers, and train only the last ones.
    for layer in model.layers[:last_layer_number]:
        layer.trainable = False
    for layer in model.layers[last_layer_number:]:
        layer.trainable = True

    # Compile model with frozen layers, and set learning rate
    model.compile(
        loss='categorical_crossentropy',
        optimizer=Adadelta(lr=lr_rate),
        metrics=[
            'accuracy',
            km.categorical_f1_score()
        ])

    # Train the model on train split, for the second half epochs
    model.fit_generator(
        multi_train_gen,
        steps_per_epoch=num_images_train // batch_size,
        epochs=epochs // 2,
        validation_data=multi_val_gen,
        validation_steps=num_images_val // batch_size,
        class_weight=class_weights,
        callbacks=callback_list,
        use_multiprocessing=True)

    # Create path to save the model
    # model.save(f'{models_dir}/{network}/{top_weights_path}.h5')


if __name__ == '__main__':

    config = configparser.ConfigParser()
    config.read('config.ini')

    # Read image parameters
    images_dir = config.get('IMAGES', 'images_dir')
    img_width = config.getint('IMAGES', 'width')
    img_height = config.getint('IMAGES', 'height')
    # Read csv parameters
    csv_dir = config.get('CSV', 'csv_dir')
    csv_data = config.get('CSV', 'csv_data').split(',')
    # Read training parameters
    lr_rate = config.getfloat('TRAINING', 'lr_rate')
    batch_size = config.getint('TRAINING', 'batch_size')
    epochs = config.getint('TRAINING', 'epochs')
    networks_list = config.get('TRAINING', 'cnn_network_list').split(',')
    gpu_number = config.getint('TRAINING', 'gpu_number')
    batch_size = batch_size * gpu_number
    # Read data paths
    models_dir = config.get('OUTPUT', 'models_dir')
    logs_dir = config.get('OUTPUT', 'logs_dir')

    setup_dirs(models_dir, logs_dir, networks_list)

    for network in networks_list:

        args = [
            img_width, img_height, batch_size, lr_rate, epochs, models_dir,
            logs_dir, gpu_number
        ]

        #train_on_images(network, images_dir, *args)
        for merge_type in ['add', 'avg', 'mul', 'concat']:
            train_combined(network, images_dir, csv_dir,
                           csv_data, merge_type, *args)
