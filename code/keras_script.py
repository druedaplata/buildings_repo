import os
import pathlib
import numpy as np
import pandas as pd
import configparser
import tensorflow as tf
from keras import backend as K
from keras.models import Model
from keras.layers import GlobalAveragePooling2D, Input
from keras.layers.merge import concatenate

from keras.utils import multi_gpu_model
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.core import Dense, Dropout
from keras.optimizers import Adam
from keras.applications import InceptionV3, VGG16, VGG19, Xception, ResNet50
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, ReduceLROnPlateau
from sklearn.utils import class_weight
import keras_metrics as km

def setup_dirs(models_dir, logs_dir, networks_list):
    for net in networks_list:
        os.makedirs(f'{models_dir}/{net}', exist_ok=True)

    os.makedirs(f'{logs_dir}', exist_ok=True)


def get_image_generator(images_dir, split, *args):
    img_width, img_height, batch_size = args
    datagen = ImageDataGenerator(
        horizontal_flip=True,
        featurewise_center=True,
        featurewise_std_normalization=True,
        zoom_range=0.1,
        channel_shift_range=0.2,
        rescale=1./255
    )
    # Estimated manually from training dataset
    datagen.mean = np.array([99.74246, 107.59523, 110.0832], dtype=np.float32).reshape((1,1,3)) # ordering: [R, G, B]
    datagen.std = 56.58319

    generator = datagen.flow_from_directory(
        f'{images_dir}/{split}',
        target_size=(img_width, img_height),
        batch_size=batch_size,
        shuffle=True,
        class_mode='categorical'
    )

    return generator


def get_combined_generator(images_dir, csv_dir, csv_data, split, *args):
    """
    Creates train/val generators on images and csv data.

    Arguments:

    images_dir : string
        Path to a directory with subdirectories for each class.

    csv_dir : string
        Path to a directory containing train/val csv files.

    csv_data : list
        List of columns to use when training.
        First value is the index.
    """
    img_width, img_height, batch_size, gpu_number = args
    datagen = ImageDataGenerator(
        horizontal_flip=True,
        featurewise_center=True,
        featurewise_std_normalization=True,
        zoom_range=0.1,
        channel_shift_range=0.2,
        rescale=1./255
    )

    # Estimated manually from training dataset
    datagen.mean = np.array([99.74246, 107.59523, 110.0832], dtype=np.float32).reshape((1,1,3)) # ordering: [R, G, B]
    datagen.std = 56.58319

    generator = datagen.flow_from_directory(
        f'{images_dir}/{split}',
        target_size=(img_width, img_height),
        batch_size=batch_size,
        shuffle=True,
        class_mode='categorical'
    )

    # TODO: Change index to something more default
    df = pd.read_csv(f'{csv_dir}/{split}.csv',
                     usecols=csv_data, index_col=csv_data[0])

    def my_generator(image_gen, data):
        while True:
            i = image_gen.batch_index
            batch = image_gen.batch_size
            row = data[i*batch:(i+1)*batch]
            images, labels = image_gen.next()
            yield [images, row], labels

    csv_generator = my_generator(generator, df)

    _, features = df.shape
    return csv_generator, generator, features


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
        'inceptionV3': InceptionV3(input_shape=input_shape, weights='imagenet', include_top=False, input_tensor=main_input),
        'vgg16': VGG16(input_shape=input_shape, weights='imagenet', include_top=False, input_tensor=main_input),
        'vgg19': VGG16(input_shape=input_shape, weights='imagenet', include_top=False, input_tensor=main_input),
        'xception': Xception(input_shape=input_shape, weights='imagenet', include_top=False, input_tensor=main_input),
        'resnet50': ResNet50(input_shape=input_shape, weights='imagenet', include_top=False, input_tensor=main_input)}

    base_model = models[network]
    if network == 'inceptionV3':
        return base_model, 249
    else:
        return base_model, len(base_model.layers)

# Define personal metric
def f2_score(y_true, y_pred):
    beta = 2
    threshold_shift = 0

    y_pred = K.clip(y_pred, 0, 1)

    y_pred_bin = K.round(y_pred + threshold_shift)

    tp = K.sum(K.round(y_true * y_pred_bin)) + K.epsilon()
    fp = K.sum(K.round(K.clip(y_pred_bin - y_true, 0, 1)))
    fn = K.sum(K.round(K.clip(y_true - y_pred, 0, 1)))

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    beta_squared = beta ** 2
    return (beta_squared + 1) * (precision * recall) / (beta_squared * precision + recall + K.epsilon())
    

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
        ModelCheckpoint(f'{models_dir}/{network}/{path}.h5',
                        monitor='val_loss', verbose=1, save_best_only=True),
        EarlyStopping(monitor='val_loss', patience=10, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', patience=5, verbose=1),
        TensorBoard(log_dir=f'{logs_dir}/{network}/{path}')]
    return callback_list

def focal_loss(gamma=2., alpha=4.):

    gamma = float(gamma)
    alpha = float(alpha)

    def focal_loss_fixed(y_true, y_pred):
        """Focal loss for multi-classification
        FL(p_t)=-alpha(1-p_t)^{gamma}ln(p_t)
        Notice: y_pred is probability after softmax
        gradient is d(Fl)/d(p_t) not d(Fl)/d(x) as described in paper
        d(Fl)/d(p_t) * [p_t(1-p_t)] = d(Fl)/d(x)
        Focal Loss for Dense Object Detection
        https://arxiv.org/abs/1708.02002

        Arguments:
            y_true {tensor} -- ground truth labels, shape of [batch_size, num_cls]
            y_pred {tensor} -- model's output, shape of [batch_size, num_cls]

        Keyword Arguments:
            gamma {float} -- (default: {2.0})
            alpha {float} -- (default: {4.0})

        Returns:
            [tensor] -- loss.
        """
        epsilon = 1.e-9
        y_true = tf.convert_to_tensor(y_true, tf.float32)
        y_pred = tf.convert_to_tensor(y_pred, tf.float32)

        model_out = tf.add(y_pred, epsilon)
        ce = tf.multiply(y_true, -tf.log(model_out))
        weight = tf.multiply(y_true, tf.pow(tf.subtract(1., model_out), gamma))
        fl = tf.multiply(alpha, tf.multiply(weight, ce))
        reduced_fl = tf.reduce_max(fl, axis=1)
        return tf.reduce_mean(reduced_fl)
    return focal_loss_fixed   


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
    train_gen = get_image_generator(
        images_dir, 'train', img_width, img_height, batch_size)
    val_gen = get_image_generator(
        images_dir, 'val', img_width, img_height, batch_size)

    # Get network model for an image input shape
    input_shape = (img_width, img_height, 3)
    main_input = Input(shape=input_shape)
    base_model, last_layer_number = get_cnn_model(
        network, input_shape, main_input)

    # Make sure train/val have the same number of classes
    num_classes = len(np.unique(train_gen.classes))
    assert num_classes == len(np.unique(val_gen.classes))

    # Create class weights, useful for imbalanced datasets
    #class_weights = class_weight.compute_class_weight(
    #    'balanced',
    #    np.unique(train_gen.classes),
    #    train_gen.classes
    #)

    class_weights = {0:48, 1:1, 2:1, 3:27, 4:30, 5:39, 6:12, 7:1}

    # Get network model and change last layers for training
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(num_classes, activation='softmax')(x)

    # Create model object in keras
    model = Model(base_model.input, predictions)

    # Create path to save training models and logs
    top_weights_path = f'A_{network}'

    # Use a multi-gpu model if available and configured
    if gpu_number > 1:
        model = multi_gpu_model(model, gpus=gpu_number)

    # Define focal loss function


    # Compile model and set learning rate
    model.compile(loss='categorical_crossentropy', 
                  optimizer=Adam(lr=lr_rate),
                  metrics=['accuracy', km.categorical_f1_score])

    # Get list of training parameters in keras
    callback_list = get_callback_list(
        network, top_weights_path, models_dir, logs_dir)

    # Train the model on train split, for half the epochs
    model.fit_generator(
        train_gen,
        steps_per_epoch=train_gen.n // batch_size,
        epochs=epochs//2,
        validation_data=val_gen,
        validation_steps=val_gen.n // batch_size,
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
    model.compile(loss='categorical_crossentropy', 
                  optimizer=Adam(lr=lr_rate),
                  metrics=['accuracy', km.categorical_f1_score])

    # Train the model on train split, for the second half epochs
    model.fit_generator(
        train_gen,
        steps_per_epoch=train_gen.n // batch_size,
        epochs=epochs//2,
        validation_data=val_gen,
        validation_steps=val_gen.n // batch_size,
        class_weight=class_weights,
        callbacks=callback_list,
        use_multiprocessing=True)

    # Create path to save the model
    model.save(f'{models_dir}/{network}/{top_weights_path}.h5')


def train_combined(network, images_dir, csv_dir, csv_data, *args):
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
    multi_train_gen, train_gen, features = get_combined_generator(
        images_dir, csv_dir, csv_data, 'train', img_width, img_height, batch_size, gpu_number)
    multi_val_gen, val_gen, _ = get_combined_generator(
        images_dir, csv_dir, csv_data, 'val', img_width, img_height, batch_size, gpu_number)

    # Get network model for an image input shape
    input_shape = (img_width, img_height, 3)
    main_input = Input(shape=input_shape)
    base_model, last_layer_number = get_cnn_model(
        network, input_shape, main_input)

    # Make sure train/val have the same number of classes
    num_classes = len(np.unique(train_gen.classes))
    assert num_classes == len(np.unique(val_gen.classes))

    # Create class weights, useful for imbalanced datasets
    #class_weights = class_weight.compute_class_weight(
    #    'balanced',
    #    np.unique(train_gen.classes),
    #    train_gen.classes
    #)

    class_weights = {0:48, 1:1, 2:1, 3:27, 4:30, 5:39, 6:12, 7:1}
    

    # Get network model and change last layers for training
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)

    # Create MLP using features from csv files
    aux_input = Input(shape=(features,))
    aux = Dense(4096, activation='relu')(aux_input)
    aux = Dropout(0.5)(aux)
    aux = Dense(4096, activation='relu')(aux)
    aux = Dropout(0.5)(aux)
    aux = Dense(1024, activation='relu')(aux)

    # Merge both networks
    # TODO: Test with different number of layers after merge.
    merge = concatenate([x, aux])
    predictions = Dense(num_classes, activation='softmax')(merge)

    # Create model object in keras for both types of inputs
    model = Model(inputs=[main_input, aux_input], outputs=predictions)

    # Use a multi-gpu model if available and configured
    if gpu_number > 1:
        model = multi_gpu_model(model, gpus=gpu_number)

    # Create path to save training models and logs
    top_weights_path = f'B_{network}'

    # Compile model and set learning rate
    model.compile(loss='categorical_crossentropy', 
                  optimizer=Adam(lr=lr_rate),
                  metrics=['accuracy', km.categorical_f1_score])

    # Get list of training parameters in keras
    callback_list = get_callback_list(
        network, top_weights_path, models_dir, logs_dir)

    # Train the model on train split, for half the epochs
    model.fit_generator(
        multi_train_gen,
        steps_per_epoch=train_gen.n // batch_size,
        epochs=epochs//2,
        validation_data=multi_val_gen,
        validation_steps=val_gen.n // batch_size,
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
    model.compile(loss='categorical_crossentropy', 
                  optimizer=Adam(lr=lr_rate),
                  metrics=['accuracy', km.categorical_f1_score])

    # Train the model on train split, for the second half epochs
    model.fit_generator(
        multi_train_gen,
        steps_per_epoch=train_gen.n // batch_size,
        epochs=epochs//2,
        validation_data=multi_val_gen,
        validation_steps=val_gen.n // batch_size,
        class_weight=class_weights,
        callbacks=callback_list,
        use_multiprocessing=True
    )

    # Create path to save the model
    model.save(f'{models_dir}/{network}/{top_weights_path}.h5')


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

        args = [img_width, img_height, batch_size,
                lr_rate, epochs, models_dir, logs_dir, gpu_number]

        train_on_images(network, images_dir, *args)
        train_combined(network, images_dir, csv_dir, csv_data, *args)
