import os
import numpy as np
import pandas as pd
import seaborn as sns
from glob import glob
import configparser
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, fbeta_score

from keras import metrics
from keras.optimizers import Adam
from keras import backend as K, Model
from keras.layers.merge import concatenate
from keras.layers.core import Dense, Dropout
from keras.layers import GlobalAveragePooling2D, Input
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import InceptionV3, ResNet50, VGG16, VGG19, Xception
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, ReduceLROnPlateau

from keras_script import get_image_generator, get_combined_generator, get_cnn_model


def get_matrix_from_gen(main_gen, model, split, name, net_id, output_dir, aux_gen=None):
    print('Creating matrix...')
    if aux_gen:
        print(f'aux_gen: {aux_gen.n}')
        y_pred = model.predict_generator(
            main_gen, steps=len(aux_gen), use_multiprocessing=False)
    else:
        aux_gen = main_gen
        y_pred = model.predict_generator(main_gen, use_multiprocessing=False, steps=len(aux_gen), verbose=1)
    y_pred = np.argmax(y_pred, axis=-1)
    y_true = aux_gen.classes

    print(y_pred.shape)
    print(y_true.shape)

    f2_score = fbeta_score(y_true, y_pred, beta=2, average='weighted')
    classes = list(aux_gen.class_indices.keys())

    cm = confusion_matrix(y_true, y_pred)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(10, 10))
    sns.heatmap(cm, annot=True, fmt='.2f',
                xticklabels=classes, yticklabels=classes)
    plt.yticks(rotation=0)
    plt.title(f'{net_id}{name}_{split} -> f2={f2_score:.3}')
    plt.savefig(f'{output_dir}/{net_id}{name}_{split}.png')


def test_on_images(network, images_dir, models_dir, *args):
    """"""
    print('Testing on images...')
    # Extract parameters from args
    img_width, img_height, batch_size, lr_rate, epochs, models_dir, logs_dir, figures_dir = args

    # Get image generators
    val_gen = get_image_generator(
        images_dir, 'val', img_width, img_height, batch_size)
    test_gen = get_image_generator(
        images_dir, 'test', img_width, img_height, batch_size)

    # Get network model for an image input shape
    input_shape = (img_width, img_height, 3)
    main_input = Input(shape=input_shape)
    base_model, last_layer_number = get_cnn_model(
        network, input_shape, main_input)

    # Make sure val/test have the same number of classes
    num_classes = len(np.unique(val_gen.classes))
    assert num_classes == len(np.unique(val_gen.classes))

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    model = Model(base_model.input, outputs=predictions)

    path = f'{models_dir}/{network}'
    models_list = glob(f'{path}/A*',)

    net_id = f'{os.path.basename(models_dir)}A_'
    print('Loading model...')
    for h5_model in models_list:
        model.load_weights(h5_model)
        print('Validation')
        get_matrix_from_gen(val_gen, model, 'val',
                            network, net_id, figures_dir)
        print('Test')
        get_matrix_from_gen(test_gen, model, 'test',
                            network, net_id, figures_dir)


def test_combined(network, images_dir, csv_dir, csv_data, models_dir, *args):
    # Extract parameters from args
    img_width, img_height, batch_size, lr_rate, epochs, models_dir, logs_dir, figures_dir = args

    # Get image generators
    multi_val_gen, val_gen, features = get_combined_generator(
        images_dir, csv_dir, csv_data, 'val', img_width, img_height, batch_size)
    multi_test_gen, test_gen, _ = get_combined_generator(
        images_dir, csv_dir, csv_data, 'test', img_width, img_height, batch_size)

    # Get network model for an image input shape
    input_shape = (img_width, img_height, 3)
    main_input = Input(shape=input_shape)
    base_model, last_layer_number = get_cnn_model(
        network, input_shape, main_input)

    # Make sure val/test have the same number of classes
    num_classes = len(np.unique(val_gen.classes))
    assert num_classes == len(np.unique(val_gen.classes))

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)

    # Load Simple MLP
    aux_input = Input(shape=(features,))
    aux = Dense(4096, activation='relu')(aux_input)
    aux = Dropout(0.3)(aux)
    aux = Dense(4096, activation='relu')(aux)
    aux = Dropout(0.3)(aux)
    aux = Dense(1024, activation='relu')(aux)

    # Merge input models
    merge = concatenate([x, aux])
    #merge = Dense(1024, activation='relu')(merge)
    #merge = Dropout(0.3)(merge)
    #merge = Dense(1024, activation='relu')(merge)
    predictions = Dense(8, activation='softmax')(merge)
    model = Model(inputs=[main_input, aux_input], outputs=predictions)

    path = f'{models_dir}/{network}'
    models_list = glob(f'{path}/B*',)
    net_id = f'{os.path.basename(models_dir)}B_'

    for h5_model in models_list:
        model.load_weights(h5_model)
        get_matrix_from_gen(multi_val_gen, model, 'val',
                            network, net_id, figures_dir, val_gen)
        get_matrix_from_gen(multi_test_gen, model, 'test',
                            network, net_id, figures_dir, test_gen)


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
    figures_dir = config.get('OUTPUT', 'figures_dir')
    os.makedirs(f'{figures_dir}', exist_ok=True)

    for network in networks_list:

        args = [img_width, img_height, batch_size,
                lr_rate, epochs, models_dir, logs_dir, figures_dir]

        test_on_images(network, images_dir, models_dir, *args)
        test_combined(network, images_dir, csv_dir,
                      csv_data, models_dir, *args)
