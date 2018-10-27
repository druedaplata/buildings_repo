import numpy as np
import pandas as pd
import configparser
from keras import backend as K
from keras.models import Model
from keras.layers import GlobalAveragePooling2D, Input
from keras.layers.merge import concatenate

from keras.preprocessing.image import ImageDataGenerator
from keras.layers.core import Dense, Dropout
from keras.optimizers import Adam
from keras.applications import InceptionV3, VGG16, VGG19, Xception, ResNet50
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, ReduceLROnPlateau
from sklearn.utils import class_weight


def get_image_generators(images_dir, *args):
    """"""
    img_width, img_height, batch_size = args
    datagen = ImageDataGenerator()

    gen_train = datagen.flow_from_directory(
        f'{images_dir}/train',
        target_size=(img_width, img_height),
        batch_size=batch_size,
        shuffle=True,
        class_mode='categorical'
    )

    gen_val = datagen.flow_from_directory(
        f'{images_dir}/val',
        target_size=(img_width, img_height),
        batch_size=batch_size,
        shuffle=True,
        class_mode='categorical'
    )

    return gen_train, gen_val


def get_combined_generators(images_dir, csv_dir, csv_index, *args):
    """"""
    img_width, img_height, batch_size = args
    datagen = ImageDataGenerator()

    gen_train = datagen.flow_from_directory(
        f'{images_dir}/train',
        target_size=(img_width, img_height),
        batch_size=batch_size,
        shuffle=True,
        class_mode='categorical'
    )

    gen_val = datagen.flow_from_directory(
        f'{images_dir}/val',
        target_size=(img_width, img_height),
        batch_size=batch_size,
        shuffle=True,
        class_mode='categorical'
    )

    # TODO: Change index to something more default
    train_df = pd.read_csv(f'{csv_dir}/train.csv',
                           header=None, index_col=csv_index)
    val_df = pd.read_csv(f'{csv_dir}/val.csv',
                         header=None, index_col=csv_index)

    def my_generator(image_gen, data):
        while True:
            i = image_gen.batch_index
            batch = image_gen.batch_size
            row = data[i*batch:(i+1)*batch]
            images, labels = image_gen.next()
            yield [images, row], labels

    csv_train_gen = my_generator(gen_train, train_df)
    csv_val_gen = my_generator(gen_val, val_df)

    _, features = train_df.shape
    return csv_train_gen, csv_val_gen, gen_train, gen_val, features


def get_cnn_model(network, input_shape):
    models = {
        'inceptionV3': InceptionV3(input_shape=input_shape, weights='imagenet', include_top=False, input_tensor=Input(shape=input_shape)),
        'vgg16': VGG16(input_shape=input_shape, weights='imagenet', include_top=False, input_tensor=Input(shape=input_shape)),
        'vgg19': VGG16(input_shape=input_shape, weights='imagenet', include_top=False, input_tensor=Input(shape=input_shape)),
        'xception': Xception(input_shape=input_shape, weights='imagenet', include_top=False, input_tensor=Input(shape=input_shape)),
        'resnet50': ResNet50(input_shape=input_shape, weights='imagenet', include_top=False, input_tensor=Input(shape=input_shape))}

    base_model = models[network]
    if network == 'inceptionV3':
        return base_model, 249
    else:
        return base_model, len(base_model.layers)


def get_callback_list(network, path, models_dir, logs_dir):
    callback_list = [
        ModelCheckpoint(f'{models_dir}/{network}/{path}.h5',
                        monitor='val_loss', verbose=1, save_best_only=True),
        EarlyStopping(monitor='val_loss', patience=30, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', patience=30, verbose=1),
        TensorBoard(log_dir=f'{logs_dir}/{network}/{path}')]
    return callback_list


def train_on_images(network, images_dir, *args):

    img_width, img_height, batch_size, lr_rate, epochs, models_dir, logs_dir = args

    train_gen, val_gen = get_image_generators(images_dir)
    input_shape = (img_width, img_height, 3)
    base_model, last_layer_number = get_cnn_model(network, input_shape)

    num_classes = len(np.unique(train_gen.classes))
    assert num_classes == len(np.unique(val_gen.classes))

    class_weights = class_weight.compute_class_weight(
        'balanced',
        np.unique(train_gen.classes),
        train_gen.classes
    )

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(num_classes, activation='softmax')(x)

    model = Model(base_model.input, predictions)
    top_weights_path = f'{network}_lr{lr_rate}_{batch_size}bs'

    opt = Adam(lr=lr_rate)
    model.compile(optimizer=opt, loss='categorical_crossentropy',
                  metrics=['accuracy'])

    """
    Directives for training
    """
    callback_list = get_callback_list(
        network, top_weights_path, models_dir, logs_dir)

    model.fit_generator(
        train_gen,
        steps_per_epoch=train_gen.n // batch_size,
        epochs=epochs//2,
        validation_data=val_gen,
        validation_steps=val_gen.n // batch_size,
        class_weight=class_weights,
        callbacks=callback_list)

    model.load_weights(f'{models_dir}/{network}/{top_weights_path}.h5')

    """
    After training for a few epochs, freeze the bottom layers, and train only on top.
    """

    for layer in model.layers[:last_layer_number]:
        layer.trainable = False
    for layer in model.layers[last_layer_number:]:
        layer.trainable = True

    opt = Adam(lr=lr_rate)
    model.compile(optimizer=opt,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit_generator(
        train_gen,
        steps_per_epoch=train_gen.n // batch_size,
        epochs=epochs//2,
        validation_data=val_gen,
        validation_steps=val_gen.n // batch_size,
        class_weight=class_weights,
        callbacks=callback_list
    )

    final_weights_path = f'{models_dir}/{network}/final_{network}_lr{lr_rate}_{batch_size}bs'
    model.save(final_weights_path)


def train_combined(network, images_dir, csv_dir, *args):

    img_width, img_height, batch_size, lr_rate, epochs, models_dir, logs_dir = args

    multi_train_gen, multi_val_gen, train_gen, val_gen, features = get_combined_generators(
        images_dir, csv_dir)
    input_shape = (img_width, img_height, 3)
    main_input = Input(shape=input_shape)
    base_model, last_layer_number = get_cnn_model(network, input_shape)

    num_classes = len(np.unique(train_gen.classes))
    assert num_classes == len(np.unique(val_gen.classes))

    class_weights = class_weight.compute_class_weight(
        'balanced',
        np.unique(train_gen.classes),
        train_gen.classes
    )

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)

    # Create simple NN
    aux_input = Input(shape=(features,))
    aux = Dense(4096, activation='relu')(aux_input)
    aux = Dropout(0.5)(aux)
    aux = Dense(4096, activation='relu')(aux)
    aux = Dropout(0.5)(aux)
    aux = Dense(1024, activation='relu')(aux)

    # Merge input models
    merge = concatenate([x, aux])
    predictions = Dense(num_classes, activation='softmax')(merge)

    model = Model(inputs=[main_input, aux_input], outputs=predictions)
    top_weights_path = f'multi_{network}_lr{lr_rate}_{batch_size}bs'

    opt = Adam(lr=lr_rate)
    model.compile(optimizer=opt, loss='categorical_crossentropy',
                  metrics=['accuracy'])

    """
    Directives for training
    """
    callback_list = get_callback_list(network, top_weights_path)

    model.fit_generator(
        multi_train_gen,
        steps_per_epoch=train_gen.n // batch_size,
        epochs=epochs//2,
        validation_data=multi_val_gen,
        validation_steps=val_gen.n // batch_size,
        class_weight=class_weights,
        callbacks=callback_list)

    model.load_weights(f'{models_dir}/{network}/{top_weights_path}.h5')

    """
    After training for a few epochs, freeze the bottom layers, and train only on top.
    """

    for layer in model.layers[:last_layer_number]:
        layer.trainable = False
    for layer in model.layers[last_layer_number:]:
        layer.trainable = True

    opt = Adam(lr=lr_rate)
    model.compile(optimizer=opt,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit_generator(
        multi_train_gen,
        steps_per_epoch=train_gen.n // batch_size,
        epochs=epochs//2,
        validation_data=multi_val_gen,
        validation_steps=val_gen.n // batch_size,
        class_weight=class_weights,
        callbacks=callback_list
    )

    final_weights_path = f'{models_dir}/{network}/final_multi_{network}_lr{lr_rate}_{batch_size}bs'
    model.save(final_weights_path)


if __name__ == '__main__':

    config = configparser.ConfigParser()
    config.read('config.ini')

    # Read image parameters
    images_dir = config.get('IMAGES', 'images_dir')
    img_width = config.getint('IMAGES', 'width')
    img_height = config.getint('IMAGES', 'height')
    # Read csv parameters
    csv_dir = config.get('CSV', 'csv_dir')
    csv_index = config.get('CSV', 'csv_index')
    # Read training parameters
    lr_rate = config.getfloat('TRAINING', 'lr_rate')
    batch_size = config.getint('TRAINING', 'batch_size')
    epochs = config.getint('TRAINING', 'epochs')
    networks_list = config.get('TRAINING', 'cnn_network_list')
    # Read data paths
    models_dir = config.get('OUTPUT', 'models_dir')
    logs_dir = config.get('OUTPUT', 'logs_dir')

    for network in networks_list:

        args = [img_width, img_height, batch_size,
                lr_rate, epochs, models_dir, logs_dir]

        train_on_images(network, images_dir, args)
        train_combined(network, images_dir, csv_dir, csv_index, args)
