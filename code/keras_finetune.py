import numpy as np
from keras import backend as K, Model
from keras.layers import GlobalAveragePooling2D

from keras.preprocessing.image import ImageDataGenerator
from keras.layers.core import Dense
from keras.optimizers import SGD
from keras.applications import InceptionV3
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard


def weighted_categorical_crossentropy(weights):
    """
    A weighted version of keras.objectives.categorical_crossentropy

    Variables:
        weights: numpy array of shape (C,) where C is the number of classes

    Usage:
        weights = np.array([0.5,2,10]) # Class one at 0.5, class 2 twice the normal weights, class 3 10x.
        loss = weighted_categorical_crossentropy(weights)
        model.compile(loss=loss,optimizer='adam')
    """

    weights = K.variable(weights)

    def loss(y_true, y_pred):
        # scale predictions so that the class probas of each sample sum to 1
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
        # clip to prevent NaN's and Inf's
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        # calc
        loss = y_true * K.log(y_pred) * weights
        loss = -K.sum(loss, -1)
        return loss

    return loss

def get_last_layer_number(network_name):
    layer_dict_info = {'inceptionV3':249, 'vgg16':0, 'vgg19':0}
    return layer_dict_info[network_name]

def get_generator(path_to_image_dir, img_width, img_height):
    datagen = ImageDataGenerator(
        horizontal_flip=True,
        rescale=1./255)

    generator = datagen.flow_from_directory(
        path_to_image_dir,
        target_size=(img_width, img_height),
        shuffle=True,
        class_mode='categorical')
    return generator

def train_inceptionV3(batch_size, num_classes, img_width=224, img_height=224, epochs=30, lr_rate=0.0001):
    train_generator = get_generator('../data/macro_dataset/train', img_width, img_height)
    val_generator = get_generator('../data/macro_dataset/val', img_width, img_height)

    weights = np.ones((2,))
    base_model_last_block_layer_number = get_last_layer_number('inceptionV3')

    base_model = InceptionV3(input_shape=(img_width, img_height, 3), weights='imagenet', include_top=False)

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(num_classes, activation='softmax')(x)

    model = Model(base_model.input, predictions)
    top_weights_path = 'models/top_inceptionV3_weights.h5'

    for layer in base_model.layers:
        layer.trainable = False

    opt = SGD(lr=lr_rate, decay=1e-5)
    model.compile(optimizer=opt,
                  loss=weighted_categorical_crossentropy(weights),
                  metrics=['accuracy'])

    """
    Directives for training
    """
    callback_list = [
        ModelCheckpoint(top_weights_path, monitor='val_acc', verbose=1, save_best_only=True),
        EarlyStopping(monitor='val_acc', patience=5, verbose=0),
        TensorBoard(log_dir='./logs/inceptionV3')
    ]

    model.fit_generator(train_generator,
                        steps_per_epoch=6812 // batch_size,
                        epochs=epochs//2,
                        validation_data=val_generator,
                        validation_steps=2270 // batch_size,
                        callbacks=callback_list)

    model.load_weights(top_weights_path)

    """
    After training for a few epochs, freeze the bottom layers, and train only on top.
    """

    for layer in model.layers[:base_model_last_block_layer_number]:
        layer.trainable = False
    for layer in model.layers[base_model_last_block_layer_number:]:
        layer.trainable = True

    opt = SGD(lr=lr_rate, decay=1e-5)
    model.compile(optimizer=opt,
                  loss=weighted_categorical_crossentropy(weights),
                  metrics=['accuracy'])

    final_weights_path = 'models/final_InceptionV3.h5'

    model.fit_generator(
        train_generator,
        steps_per_epoch=6812 // batch_size,
        epochs=epochs//2,
        validation_data=val_generator,
        validation_steps=2270 // batch_size,
        callbacks=callback_list
    )

    model.save(final_weights_path)




if __name__ == '__main__':
    """
    Setup training parameters
    """
    batch_size = 8
    num_classes = 2
    img_width, img_height = 224, 224
    num_epoch = 30

    train_inceptionV3(batch_size, num_classes, epochs=30)