import numpy as np
from keras import backend as K
from keras.models import Model
from keras.layers import GlobalAveragePooling2D

from keras.preprocessing.image import ImageDataGenerator
from keras.layers.core import Dense
from keras.optimizers import Adam
from keras.applications import InceptionV3, VGG16, VGG19, Xception, ResNet50
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, ReduceLROnPlateau
from sklearn.utils import class_weight


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

def get_generators(dataset='macro', batch_size=32, img_width=224, img_height=224):
    datagen = ImageDataGenerator(
        horizontal_flip=True,
        width_shift_range=0.3,
        height_shift_range=0.1,
        rotation_range=5,
        zoom_range=0.3,
        rescale=1./255)

    if dataset == 'macro':
        path_to_image_dir = '../data/macro_dataset'
    else:
        path_to_image_dir = '../data/micro_dataset'

    generator_train = datagen.flow_from_directory(
        f'{path_to_image_dir}/train',
        target_size=(img_width, img_height),
        batch_size=batch_size,
        shuffle=True,
        class_mode='categorical')
    
    generator_val = datagen.flow_from_directory(
        f'{path_to_image_dir}/val',
        target_size=(img_width, img_height),
        batch_size=batch_size,
        shuffle=True,
        class_mode='categorical')

    return generator_train, generator_val

def get_base_model_and_layer_number(model_name, img_width, img_height):
    if model_name == 'inceptionV3':
        base_model = InceptionV3(input_shape=(img_width, img_height, 3), weights='imagenet', include_top=False)
        last_layer_number = 249
    elif model_name == 'vgg16':
        base_model = VGG16(input_shape=(img_width, img_height, 3), weights='imagenet', include_top=False)
        last_layer_number = len(base_model.layers)
    elif model_name == 'vgg19':
        base_model = VGG16(input_shape=(img_width, img_height, 3), weights='imagenet', include_top=False)
        last_layer_number = len(base_model.layers)
    elif model_name == 'xception':
        base_model = Xception(input_shape=(img_width, img_height, 3), weights='imagenet', include_top=False)
        last_layer_number = len(base_model.layers)
    elif model_name == 'resnet50':
        base_model = ResNet50(input_shape=(img_width, img_height, 3), weights='imagenet', include_top=False)
        last_layer_number = len(base_model.layers)

    return base_model, last_layer_number
    

def get_callback_list(path):
    callback_list = [
        ModelCheckpoint(f'models/{path}.h5', monitor='val_loss', verbose=1, save_best_only=True),
        EarlyStopping(monitor='val_loss', patience=30, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', patience=30, verbose=1),
        TensorBoard(log_dir=f'./logs/{path}')]
    return callback_list

def train(name='vgg16', dataset='macro', epochs=30, img_width=227, img_height=227, batch_size=2, lr_rate=0.001):
    train_generator, val_generator = get_generators(dataset, batch_size, img_width, img_height)
    base_model, last_layer_number = get_base_model_and_layer_number(name, img_width, img_height)
    assert len(set(train_generator.classes)) == len(set(val_generator.classes))
    num_classes = len(set(train_generator.classes))

    class_weights = class_weight.compute_class_weight(
                    'balanced',
                    np.unique(train_generator.classes),
                    train_generator.classes
    )

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(num_classes, activation='softmax')(x)

    for layer in base_model.layers:
        layer.trainable = False
    
    model = Model(base_model.input, predictions)
    top_weights_path = f'{name}_{dataset}_lr{lr_rate}_{batch_size}bs'

    opt = Adam(lr=lr_rate)
    model.compile(optimizer=opt,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    """
    Directives for training
    """
    callback_list = get_callback_list(top_weights_path)
    
    model.fit_generator(train_generator,
                        steps_per_epoch= train_generator.n // batch_size,
                        epochs=epochs//2,
                        validation_data=val_generator,
                        validation_steps=val_generator.n // batch_size,
                        class_weight=class_weights,
                        callbacks=callback_list)

    model.load_weights(f'models/{top_weights_path}.h5')

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

    final_weights_path = f'models/final_{name}_{dataset}_lr{lr_rate}_{batch_size}bs'

    model.fit_generator(
        train_generator,
        steps_per_epoch=train_generator.n // batch_size,
        epochs=epochs//2,
        validation_data=val_generator,
        validation_steps=val_generator.n // batch_size,
        class_weight=class_weights,
        callbacks=callback_list
    )

    model.save(final_weights_path)


if __name__ == '__main__':
    # Train for Macro dataset
    
    networks_list = ['vgg16','vgg19','xception','resnet50', 'inceptionV3']
    for dataset in ['micro']:
        for lr in [0.1, 0.01, 0.001, 0.0001, 0.00001]:
            for network in networks_list:
                for batch in [128, 64, 32, 16, 8]:
                    train(network, dataset, epochs=200, batch_size=batch, lr_rate=lr)


