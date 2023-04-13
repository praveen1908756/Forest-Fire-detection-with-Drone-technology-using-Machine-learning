import imghdr
import math
import os
import numpy as np
from keras import Model
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input as inception_preprocess_input
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input as vgg16_preprocess_input
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.layers import GlobalAveragePooling2D, Dense
from keras.preprocessing import image
from keras.utils import np_utils


classes = ['fire', 'no_fire', 'start_fire']
nbr_classes = 3


def generate_from_paths_and_labels(images_paths, labels, batch_size, preprocessing, image_size=(224, 224)):

    number_samples = len(images_paths)

    while 1:
        perm = np.random.permutation(number_samples) 
        images_paths = images_paths[perm]
        labels = labels[perm]

        for i in range(0, number_samples, batch_size):
            inputs = list(map(
                lambda x: image.load_img(x, target_size=image_size),
                images_paths[i:i + batch_size]
            ))
            inputs = np.array(list(map(
                lambda x: image.img_to_array(x),
                inputs
            )))

            inputs = preprocessing(inputs)

            yield (inputs, labels[i:i + batch_size])


def extract_dataset(dataset_path, classes_names, percentage):

    num_classes = len(classes_names)

    images_paths, labels = [], []
    for class_name in os.listdir(dataset_path):
        class_path = os.path.join(dataset_path, class_name)
        class_id = classes_names.index(class_name) 
        for path in os.listdir(class_path):
            path = os.path.join(class_path, path)
            if imghdr.what(path) is None:
                continue
            images_paths.append(path)
            labels.append(class_id)

    labels_oh = np_utils.to_categorical(labels, num_classes)
    images_paths = np.array(images_paths)

    number_samples = len(images_paths)
    perm = np.random.permutation(number_samples)
    labels_oh = labels_oh[perm]
    images_paths = images_paths[perm]

    border = math.floor(percentage * len(images_paths))

    train_labels, val_labels = labels_oh[:border], labels_oh[border:]
    train_samples, val_samples = images_paths[:border], images_paths[border:]

    print("Training on %d samples" % (len(train_samples)))
    print("Validation on %d samples" % (len(val_samples)))

    return (train_samples, train_labels), (val_samples, val_labels)


def create_VGG16_based_model():
    base_model = VGG16(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
    x = base_model.output
    x = GlobalAveragePooling2D()(x) 
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(nbr_classes, activation='softmax')(x) 
    model = Model(inputs=base_model.inputs, outputs=predictions) 

    for layer in model.layers:
        layer.trainable = True

    return model


def train_and_save_VGG16_based_model(dataset_path, percentage=0.9, nbr_epochs=10, batch_size=32):
    VGG16_based_model = create_VGG16_based_model()

    VGG16_based_model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

    VGG16_based_model_save_folder = "model-saves/VGG16_based/"

    if not os.path.exists(VGG16_based_model_save_folder):
        os.makedirs(VGG16_based_model_save_folder)

    (train_samples, train_labels), (val_samples, val_labels) = extract_dataset(dataset_path, classes, percentage)

    training_sample_generator = generate_from_paths_and_labels(train_samples, train_labels, batch_size,
                                                               vgg16_preprocess_input, image_size=(224, 224, 3))

    validation_sample_generator = generate_from_paths_and_labels(val_samples, val_labels, batch_size,
                                                                 vgg16_preprocess_input, image_size=(224, 224, 3))

    nbr_train_samples = len(train_samples)
    nbr_val_samples = len(val_samples)

    history = VGG16_based_model.fit_generator(
        generator=training_sample_generator,
        steps_per_epoch=math.ceil(nbr_train_samples / batch_size),
        epochs=nbr_epochs,
        validation_data=validation_sample_generator,
        validation_steps=math.ceil(nbr_val_samples / batch_size),
        verbose=1)

    VGG16_based_model_save_path = VGG16_based_model_save_folder + "trained_save.h5"
    VGG16_based_model.save(VGG16_based_model_save_path)


def create_Inception_based_model():
    base_model = InceptionV3(include_top=False, weights='imagenet', pooling='max', input_shape=(224, 224, 3))

    x = base_model.output
    x = Dense(512, activation='relu')(x)
    x = Dense(256, activation='relu')(x)
    x = Dense(128, activation='relu')(x)
    predictions = Dense(nbr_classes, activation='softmax')(x)
    model = Model(inputs=base_model.inputs, outputs=predictions) 

    for layer in model.layers:
        layer.trainable = True

    return model


def train_and_save_Inception_based_model(dataset_path, percentage=0.9, nbr_epochs=10, batch_size=32):

    Inception_based_model = create_Inception_based_model()

    Inception_based_model_save_folder = "model-saves/Inception_based/"
    if not os.path.exists(Inception_based_model_save_folder):
        os.makedirs(Inception_based_model_save_folder)

    Inception_based_model_save_path = Inception_based_model_save_folder + "best_trained_save.h5"
    save_on_improve = ModelCheckpoint(Inception_based_model_save_path, monitor='val_acc', verbose=1,
                                      save_best_only=True, save_weights_only=False, mode='max')


    tensorboard = TensorBoard(log_dir='./logs', histogram_freq=0, batch_size=32, write_graph=True,
                              write_grads=False, write_images=False, embeddings_freq=0,
                              embeddings_layer_names=None, embeddings_metadata=None,
                              embeddings_data=None, update_freq='epoch')

    cb = [save_on_improve, tensorboard]

    Inception_based_model.compile(loss='categorical_crossentropy', optimizer="sgd", metrics=['accuracy'])

    (train_samples, train_labels), (val_samples, val_labels) = extract_dataset(dataset_path, classes, percentage)

    training_sample_generator = generate_from_paths_and_labels(train_samples, train_labels, batch_size,
                                                               inception_preprocess_input, image_size=(224, 224, 3))

    validation_sample_generator = generate_from_paths_and_labels(val_samples, val_labels, batch_size,
                                                                 inception_preprocess_input, image_size=(224, 224, 3))

    nbr_train_samples = len(train_samples)
    nbr_val_samples = len(val_samples)

    history = Inception_based_model.fit_generator(
        generator=training_sample_generator,
        steps_per_epoch=math.ceil(nbr_train_samples / batch_size),
        epochs=nbr_epochs,
        validation_data=validation_sample_generator,
        validation_steps=math.ceil(nbr_val_samples / batch_size),
        callbacks=cb, verbose=1)
