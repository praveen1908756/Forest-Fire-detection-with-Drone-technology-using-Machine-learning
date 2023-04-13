import imghdr
import os
import math
import numpy as np
from keras.models import load_model
from keras.optimizers import SGD
from keras.preprocessing import image
from matplotlib import pyplot as plt
from keras import Model
from keras.applications import InceptionV3
from keras.applications.inception_v3 import preprocess_input as inception_preprocess_input
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.layers import Dense
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator


classes = ['fire', 'no_fire', 'start_fire']
nbr_classes = 3


def augmented_batch_generator(images_paths, labels, batch_size, preprocessing, augment, image_size=(224, 224)):
    

    display = False 

    number_samples = len(images_paths) 
    if augment:
        data_transformer = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True,
                                              rotation_range=20, brightness_range=[0.7, 1.3], zoom_range=[0.8, 1.3])

    while 1:
        perm = np.random.permutation(number_samples)

        images_paths = images_paths[perm]
        labels = labels[perm]

        
        for i in range(0, number_samples, batch_size):

            batch = list(map(
                lambda x: image.load_img(x, target_size=image_size),
                images_paths[i:i + batch_size]
            ))

            if augment:
                batch = np.array(list(map(
                    lambda x: data_transformer.random_transform(image.img_to_array(x)),
                    batch
                )))
            else:
                batch = np.array(list(map(
                    lambda x: image.img_to_array(x),
                    batch
                )))

            if display:
                for j in range(9):
                    plt.subplot(330 + 1 + j)
                    img = batch[j].astype('uint8')
                    plt.imshow(img)
                    print(labels[j])

            batch = preprocessing(batch)

            yield (batch, labels[i:i + batch_size])


def extract_dataset(dataset_path, classes_names, percentage):
    

    num_classes = len(classes_names)

    def listdir_nohidden(path):
        for f in os.listdir(path):
            if not f.startswith('.'):
                yield f

    train_labels, val_labels = np.empty([1, 0]), np.empty([1, 0])
    train_samples, val_samples = np.empty([1, 0]), np.empty([1, 0])

    for class_name in listdir_nohidden(dataset_path):
        images_paths, labels = [], []

        class_path = os.path.join(dataset_path, class_name)
        class_id = classes_names.index(class_name)

        for path in listdir_nohidden(class_path):
            path = os.path.join(class_path, path) 
            if imghdr.what(path) is None:
                continue
            images_paths.append(path)
            labels.append(class_id)

        labels_oh = np.array(labels)
        images_paths = np.array(images_paths)

        number_samples = len(images_paths)
        perm = np.random.permutation(number_samples)
        labels_oh = labels_oh[perm]
        images_paths = images_paths[perm]

        border = math.floor(percentage * len(images_paths))

        train_labels_temp, val_labels_temp = labels_oh[:border], labels_oh[border:]
        train_samples_temp, val_samples_temp = images_paths[:border], images_paths[border:]

        train_labels = np.append(train_labels, train_labels_temp)
        val_labels = np.append(val_labels, val_labels_temp)

        train_samples = np.append(train_samples, train_samples_temp)
        val_samples = np.append(val_samples, val_samples_temp)

    number_samples_train = len(train_samples)
    perm = np.random.permutation(number_samples_train)
    train_labels = np_utils.to_categorical(train_labels, num_classes)
    train_labels = train_labels[perm]
    train_samples = train_samples[perm]

    number_samples_val = len(val_samples)
    perm = np.random.permutation(number_samples_val)
    val_labels = np_utils.to_categorical(val_labels, num_classes)
    val_labels = val_labels[perm]
    val_samples = val_samples[perm]

    print("Training on %d samples" % number_samples_train)
    print("Validation on %d samples" % number_samples_val)

    return (train_samples, train_labels), (val_samples, val_labels)


def create_inception_based_model():
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


def train_inception_based_model(dataset_path,
                                fine_tune_existing=None,
                                learning_rate=0.001,
                                percentage=0.9,
                                nbr_epochs=10,
                                batch_size=32):
    
    if fine_tune_existing is not None:
        inception_based_model = load_model(fine_tune_existing)
    else:
        inception_based_model = create_inception_based_model()

    inception_based_model_save_folder = "model-saves/Inception_based/"

    if not os.path.exists(inception_based_model_save_folder):
        os.makedirs(inception_based_model_save_folder)

    inception_based_model_save_path = inception_based_model_save_folder + "best_trained_save.h5"

    
    save_on_improve = ModelCheckpoint(inception_based_model_save_path,
                                      monitor='val_acc',
                                      verbose=1,
                                      save_best_only=True,
                                      save_weights_only=False,
                                      mode='max')

    tensorboard = TensorBoard(log_dir='./logs',
                              histogram_freq=0,
                              batch_size=32,
                              write_graph=True,
                              write_grads=False,
                              write_images=False,
                              embeddings_freq=0,
                              embeddings_layer_names=None,
                              embeddings_metadata=None,
                              embeddings_data=None,
                              update_freq='epoch')

    cb = [save_on_improve, tensorboard]

    if fine_tune_existing is not None:
        sgd = SGD(lr=learning_rate, momentum=0.0, nesterov=False) 
        inception_based_model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    else:
        inception_based_model.compile(loss='categorical_crossentropy', optimizer="sgd", metrics=['accuracy'])

    (train_samples, train_labels), (val_samples, val_labels) = extract_dataset(dataset_path, classes, percentage)

    training_sample_generator = augmented_batch_generator(train_samples,
                                                          train_labels,
                                                          batch_size,
                                                          inception_preprocess_input,
                                                          augment=True,
                                                          image_size=(224, 224, 3))

    validation_sample_generator = augmented_batch_generator(val_samples,
                                                            val_labels,
                                                            batch_size,
                                                            inception_preprocess_input,
                                                            augment=False,
                                                            image_size=(224, 224, 3))

    nbr_train_samples = len(train_samples)
    nbr_val_samples = len(val_samples)

    history = inception_based_model.fit_generator(
        generator=training_sample_generator,
        steps_per_epoch=math.ceil(nbr_train_samples / batch_size),
        epochs=nbr_epochs,
        validation_data=validation_sample_generator,
        validation_steps=math.ceil(nbr_val_samples / batch_size),
        callbacks=cb,
        verbose=1)


def create_simpler_inception_based_model():
    base_model = InceptionV3(include_top=False, weights='imagenet', pooling='max', input_shape=(224, 224, 3))

    x = base_model.output
    x = Dense(2048, activation='relu', name='fc_1')(x)
    x = Dense(1024, activation='relu', name='fc_2')(x)
    predictions = Dense(nbr_classes, activation='softmax', name='fc_class')(x) 
    model = Model(inputs=base_model.inputs, outputs=predictions) 

    for layer in base_model.layers:
        layer.trainable = False

    return model


def train_simpler_inception_based_model(dataset_path,
                                        fine_tune_existing=None,
                                        save_path="best_trained_save.h5",
                                        freeze=True,
                                        learning_rate=0.001,
                                        percentage=0.9,
                                        nbr_epochs=10,
                                        batch_size=32):
    
    if fine_tune_existing is not None:
        simpler_inception_based_model = load_model(fine_tune_existing)
    else:
        simpler_inception_based_model = create_simpler_inception_based_model()

    if not freeze:
        for layer in simpler_inception_based_model.layers:
            layer.trainable = True
    else:
        for layer in simpler_inception_based_model.layers:
            if layer.name != 'fc_1' and layer.name != 'fc_2' and layer.name != 'fc_class':
                layer.trainable = False

    simpler_inception_based_model_save_folder = "model-saves/Inception_based/"

    if not os.path.exists(simpler_inception_based_model_save_folder):
        os.makedirs(simpler_inception_based_model_save_folder)

    simpler_inception_based_model_save_path = simpler_inception_based_model_save_folder + save_path

    simpler_inception_based_model.summary()

    
    save_on_improve = ModelCheckpoint(simpler_inception_based_model_save_path,
                                      monitor='val_acc',
                                      verbose=1,
                                      save_best_only=True,
                                      save_weights_only=False,
                                      mode='max')

    cb = [save_on_improve]

    
    if fine_tune_existing is not None:
        sgd = SGD(lr=learning_rate, momentum=0.0, nesterov=False) 
        simpler_inception_based_model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    else:
        simpler_inception_based_model.compile(loss='categorical_crossentropy', optimizer="sgd", metrics=['accuracy'])

    (train_samples, train_labels), (val_samples, val_labels) = extract_dataset(dataset_path, classes, percentage)

    training_sample_generator = augmented_batch_generator(train_samples,
                                                          train_labels,
                                                          batch_size,
                                                          inception_preprocess_input,
                                                          augment=True,
                                                          image_size=(224, 224, 3))

    validation_sample_generator = augmented_batch_generator(val_samples,
                                                            val_labels,
                                                            batch_size,
                                                            inception_preprocess_input,
                                                            augment=False,
                                                            image_size=(224, 224, 3))

    nbr_train_samples = len(train_samples)
    nbr_val_samples = len(val_samples)

    history = simpler_inception_based_model.fit_generator(
        generator=training_sample_generator,
        steps_per_epoch=math.ceil(nbr_train_samples / batch_size),
        epochs=nbr_epochs,
        validation_data=validation_sample_generator,
        validation_steps=math.ceil(nbr_val_samples / batch_size),
        callbacks=cb, verbose=1)
