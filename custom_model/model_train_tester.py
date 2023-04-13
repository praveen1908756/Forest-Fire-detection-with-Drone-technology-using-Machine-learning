import math
import os

from keras.callbacks import ModelCheckpoint, TensorBoard
from custom_model.cladoh import Cladoh
from setup.naive_approach import generate_from_paths_and_labels, extract_dataset

whole_printer = 0

def train_and_save_cladoh_model(dataset_path, percentage=0.8, nbr_epochs=10, batch_size=32):
    

    model = Cladoh(include_top=True, pooling='max', input_shape=(224, 224, 3))

    custom_based_model_save_folder = "model-saves/custom_based/"

    if not os.path.exists(custom_based_model_save_folder):
        os.makedirs(custom_based_model_save_folder)

    custom_based_model_save_path = custom_based_model_save_folder + "cladoh_save.h5"

    

    save_on_improve = ModelCheckpoint(custom_based_model_save_path, monitor='val_accuracy', verbose=1,
                                      save_best_only=True, save_weights_only=False, mode='max')

    
    tensorboard = TensorBoard(log_dir='./logs', histogram_freq=0, batch_size=32, write_graph=True,
                              write_grads=False, write_images=False, embeddings_freq=0,
                              embeddings_layer_names=None, embeddings_metadata=None,
                              embeddings_data=None, update_freq='epoch')

    callbacks = [save_on_improve, tensorboard]

    model.compile(loss='categorical_crossentropy', optimizer="sgd", metrics=['accuracy'], )

    (train_samples, train_labels), (val_samples, val_labels) = extract_dataset(dataset_path, classes, percentage)

    if whole_printer:
        print(train_samples.shape)
        print(train_labels.shape)
        print(val_samples.shape)
        print(val_labels.shape)
    training_sample_generator = generate_from_paths_and_labels(train_samples, train_labels, batch_size,
                                                               image_size=(224, 224, 3))

    validation_sample_generator = generate_from_paths_and_labels(val_samples, val_labels, batch_size,
                                                                 image_size=(224, 224, 3))

    nbr_train_samples = len(train_samples)
    nbr_val_samples = len(val_samples)

    history = model.fit_generator(
        generator=training_sample_generator,
        steps_per_epoch=math.ceil(nbr_train_samples / batch_size),
        epochs=nbr_epochs,
        validation_data=validation_sample_generator,
        validation_steps=math.ceil(nbr_val_samples / batch_size),
        verbose=1)

    model.save(custom_based_model_save_path)

    print('model saved to: ', custom_based_model_save_path)
    return custom_based_model_save_path, history


if __name__ == '__main__':
    classes = ['fire', 'no_fire', 'start_fire']
    nbr_classes = 3

    classes_value = classes
    split_percentage = 0.8 

    nbr_batch_size = 64  
    dataset_name = 'big'

    dataset_path = os.path.join('datasets/', dataset_name)
    epochs = 30 

    print('nbr_classes: ', nbr_classes)
    print('classes_value: ', classes_value)
    print('split_percentage: ', split_percentage)
    print('nbr_batch_size: ', nbr_batch_size)
    print('dataset_name: ', dataset_name)
    print('dataset_path: ', dataset_path)
    print('epochs: ', epochs)

    if not os.path.exists(dataset_path):
        download_and_setup_dataset(dataset_name)

    os.system('rm -r ' + dataset_path + '/de*')

    model_path, history = train_and_save_custom_model(dataset_path, percentage=split_percentage, nbr_epochs=epochs,
                                                      batch_size=nbr_batch_size)

