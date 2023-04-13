import math
import os

import cv2
import numpy as np
from keras.models import load_model
from keras.preprocessing import image
from matplotlib import pyplot as plt

from setup.naive_approach import extract_dataset, generate_from_paths_and_labels


def graphically_evaluate_model(model_path, classes_names, test_image_dir, preprocess_input, image_size=(224, 224)):
    nbr_classes = len(classes_names)

    model = load_model(model_path)

    for test_image_path in os.listdir(test_image_dir):

        img = image.load_img(test_image_dir + "/" + test_image_path, target_size=image_size)

        processed_img = image.img_to_array(img)
        processed_img = np.expand_dims(processed_img, axis=0)
        processed_img = preprocess_input(processed_img)

        predictions = model.predict(processed_img)[0]
        result = [(classes_names[i], float(predictions[i]) * 100.0) for i in range(nbr_classes)]

        result.sort(reverse=True, key=lambda x: x[1])

        img = cv2.imread(test_image_dir + "/" + test_image_path)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        font = cv2.FONT_HERSHEY_COMPLEX

        for i in range(nbr_classes):

            (class_name, prob) = result[i]

            textsize = cv2.getTextSize(class_name, font, 1, 2)[0]
            textX = (img.shape[1] - textsize[0]) / 2
            textY = (img.shape[0] + textsize[1]) / 2

            if i == 0:
                cv2.putText(img, class_name, (int(textX) - 100, int(textY)), font, 5, (255, 255, 255), 6, cv2.LINE_AA)

            print("Class name: %s" % class_name)
            print("Probability: %.2f%%" % prob)

        plt.imshow(img)
        plt.show()


def evaluate_model(model_path, classes, preprocessing, dataset_path):
    (train_samples, train_labels), (val_samples, val_labels) = extract_dataset(dataset_path, classes, 0)

    batch_size = 16

    nbr_val_samples = len(val_samples)

    validation_sample_generator = generate_from_paths_and_labels(val_samples,
                                                                 val_labels,
                                                                 batch_size,
                                                                 preprocessing,
                                                                 image_size=(224, 224, 3))

    model = load_model(model_path)

    metrics = model.evaluate_generator(validation_sample_generator,
                                       steps=math.ceil(nbr_val_samples / 16),
                                       max_queue_size=10,
                                       workers=1,
                                       use_multiprocessing=True,
                                       verbose=1)

    out = ""
    for i in range(len(model.metrics_names)):
        out += model.metrics_names[i]
        out += " : "
        out += str(float(metrics[i]))
        out += " | "

    return out


def extract_hard_samples(model_path, preprocess_input, dataset_path, threshold, image_size=(224, 224)):
    classes = ['fire', 'no_fire', 'start_fire']

    nbr_classes = 3

    model = load_model(model_path)
    hard_examples = [[] for j in range(nbr_classes)]

    for i in range(nbr_classes):

        class_name = classes[i]

        for sample_path in os.listdir(dataset_path + class_name):

            img = image.load_img(dataset_path + class_name + "/" + sample_path, target_size=image_size)

            processed_img = image.img_to_array(img)
            processed_img = np.expand_dims(processed_img, axis=0)
            processed_img = preprocess_input(processed_img)

            predictions = model.predict(processed_img)[0]

            if float(predictions[i]) < threshold:
                hard_examples[i].append(sample_path)

    return hard_examples


def display_hard_samples(hard_examples, dataset_path):

    classes = ['fire', 'no_fire', 'start_fire']

    nbr_classes = 3

    for i in range(nbr_classes):

        class_name = classes[i]

        print("========== CLASS : " + class_name + " ==========")
        for sample_path in hard_examples[i]:

            img = cv2.imread(dataset_path + "/" + class_name + "/" + sample_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            plt.imshow(img)
            plt.show()
