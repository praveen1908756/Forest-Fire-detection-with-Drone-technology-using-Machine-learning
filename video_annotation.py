import os
import cv2
import numpy as np  
from keras.models import load_model
import keras.utils as image

def video_fire_detection(input_video_path, output_video_path, model_path, model_preprocess, image_size, detection_freq):
    if not os.path.exists("temp_frames"):
        os.makedirs("temp_frames")

    classes = ['fire', 'no_fire', 'start_fire']

    nbr_classes = 3

    frame_width = 1920
    frame_height = 1080
    size = (frame_width, frame_height)
    video_writer = cv2.VideoWriter('output_vids/fire4.mp4', cv2.VideoWriter_fourcc(*'MJPG'), 10, size)

    model = load_model(model_path)

    video = cv2.VideoCapture(input_video_path)

    if not video.isOpened():
        print("Error opening video stream or file")

    frame_nbr = 0
    img, max_class, max_proba = None, "unknown", 0

    while video.isOpened():
        not_done, frame = video.read()

        if not_done:
            img_name = "temp-frame" + str(frame_nbr) + ".png"
            img_path = "temp_frames/" + img_name

            cv2.imwrite(img_path, frame)

            if frame_nbr % detection_freq != 0:
                img = cv2.imread(img_path)
                height, width, channels = img.shape

                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                font = cv2.FONT_HERSHEY_SIMPLEX
                text = str(max_class) + " : " + str("{:.2f}".format(max_proba)) + "%"

                textsize = cv2.getTextSize(text, font, 1, 2)[0]

                textX = (img.shape[1] - textsize[0]) // 2
                textY = (img.shape[0] + textsize[1]) // 2

                rectangle_bgr = (0, 0, 0)

                box_coords = ((textX, textY), (textX + textsize[0], textY - textsize[1]))
                cv2.rectangle(img, box_coords[0], box_coords[1], rectangle_bgr, cv2.FILLED)

                cv2.putText(img, text, (textX, textY), font, 1, (255, 255, 255), 2)

            else:
                img = image.load_img(img_path, target_size=image_size)
                img = image.img_to_array(img)
                img = np.expand_dims(img, axis=0)
                img = model_preprocess(img)

                probabilities = model.predict(img,
                                              batch_size=1,
                                              verbose=0)[0]

                result = [(classes[i], float(probabilities[i]) * 100.0) for i in range(nbr_classes)]

                result.sort(reverse=True, key=lambda x: x[1])

                max_class, max_proba = result[0][0], result[0][1]

                img = cv2.imread(img_path)

                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                font = cv2.FONT_HERSHEY_SIMPLEX

                text = str(max_class) + " : " + str("{:.2f}".format(max_proba)) + "%"

                textsize = cv2.getTextSize(text, font, 1, 2)[0]

                textX = (img.shape[1] - textsize[0]) // 2
                textY = (img.shape[0] + textsize[1]) // 2

                rectangle_bgr = (0, 0, 0)

                box_coords = ((textX, textY), (textX + textsize[0], textY - textsize[1]))
                cv2.rectangle(img, box_coords[0], box_coords[1], rectangle_bgr, cv2.FILLED)

                cv2.putText(img, text, (textX, textY), font, 1, (255, 255, 255), 2)

            frame_nbr = frame_nbr + 1
            video_writer.write(img)

        else:
            break

    video_writer.release()    
    video.release()


def extract_images_from_video(video_path, images_directory):
    if not os.path.exists(images_directory):
        os.makedirs(images_directory)

    video = cv2.VideoCapture(video_path)

    if not video.isOpened():
        print("Error opening video stream or file")

    frame_nbr = 0

    while video.isOpened():
        not_done, frame = video.read()

        if not_done:
            img_name = "frame_" + str(frame_nbr) + ".png"
            img_path = images_directory + img_name
            cv2.imwrite(img_path, frame)
            frame_nbr = frame_nbr + 1
        else:
            break

    video.release()


def detect_fire_save_frames(input_video_path, output_video_path, model_path, model_preprocess, image_size,
                            detection_freq):
    classes = ['fire', 'no_fire', 'start_fire']

    nbr_classes = 3

    extract_images_from_video(input_video_path, "./video_frames/")

    video_writer = imageio.get_writer('new_vids', fps=24)

    model = load_model(model_path)

    max_class, max_proba = "unknown", 0

    frames = []
    counter = 0

    for img_path in sorted(os.listdir('video_frames'), key=lambda f: int("".join(list(filter(str.isdigit, f))))):

        complete_path = 'video_frames/' + img_path

        frames.append(complete_path)

        img = cv2.imread(complete_path)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        font = cv2.FONT_HERSHEY_SIMPLEX

        text = str(max_class) + " : " + str(max_proba) + "%"

        textsize = cv2.getTextSize(text, font, 1, 2)[0]

        textX = (img.shape[1] - textsize[0]) // 2
        textY = (img.shape[0] + textsize[1]) // 2


        cv2.putText(img, text, (textX, textY), font, 1, (255, 255, 255), 2)

        if counter % detection_freq == 0:
            img = image.load_img(complete_path, target_size=image_size)
            img = image.img_to_array(img)
            img = np.expand_dims(img, axis=0)
            img = model_preprocess(img)

            probabilities = model.predict(img,
                                          batch_size=1,
                                          verbose=0,
                                          steps=None,
                                          callbacks=None,
                                          max_queue_size=10,
                                          workers=1,
                                          use_multiprocessing=False)[0]

            result = [(classes[i], float(probabilities[i]) * 100.0) for i in range(nbr_classes)]

            result.sort(reverse=True, key=lambda x: x[1])

            max_class, max_proba = result[0][0], result[0][1]

            img = cv2.imread(complete_path)

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            font = cv2.FONT_HERSHEY_SIMPLEX

            text = str(max_class) + " : " + str("{:.2f}".format(max_proba)) + "%"

            textsize = cv2.getTextSize(text, font, 1, 2)[0]

            textX = (img.shape[1] - textsize[0]) // 2
            textY = (img.shape[0] + textsize[1]) // 2

            cv2.putText(img, text, (textX, textY), font, 1, (255, 255, 255), 2)

        counter = counter + 1
        video_writer.append_data(img)

    video_writer.close()
