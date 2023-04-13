import streamlit as st
import subprocess
from PIL import Image
import re
import os

def run_command(command):
    output = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE).communicate()[0]
    return output.decode('utf-8')

def get_percentage(output):
    input_str = output
    match = re.search(r'\[(\d+\.\d+e?-?\d*)', input_str)
    if match:
        val = float(match.group(1))
        percent = val * 100
        return percent
    else:
        return None

st.title("Fire Detection App")

option = st.selectbox(
    'What do you want to do?',
    ('Select an option', 'Image Input', 'Video Input')
)

if option == 'Select an option':
    st.warning("Please select a valid option")

elif option == 'Image Input':
    st.subheader("Fire Detection from Image")

    image_file = st.file_uploader("Upload Image", type=['jpg', 'jpeg', 'png'])

    if image_file is not None:
        with open('input_image.jpg', 'wb') as f:
            f.write(image_file.read())
        image = Image.open(image_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        command = f"python launcher.py predict -path input_image.jpg -model transfer_learned_model.h5"
        output = run_command(command)
        percent = get_percentage(output)

        if percent is not None:
            if percent >= 60:
                st.error(f"Percentage Probability of fire: {percent:.2f}%")
            elif percent >= 30:
                st.warning(f"Percentage Probability of fire: {percent:.2f}%")
            else:
                st.success(f"Percentage Probability of fire: {percent:.2f}%")

elif option == 'Video Input':
    st.subheader("Fire Detection from Video")

    video_file = st.file_uploader("Upload Video", type=['mp4'])

    if video_file is not None:
        with open('input_video.mp4', 'wb') as f:
            f.write(video_file.read())
        command = f"python launcher.py video -in input_video.mp4 -out output_vids -model transfer_learned_model.h5"
        output = run_command(command)
        st.write(f"Output: {output}")
        st.warning("The processed video can be found in 'output_vids' folder")

        # Display processed video
        video_path = os.path.join('output_vids', 'output.mp4')
        if os.path.exists(video_path):
            with open(video_path, "rb") as f:
                video_bytes = f.read()
            st.video(video_bytes)
