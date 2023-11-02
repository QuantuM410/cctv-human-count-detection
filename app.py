import streamlit as st
from cctv_counter import CCTVCounter
import tempfile
import os
import subprocess


st.set_page_config(page_title="CCTV Human Counter", page_icon="📷")

def main():
    st.title("CCTV People Counter📷")

    st.write("This webapp uses YOLOv8 to count the number of people in a CCTV footage.")

    if os.path.exists("temp_output/output_video.mp4"):
        os.remove("temp_output/output_video.mp4")

    uploaded_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])
    
    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
            temp_file.write(uploaded_file.read())
            temp_file_path = temp_file.name

        output_video_path = "temp_output/output_video.mp4"

        with st.spinner("Processing video..."):
            average_count = CCTVCounter.process_video(temp_file_path, output_video_path, "yolov8m.pt")

        if average_count is not None:

            # Convert the output video to H.264 format using FFmpeg to display it in Streamlit as cv didnt allow HTML5 video display directly
            with st.spinner("Converting video to H.264 format..."):
                converted_output_path = convert_to_h264(output_video_path)

            if converted_output_path:
                st.write(f"Average person count per frame: {average_count}")
                display_output_video(converted_output_path)
                
                os.remove(temp_file_path)
                os.remove(converted_output_path)
            else:
                st.error("Video conversion failed.")
        else:
            st.error("Video processing failed.")

def convert_to_h264(input_video_path):

    output_video_path = "converted_video.mp4"

    # Uses FFmpeg to convert the video to H.264 format
    try:
        subprocess.run(["ffmpeg", "-i", input_video_path, "-vcodec", "libx264", output_video_path], check=True)
        return output_video_path
    except subprocess.CalledProcessError:
        return None

def display_output_video(video_path):

    with open(video_path, "rb") as f: 
        video_bytes = f.read()
        st.video(video_bytes, format="video/mp4")

if __name__ == "__main__":
    main()
