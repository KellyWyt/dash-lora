import cv2
import random
import os
import numpy as np
from pydub import AudioSegment
from moviepy.editor import VideoFileClip

def extract_frames(video_path, start_time, end_time, mode='train', num_frames=10):
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Unable to open video file {video_path}")
        return None

    # Get the FPS and total number of frames
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Convert start and end times to frame numbers
    start_frame = int(start_time * fps)
    end_frame = int(end_time * fps)

    # Ensure end_frame is within the video length
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    end_frame = min(end_frame, total_frames - 1)

    extracted_frames = []
    frame_indices = np.linspace(start_frame, end_frame, num=num_frames, dtype=int)

    # Sort the frame indices to process in order
    # frame_indices = sorted(frame_indices)

    # Extract the frames
    for frame_index in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = cap.read()
        if ret:
            extracted_frames.append(frame)
            # print(f"Extracted frame at index: {frame_index}")
        else:
            print(f"Error: Failed to read frame at index {frame_index}")
    
    # Release the video capture object
    cap.release()

    return extracted_frames

def extract_audio_from_video(video_path, start_time, end_time, output_audio_folder, filename):
    try:
        # Load the video using moviepy to extract audio
        video_clip = VideoFileClip(video_path)
        
        # Extract the audio from the video
        audio = video_clip.audio.subclip(start_time, end_time)

        # Save the extracted audio as WAV in the specified folder
        audio_output_path = os.path.join(output_audio_folder, f"{filename}.wav")
        audio.write_audiofile(audio_output_path)
        # print(f"Extracted audio saved as {audio_output_path}")

    except Exception as e:
        print(f"Error extracting audio: {e}")

def process_video_from_txt(txt_line, video_directory, output_video_folder, output_audio_folder, mode='train', skip_if_exists=True):
    # Parse the txt line (format: class$filename&good&sttime&endtime)
    parts = txt_line.strip().split('&')
    class_name = parts[0]
    filename = parts[1]
    start_time = int(parts[3])
    end_time = int(parts[4])

    # Construct the video file path
    video_path = os.path.join(video_directory, f"{filename}.mp4")
    
    # Check if already processed
    video_frame_folder = os.path.join(output_video_folder, filename)
    frame_1_path = os.path.join(video_frame_folder, 'frame_1.jpg')
    
    if skip_if_exists and os.path.exists(frame_1_path):
        print(f"Skipping {filename} - already processed")
        return
    
    # Extract frames from the video
    frames = extract_frames(video_path, start_time, end_time, mode=mode)
    
    if frames is None or len(frames) == 0:
        print(f"Error: Failed to extract frames from {filename}")
        return

    # Save the frames as jpg in the specified output video folder
    os.makedirs(video_frame_folder, exist_ok=True)

    for idx, frame in enumerate(frames):
        frame_filename = os.path.join(output_video_folder, filename, f"frame_{idx + 1}.jpg")
        cv2.imwrite(frame_filename, frame)
        # print(f"Saved frame: {frame_filename}")

    # Extract audio from the video and save to output_audio_folder
    extract_audio_from_video(video_path, start_time, end_time, output_audio_folder, filename)

def process_dataset(txt_file_path, video_directory, output_base_folder, mode, skip_if_exists=True):
    # Create output folders for video and audio
    output_video_folder = os.path.join(output_base_folder, mode, 'video')
    output_audio_folder = os.path.join(output_base_folder, mode, 'audio')
    os.makedirs(output_video_folder, exist_ok=True)
    os.makedirs(output_audio_folder, exist_ok=True)

    # Process each line in the txt file
    with open(txt_file_path, 'r') as txt_file:
        lines = txt_file.readlines()
        total = len(lines)
        processed = 0
        skipped = 0
        
        for line in lines:
            video_frame_folder = os.path.join(output_video_folder, line.strip().split('&')[1])
            frame_1_path = os.path.join(video_frame_folder, 'frame_1.jpg')
            
            if skip_if_exists and os.path.exists(frame_1_path):
                skipped += 1
                print(f"[{processed + skipped}/{total}] Skipping {line.strip().split('&')[1]} - already processed")
            else:
                process_video_from_txt(line, video_directory, output_video_folder, output_audio_folder, mode, skip_if_exists)
                processed += 1
                print(f"[{processed + skipped}/{total}] Processed {line.strip().split('&')[1]}")
        
        print(f"\nSummary: Processed {processed}, Skipped {skipped}, Total {total}")

# Example usage
train_txt_file = "/nfs1/outdated/WYT/MokA-copy/AVE_Dataset/trainSet.txt"
test_txt_file = "/nfs1/outdated/WYT/MokA-copy/AVE_Dataset/testSet.txt"
video_directory = "/nfs1/outdated/WYT/MokA-copy/AVE_Dataset/AVE"

# Output base folder where train/ and test/ subfolders will be created
output_base_folder = "/nfs1/outdated/WYT/MokA-copy/AVE_Dataset"

# Process train and test datasets (skip_if_exists=True to skip already processed videos)
process_dataset(train_txt_file, video_directory, output_base_folder, mode='train', skip_if_exists=True)
process_dataset(test_txt_file, video_directory, output_base_folder, mode='test', skip_if_exists=True)
