import yt_dlp
import cv2
import mediapipe
import os

def download_video(youtube_url: str, output_path: str = "temp_video.mp4") -> str | None:
    """
    Downloads a video from YouTube.

    Args:
        youtube_url: The URL of the YouTube video.
        output_path: The path to save the downloaded video.

    Returns:
        The path to the downloaded video file if successful, None otherwise.
    """
    ydl_opts = {
        'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
        'outtmpl': output_path,
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([youtube_url])
        return output_path
    except Exception as e:
        print(f"Error downloading video: {e}")
        return None

def process_video_with_pose(video_path: str, output_video_path: str = "processed_video.mp4") -> str | None:
    """
    Processes a video to detect and draw pose landmarks.

    Args:
        video_path: The path to the video file.
        output_video_path: The path to save the processed video.

    Returns:
        The path to the processed video file if successful, None otherwise.
    """
    mp_pose = mediapipe.solutions.pose
    pose = mp_pose.Pose()
    mp_drawing = mediapipe.solutions.drawing_utils

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return None

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Ensure fps is valid
    if fps <= 0:
        print(f"Warning: Video FPS is {fps}. Setting to a default of 30.")
        fps = 30


    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Convert the BGR image to RGB.
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Process the image and detect the pose.
            results = pose.process(rgb_frame)

            # Convert the image back to BGR.
            bgr_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)

            # Draw the pose annotation on the image.
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    bgr_frame,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS)
            
            out.write(bgr_frame)
        
        return output_video_path

    except Exception as e:
        print(f"Error processing video: {e}")
        return None
    finally:
        cap.release()
        out.release()
        pose.close() # Release MediaPipe Pose resources
        cv2.destroyAllWindows()
