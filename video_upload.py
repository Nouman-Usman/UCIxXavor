import os
import shutil

UPLOAD_DIR = 'uploads'

def upload_video(video_file):
    allowed_extensions = ['.mp4', '.avi', '.mov']  # Add other video formats if needed
    filename = video_file.filename
    extension = os.path.splitext(filename)[1].lower()

    if extension not in allowed_extensions:
        return None, "Invalid file type. Please upload a video file."

    # Ensure the uploads directory exists
    if not os.path.exists(UPLOAD_DIR):
        os.makedirs(UPLOAD_DIR)

    new_path = os.path.join(UPLOAD_DIR, filename)
    video_file.save(new_path)
    return filename, None
