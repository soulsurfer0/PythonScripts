import sys
import importlib.util
import subprocess

# Check if package is installed
package_name = 'youtube_transcript_api'
spec = importlib.util.find_spec(package_name)

# If package is not installed, install it
if spec is None:
    print(f"{package_name} is not installed. Installing now...")
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', package_name])
    print(f"Successfully installed {package_name}")

# Now we can safely import
from youtube_transcript_api import YouTubeTranscriptApi

def get_transcript(video_id):
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        return transcript
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def save_transcript_to_file(transcript, filename):
    if not transcript:
        return
    
    with open(filename, 'w', encoding='utf-8') as file:
        for entry in transcript:
            start_time = entry['start']
            duration = entry['duration']
            text = entry['text']
            
            # Format: [MM:SS] Text
            minutes = int(start_time // 60)
            seconds = int(start_time % 60)
            time_stamp = f"[{minutes:02d}:{seconds:02d}] "
            
            file.write(f"{time_stamp}{text}\n")
    
    print(f"Transcript saved to {filename}")

def main():
    # Prompt user for video ID
    print("YouTube Transcript Extractor")
    print("----------------------------")
    print("Please enter the YouTube video ID (the part after v= in the URL)")
    print("For example, for 'https://www.youtube.com/watch?v=dQw4w9WgXcQ', enter 'dQw4w9WgXcQ'")
    
    video_id = input("Video ID: ").strip()
    
    if not video_id:
        print("No video ID provided. Exiting.")
        return
    
    # Prompt for output filename or use default
    print("\nEnter output filename (press Enter to use default):")
    output_filename = input("Filename: ").strip()
    
    if not output_filename:
        output_filename = f"{video_id}_transcript.txt"
    
    transcript = get_transcript(video_id)
    if transcript:
        save_transcript_to_file(transcript, output_filename)
    else:
        print("Could not retrieve transcript. Please check the video ID and make sure the video has captions.")

if __name__ == "__main__":
    main()