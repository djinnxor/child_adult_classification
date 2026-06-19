import subprocess
import os
import re
from pathlib import Path


def _get_duration(input_path):
    """Return video duration in seconds, or None on failure."""
    try:
        result = subprocess.run(
            ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
             '-of', 'default=noprint_wrappers=1:nokey=1', str(input_path)],
            capture_output=True, text=True, check=True
        )
        return float(result.stdout.strip())
    except (subprocess.CalledProcessError, ValueError, FileNotFoundError):
        return None


def _time_str_to_seconds(time_str):
    """Convert HH:MM:SS.xx string to total seconds."""
    m = re.match(r'(\d+):(\d+):(\d+(?:\.\d+)?)', time_str)
    if m:
        h, mn, s = m.groups()
        return int(h) * 3600 + int(mn) * 60 + float(s)
    return 0.0


def convert_mov_to_mp4(input_path, output_path=None):
    """
    Convert MOV video file to MP4 format.
    
    Args:
        input_path (str): Path to input MOV file
        output_path (str): Path to output MP4 file (optional, defaults to input filename with .mp4 extension)
    
    Returns:
        bool: True if conversion successful, False otherwise
    """
    input_path = Path(input_path)
    
    if not input_path.exists():
        print(f"Error: Input file '{input_path}' not found")
        return False
    
    if output_path is None:
        output_path = input_path.with_suffix('.mp4')
    else:
        output_path = Path(output_path)

    duration = _get_duration(input_path)
    print(f"Converting '{input_path.name}' ...")
    
    try:
        cmd = [
            'ffmpeg',
            '-i', str(input_path),
            '-c:v', 'libx264',
            '-c:a', 'aac',
            '-y',
            str(output_path)
        ]
        
        process = subprocess.Popen(
            cmd, stderr=subprocess.PIPE, text=True, universal_newlines=True
        )

        for line in process.stderr:
            match = re.search(r'time=(\d+:\d+:\d+\.\d+)', line)
            if match:
                elapsed = _time_str_to_seconds(match.group(1))
                if duration:
                    pct = min(elapsed / duration * 100, 100)
                    bar_len = 30
                    filled = int(bar_len * pct / 100)
                    bar = '#' * filled + '-' * (bar_len - filled)
                    print(f"\r  [{bar}] {pct:5.1f}%  {elapsed:.1f}s / {duration:.1f}s",
                          end='', flush=True)
                else:
                    print(f"\r  {elapsed:.1f}s", end='', flush=True)

        process.wait()
        print()  # newline after progress line

        if process.returncode != 0:
            print(f"Error during conversion of '{input_path}'")
            return False

        print(f"Done -> '{output_path}'")
        return True
        
    except FileNotFoundError:
        print("Error: FFmpeg not found. Please install FFmpeg.")
        return False


def batch_convert_mov_to_mp4(directory, output_directory=None):
    """
    Convert all MOV files in a directory to MP4 format.
    
    Args:
        directory (str): Path to directory containing MOV files
        output_directory (str): Path to output directory (optional, defaults to same directory)
    
    Returns:
        int: Number of files successfully converted
    """
    directory = Path(directory)
    
    if not directory.is_dir():
        print(f"Error: Directory '{directory}' not found")
        return 0
    
    if output_directory is None:
        output_directory = directory
    else:
        output_directory = Path(output_directory)
        output_directory.mkdir(parents=True, exist_ok=True)
    
    mov_files = list(directory.glob('*.mov')) + list(directory.glob('*.MOV'))
    
    if not mov_files:
        print(f"No MOV files found in '{directory}'")
        return 0
    
    successful_conversions = 0
    total = len(mov_files)

    for idx, mov_file in enumerate(mov_files, 1):
        print(f"\n[{idx}/{total}]", end=' ')
        output_path = output_directory / mov_file.with_suffix('.mp4').name
        if convert_mov_to_mp4(mov_file, output_path):
            successful_conversions += 1
    
    print(f"\nConversion complete: {successful_conversions}/{len(mov_files)} files converted")
    return successful_conversions


if __name__ == '__main__':
    # Example usage
    # Single file conversion
    # source_video_path = "/home/rookie/Desktop/zhut/ZHUT CAMER/TCS-25   PTZ 2533.mov"
    # output_video_path = "/home/rookie/Desktop/zhut/ZHUT CAMER/TCS-25   PTZ 2533.mp4"
    # convert_mov_to_mp4(source_video_path, output_video_path)
    
    # Batch conversion
    batch_convert_mov_to_mp4("/media/rookie/One Touch/ZHUT CAMER")
    pass
