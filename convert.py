import os
from typing import Optional
from pydub import AudioSegment

def mp3_to_wav(input_folder: str, output_folder: Optional[str] = None) -> None:
    """
    Converts files from mp3 to wav
     
    Args:
        input_folder (str): folder with mp3 files to be converted
        output_folder (str): folder where wav files are saved
    """
    if output_folder is None:
        output_folder = input_folder

    for filename in os.listdir(input_folder):
        if filename.lower().endswith(".mp3"):
            mp3_path = os.path.join(input_folder, filename)
            wav_filename = os.path.splitext(filename)[0] + ".wav"
            wav_path = os.path.join(output_folder, wav_filename)

            # Load and convert
            audio = AudioSegment.from_mp3(mp3_path)
            audio.export(wav_path, format="wav")

            print(f"Converted: {filename} → {wav_filename}")
    print("All conversions completed.")
