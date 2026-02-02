import os
import sys
import logging
import csv
from faster_whisper import WhisperModel
from datetime import timedelta

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def parse_ground_truth(gt_path):
    """Parse the ground truth file to get speaker names and file paths."""
    speakers = []
    with open(gt_path, 'r') as f:
        lines = f.readlines()
        # Skip header "Speakers: ..."
        for line in lines[1:]:
            if ':' in line:
                name, path = line.split(':', 1)
                speakers.append({'name': name.strip(), 'path': path.strip()})
    return speakers

def main():
    # Configuration
    model_size = "medium"
    device = "auto"
    
    # Initialize Whisper
    logging.info(f"Loading Whisper model ({model_size})...")
    # Force CPU to avoid CUDA driver issues
    model = WhisperModel(model_size, device="cpu", compute_type="int8")

    # Process all mix files in data/demo
    demo_dir = "data/demo"
    if not os.path.exists(demo_dir):
        logging.error(f"{demo_dir} does not exist. Run create_and_mix_audio.py first.")
        return

    mix_files = [f for f in os.listdir(demo_dir) if f.startswith("mix_") and f.endswith(".wav")]
    
    for mix_file in mix_files:
        base_name = os.path.splitext(mix_file)[0]
        # mix_2_speakers -> mix_2
        prefix = base_name.replace("_speakers", "")
        gt_file = os.path.join(demo_dir, f"{prefix}_ground_truth.txt")
        
        if not os.path.exists(gt_file):
            logging.warning(f"No ground truth found for {mix_file}, skipping.")
            continue

        logging.info(f"Processing {mix_file}...")
        speakers = parse_ground_truth(gt_file)
        
        full_transcript = []

        for speaker in speakers:
            name = speaker['name']
            audio_path = speaker['path']
            
            if not os.path.exists(audio_path):
                logging.warning(f"Audio file for {name} not found: {audio_path}")
                continue

            logging.info(f"Transcribing {name}...")
            segments, info = model.transcribe(audio_path, language="pt")
            
            text = " ".join([segment.text for segment in segments]).strip()
            if text:
                full_transcript.append(f"{name}: {text}")
            else:
                full_transcript.append(f"{name}: [No speech detected]")

        # Output results
        print(f"\n=== Result for {mix_file} ===")
        for line in full_transcript:
            print(line)
        print("================================\n")
        
        # Save to file
        output_path = os.path.join("data/outputs", f"{base_name}_transcript.txt")
        os.makedirs("data/outputs", exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            for line in full_transcript:
                f.write(line + "\n")
        logging.info(f"Saved result to {output_path}")

if __name__ == "__main__":
    main()
