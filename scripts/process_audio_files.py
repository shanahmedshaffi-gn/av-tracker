
import os
import csv

def create_audio_file_list(data_dir, output_csv):
    with open(output_csv, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['actor_id', 'audio_file'])
        for actor_id in os.listdir(data_dir):
            actor_dir = os.path.join(data_dir, actor_id)
            if os.path.isdir(actor_dir):
                for video_id in os.listdir(actor_dir):
                    video_dir = os.path.join(actor_dir, video_id)
                    if os.path.isdir(video_dir):
                        for audio_file in os.listdir(video_dir):
                            if audio_file.endswith('.wav'):
                                audio_path = os.path.join(video_dir, audio_file)
                                csv_writer.writerow([actor_id, audio_path])

if __name__ == '__main__':
    create_audio_file_list('wav', 'data/audio_files.csv')
