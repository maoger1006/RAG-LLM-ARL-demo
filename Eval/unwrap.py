import argparse
import pickle
import os
import huggingface_hub

try:
    from .path_config import default_dataset_root
except ImportError:
    from path_config import default_dataset_root


def unwrap_hf_pkl(pth, suffix='.mp4'):
    base_dir = os.path.join(pth, 'video_pkl/')
    target_dir = os.path.join(pth, 'video/')
    pickle_files = [os.path.join(base_dir, file) for file in os.listdir(base_dir)]
    pickle_files.sort()

    if not os.path.exists(target_dir):
        os.makedirs(target_dir, exist_ok=True)
        for pickle_file in pickle_files:
            with open(pickle_file, 'rb') as file:
                video_data = pickle.load(file)
            # For each video file in the pickle file, write its contents to a new mp4 file
            for video_name, video_content in video_data.items():
                output_path = os.path.join(target_dir, f'{video_name}{suffix}')
                with open(output_path, 'wb') as output_file:
                    output_file.write(video_content)
        print('The video file has been restored and stored from the pickle file.')
    else:
        print('The video file already exists.')
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Restore benchmark videos from pickled chunks.")
    parser.add_argument(
        "--dataset-root",
        default=str(default_dataset_root()),
        help="Directory containing video_pkl/ and where video/ will be written.",
    )
    args = parser.parse_args()
    unwrap_hf_pkl(args.dataset_root)
