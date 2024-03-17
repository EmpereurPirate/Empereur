import os
import shutil
import time
import requests
import subprocess
from tqdm import tqdm

def snapshot_download_with_retry(repo_id, local_dir, filename):
    max_retries = 5
    retry_delay = 60
    url = f"https://huggingface.co/{repo_id}/resolve/main/{filename}"
    file_path = os.path.join(local_dir, filename)

    for attempt in range(max_retries):
        try:
            existing_file_size = 0
            if os.path.exists(file_path):
                existing_file_size = os.path.getsize(file_path)
                headers = {'Range': f'bytes={existing_file_size}-'}
                response = requests.get(url, headers=headers, stream=True)
            else:
                response = requests.get(url, stream=True)

            total_size = int(response.headers.get('content-length', 0)) + existing_file_size

            if existing_file_size > 0 and existing_file_size == total_size:
                print(f"File {filename} already exists and appears to be complete.")
                return

            if existing_file_size > 0:
                print(f"Resuming download of {filename} from {existing_file_size} bytes.")

            response.raise_for_status()
            block_size = 1024
            progress_bar = tqdm(total=total_size, unit='iB', unit_scale=True, desc=f"{filename}", initial=existing_file_size)

            with open(file_path, 'ab') as f:
                for chunk in response.iter_content(chunk_size=block_size):
                    if chunk:
                        progress_bar.update(len(chunk))
                        f.write(chunk)

            progress_bar.close()
            print(f"File {filename} downloaded successfully.")
            return

        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 416:  # Requested Range Not Satisfiable
                print(f"File {filename} appears to be corrupted. Restarting download from the beginning.")
                if os.path.exists(file_path):
                    os.remove(file_path)
                existing_file_size = 0
            elif e.response.status_code == 404:
                print(f"File {filename} not found in the repository. Skipping download.")
                return
            else:
                print(f"Error downloading {filename}: {e}")

            if attempt < max_retries - 1:
                print(f"Retrying download of {filename} in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                print(f"Failed to download {filename} after {max_retries} attempts. Skipping to the next file.")

if __name__ == "__main__":
    print("Starting download function...")
    default_model_name = "ShinojiResearch/Senku-70B-Full"
    model_name = input(f"Enter the model name to download (default: '{default_model_name}'): ")
    if not model_name:
        model_name = default_model_name

    output_dir = "."
    download_dir = os.path.join(output_dir, model_name)
    if not os.path.exists(download_dir):
        os.makedirs(download_dir)

    temp_dir = os.path.join(output_dir, "temp")
    repo_dir = os.path.join(temp_dir, model_name.split("/")[-1])

    if os.path.exists(repo_dir):
        subprocess.run(["git", "-C", repo_dir, "config", "pull.rebase", "false"])
        subprocess.run(["git", "-C", repo_dir, "pull", "origin", "main"])
    else:
        os.makedirs(temp_dir)
        subprocess.run(["git", "clone", f"https://huggingface.co/{model_name}", repo_dir])

    repo_files = []
    for root, _, files in os.walk(repo_dir):
        for file in files:
            repo_files.append(os.path.join(root, file))

    for repo_file in repo_files:
        relative_file_path = os.path.relpath(repo_file, repo_dir)
        snapshot_download_with_retry(model_name, download_dir, relative_file_path)

    shutil.rmtree(temp_dir)
    print("Download function completed.")
