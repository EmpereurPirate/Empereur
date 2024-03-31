import os
import shutil
import time
import requests
import subprocess
from tqdm import tqdm
import json

def snapshot_download_with_retry(repo_id, local_dir, filename):
    max_retries = 5
    retry_delay = 60
    url = f"https://huggingface.co/{repo_id}/resolve/main/{filename}"
    file_path = os.path.join(local_dir, filename)
    size_info_path = os.path.join(local_dir, "size_info.json")
    completed_files_path = os.path.join(local_dir, "completed_files.json")

    if os.path.exists(completed_files_path):
        with open(completed_files_path, 'r') as f:
            completed_files = json.load(f)
    else:
        completed_files = []

    if filename in completed_files:
        print(f"File {filename} is already marked as completed.")

        # Check if the file size is complete
        if os.path.exists(size_info_path):
            with open(size_info_path, 'r') as f:
                size_info = json.load(f)
                expected_total_size = size_info.get(filename, None)

            if expected_total_size is not None:
                existing_file_size = os.path.getsize(file_path)
                if existing_file_size != expected_total_size:
                    print(f"File {filename} size ({existing_file_size} bytes) does not match the expected size ({expected_total_size} bytes). Resuming download.")
                else:
                    print(f"File {filename} is complete and has the expected size ({expected_total_size} bytes).")
                    return
        else:
            print(f"Warning: No size information found for {filename}. Assuming the file is complete.")
            return

    for attempt in range(max_retries):
        try:
            existing_file_size = 0
            if os.path.exists(file_path):
                existing_file_size = os.path.getsize(file_path)

            if existing_file_size > 0:
                if os.path.exists(size_info_path):
                    with open(size_info_path, 'r') as f:
                        size_info = json.load(f)
                        expected_total_size = size_info.get(filename, None)
                else:
                    expected_total_size = None

                if expected_total_size is not None and existing_file_size == expected_total_size:
                    print(f"File {filename} already exists and is complete.")
                    completed_files.append(filename)
                    with open(completed_files_path, 'w') as f:
                        json.dump(completed_files, f)
                    return

                headers = {'Range': f'bytes={existing_file_size}-'}
                response = requests.get(url, headers=headers, stream=True)

            else:
                response = requests.get(url, stream=True)

            total_size = int(response.headers.get('content-length', 0)) + existing_file_size

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

            # Check if the download reached 100%
            if progress_bar.n == total_size:
                print(f"File {filename} downloaded successfully.")

                # Store the expected total size for this file
                size_info = {}
                if os.path.exists(size_info_path):
                    with open(size_info_path, 'r') as f:
                        size_info = json.load(f)
                size_info[filename] = total_size
                with open(size_info_path, 'w') as f:
                    json.dump(size_info, f)

                # Add the successfully downloaded file to the completed_files list
                completed_files.append(filename)
                with open(completed_files_path, 'w') as f:
                    json.dump(completed_files, f)

                return
            else:
                print(f"Download of {filename} stopped before reaching 100%. Resuming download from {progress_bar.n} bytes.")

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
