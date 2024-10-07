import os
import requests
from tqdm import tqdm

# Directory to save datasets
save_dir = '../data/'

# Ensure save directory exists
os.makedirs(save_dir, exist_ok=True)

# List of dataset URLs
dataset_urls = [
    "https://zenodo.org/record/3893508/files/ribfrac-train-images-1.zip", 
    "https://zenodo.org/record/3893508/files/ribfrac-train-info-1.csv", 
    "https://zenodo.org/record/3893508/files/ribfrac-train-labels-1.zip",
    "https://zenodo.org/record/3893498/files/ribfrac-train-images-2.zip", 
    "https://zenodo.org/record/3893498/files/ribfrac-train-info-2.csv", 
    "https://zenodo.org/record/3893498/files/ribfrac-train-labels-2.zip",  
    "https://zenodo.org/record/3893496/files/ribfrac-val-images.zip", 
    "https://zenodo.org/record/3893496/files/ribfrac-val-info.csv", 
    "https://zenodo.org/record/3893496/files/ribfrac-val-labels.zip",       
    "https://zenodo.org/record/3993380/files/ribfrac-test-images.zip",           
]

# Download function with progress bar
def download_file_with_progress(url, dest_folder):
    local_filename = url.split('/')[-1]
    local_path = os.path.join(dest_folder, local_filename)
    
    # Send GET request to start the download
    with requests.get(url, stream=True) as r:
        r.raise_for_status()  # Check if request was successful
        total_size = int(r.headers.get('content-length', 0))  # Get total file size
        
        # Use tqdm to display the progress bar
        with open(local_path, 'wb') as f, tqdm(
            desc=local_filename,
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
                bar.update(len(chunk))  # Update progress bar with chunk size
    return local_path

# Download each dataset
for url in dataset_urls:
    print(f"Downloading {url}...")
    downloaded_file = download_file_with_progress(url, save_dir)
    print(f"Saved to {downloaded_file}")

print("Download completed.")
