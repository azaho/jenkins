import requests
import os

def download_file(url, local_path):
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(local_path), exist_ok=True)

    print(f"Downloading file from {url}")
    print(f"Saving to {local_path}")

    # Download the file with streaming to handle large files
    response = requests.get(url, stream=True)
    response.raise_for_status()  # Raise an exception for bad status codes

    with open(local_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

    print("Download completed successfully")

#### SESSION 1 ####
# URL of the file to download
url = 'https://api.dandiarchive.org/api/assets/9a3225ad-4925-4174-918b-973c057d71b8/download/'
# Local path where the file will be saved  
local_path = '000070/sub-Jenkins/sub-Jenkins_ses-20090916_behavior+ecephys.nwb'
# Check if file already exists
if os.path.exists(local_path):
    print(f"File already exists at {local_path}, skipping download")
else:
    download_file(url, local_path)

#### SESSION 2 #### -- not needed for now
# # URL of the file to download
# url = 'https://api.dandiarchive.org/api/assets/033a04c0-9ce7-45c5-8012-c571a7b9bf12/download/'
# # Local path where the file will be saved  
# local_path = '000070/sub-Jenkins/sub-Jenkins_ses-20090918_behavior+ecephys.nwb'
# # Check if file already exists
# if os.path.exists(local_path):
#     print(f"File already exists at {local_path}, skipping download")
# else:
#     download_file(url, local_path)
