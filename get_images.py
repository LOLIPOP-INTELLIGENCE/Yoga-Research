import os
import requests

# Base directory for dataset
base_dir = 'Yoga-82/Dataset/'

# Loop through each file in the directory containing links
for file in os.listdir('Yoga-82/yoga_dataset_links'):
    if file.endswith(".txt"):
        with open(os.path.join('Yoga-82/yoga_dataset_links', file)) as f:
            for line in f:
                try:
                    # Split line into name and URL
                    name, url = line.strip().split('\t')
                    # Create full directory path
                    dir_path = os.path.join(base_dir, os.path.dirname(name))
                    # Create full file path
                    full_path = os.path.join(base_dir, name)

                    # Create directory if it doesn't exist
                    if not os.path.exists(dir_path):
                        os.makedirs(dir_path)

                    # Download and save the image with a 5-second timeout
                    print('Downloading', name, 'from', url)
                    r = requests.get(url, stream=True, timeout=5)
                    if r.status_code == 200:
                        with open(full_path, 'wb') as outfile:
                            for chunk in r:
                                outfile.write(chunk)
                    
                    print(f"Downloaded {name}")
                except Exception as e:
                    print(f"Error processing {line}: {e}")
