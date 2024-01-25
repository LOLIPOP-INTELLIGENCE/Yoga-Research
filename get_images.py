import os
import requests
import random

# Base directory for dataset
base_dir = 'Yoga-82/Dataset/'

count = 0
# Loop through each file in the directory containing links
for file in os.listdir('Yoga-82/yoga_dataset_links'):
    #print all files in the directory
    print(file)
    # count length
    count += 1
    if file.endswith(".txt"):
        with open(os.path.join('Yoga-82/yoga_dataset_links', file)) as f:
            # Read all lines into a list
            lines = f.readlines()
            # Determine the smaller number between 100 and the total number of lines
            num_lines_to_download = min(100, len(lines))
            # Randomly select lines
            selected_lines = random.sample(lines, num_lines_to_download)

            for line in selected_lines:
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

print(f"Processed {count} files")