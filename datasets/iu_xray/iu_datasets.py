import os
import json

with open('annotation.json', 'r') as file:
    data = json.load(file)

base_directory = 'E:\\CKRRG\\Datasets\\iu_xray\\images\\'

processed_data = {
    'train': [],
    'val': [],
    'test': []
}

for split in data:
    for entry in data[split]:
        report = entry.get('report', '')
        image_paths = entry.get('image_path', [])
        new_entry = {
            'report': report,
        }

        if len(image_paths) > 0:
            new_entry['image_path_1'] = os.path.join(base_directory, image_paths[0]).replace('/', '\\')
        if len(image_paths) > 1:
            new_entry['image_path_2'] = os.path.join(base_directory, image_paths[1]).replace('/', '\\')

        processed_data[split].append(new_entry)

with open('dataset_iu.json', 'w') as file:
    json.dump(processed_data, file, indent=4)

print("New dataset has been created as 'dataset_iu.json'.")
