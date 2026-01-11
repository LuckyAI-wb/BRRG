import os
import json
from collections import defaultdict

with open('annotation_M.json', 'r') as file:
    data = json.load(file)

base_directory = 'E:\\CKRRG\\Datasets\\MIMC-CXR\\cxr_image\\'

processed_data = {
    'train': [],
    'val': [],
    'test': []
}

def get_study_id(path):
    """
    根据你的路径规则提取 study_id.
    例子: ...\\p10\\p10000032\\s50414267\\xxx.jpg
    study_id = s50414267
    """
    parts = os.path.normpath(path).split(os.sep)
    for token in parts:
        if token.startswith("s") and token[1:].isdigit():
            return token
    return None

for split in data:
    # step1: 按 study_id 分组
    study_groups = defaultdict(list)
    for entry in data[split]:
        report = entry.get('report', '')
        image_paths = entry.get('image_path', [])
        for img in image_paths:
            abs_path = os.path.join(base_directory, img).replace('/', '\\')
            study_id = get_study_id(abs_path)
            if study_id:
                study_groups[(study_id, report)].append(abs_path)

    # step2: 每个 study_id 下聚合成一条记录
    for (study_id, report), paths in study_groups.items():
        new_entry = {
            'report': report
        }
        if len(paths) > 0:
            new_entry['image_path_1'] = paths[0]
        if len(paths) > 1:
            new_entry['image_path_2'] = paths[1]
        processed_data[split].append(new_entry)

with open('dataset_cxr.json', 'w', encoding='utf-8') as file:
    json.dump(processed_data, file, indent=4, ensure_ascii=False)

print("New dataset has been created as 'dataset_cxr.json'.")
