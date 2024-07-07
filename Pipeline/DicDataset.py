import os
import json
import torch
from torch.utils.data import Dataset

from celldet import CellDetectionMetric

class DicDataset(Dataset):
    def __init__(self, txt_path, json_path):
        self.txt_path = txt_path
        self.json_path = json_path
        self.txt_files = sorted([f for f in os.listdir(txt_path) if f.endswith('.txt')])

    def __len__(self):
        return len(self.txt_files)

    def __getitem__(self, idx):
        txt_file = self.txt_files[idx]
        base_name = os.path.splitext(txt_file)[0]
        json_file = base_name + '.json'

        # Process TXT file
        txt_file_path = os.path.join(self.txt_path, txt_file)
        boxes = []
        with open(txt_file_path, 'r') as file:
            for line in file:
                columns = line.strip().split()
                if len(columns) >= 2:
                    boxes.append([float(columns[0]), float(columns[1])])
        boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
        if boxes_tensor.size(0)==0:
            boxes_tensor=torch.empty(0,2)

        # Create dictionary from TXT
        dict_from_txt = {
            'boxes': boxes_tensor,
            'labels': torch.ones(len(boxes), dtype=torch.float32)
        }

        # Process JSON file
        json_file_path = os.path.join(self.json_path, json_file)
        centroids = []
        if os.path.exists(json_file_path):
            with open(json_file_path, 'r') as file:
                data = json.load(file)
            for key, value in data.get('nuc', {}).items():
                if 'centroid' in value:
                    centroids.append(value['centroid'])
        centroids_tensor = torch.tensor(centroids, dtype=torch.float32)
        if centroids_tensor.size(0)==0:
            centroids_tensor=torch.empty(0,2)

        # Create dictionary from JSON
        dict_from_json = {
            'boxes': centroids_tensor,
            'labels': torch.ones(len(centroids), dtype=torch.float32),
            'scores': torch.ones(len(centroids), dtype=torch.float32)
        }

        return dict_from_txt, dict_from_json, base_name