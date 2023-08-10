import glob
import pandas as pd
import numpy as np
import json

def process_data(folders, file_limit, output_file_path):
    file_paths = []

    for folder in folders:
        folder_path = f'C:/Users/ADMIN/Desktop/train/{folder}/20per/*.json' ## 데이터셋 파일 경로
        files = glob.glob(folder_path)[:file_limit]
        file_paths.extend(files)
        
    data_list = []

    for file_path in file_paths:
        with open(file_path, "r", encoding="utf-8") as file:
            json_data = json.load(file)

        doc_id = json_data["Meta(Acqusition)"]["doc_id"]
        doc_type = json_data["Meta(Acqusition)"]["doc_type"]
        doc_name = json_data["Meta(Acqusition)"]["doc_name"]
        passage = json_data["Meta(Refine)"]["passage"]

        data_list.append({"doc_id": doc_id, "doc_type": doc_type, "doc_name": doc_name, "passage": passage})

    df = pd.DataFrame(data_list)

    with open(output_file_path, 'w', encoding='utf-8') as output_file:
        json.dump(data_list, output_file, ensure_ascii=False, indent=4)

# Usage
folders = ['01.news_r', '03.his_cul', '04.paper', '06.edit']
file_limit = 4000
output_file_path = 'C:/Users/ADMIN/Desktop/trainer.json'

