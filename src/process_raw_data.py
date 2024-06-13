import json, os
from config import PATHES
from helper_functions import read_json_ 




def safe_json_(file_path: str, processed_data: dict):
  with open(file_path, 'w') as file:
        json.dump(processed_data, file, indent=4,ensure_ascii=False)



# remove span annotations from raw data:
raw_data = read_json_(PATHES["PATH_PROCESSED_DATA"])
filtered_data = [{key: item[key] for key in ['id', 'text', 'category']} for item in raw_data]
save_filepath = PATHES["PATH_PROCESSED_DATA"]
safe_json_(save_filepath, filtered_data)