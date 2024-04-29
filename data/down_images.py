import os, json 
from tqdm import tqdm
data_name = "Clothing Shoes and Jewelry"
process_dir = "processed_filter_5_cloth"

import requests

def down_save(url, image_name):
    r = requests.get(url, stream=True)
    with open(image_name, 'wb') as f:
        f.write(r.content)

with open(f'{process_dir}/meta_{data_name}.json') as f:
    meta_data = json.load(f)
if not os.path.exists(f"{process_dir}/{data_name}"):
    os.mkdir(f"{process_dir}/{data_name}")
missing_images = []
for key, values in tqdm(meta_data.items()):
    imUrl = values["imUrl"]
    out_filepath=f"{process_dir}/{data_name}/{key}.jpg"
    try:
        down_save(imUrl, out_filepath)
    except:
        missing_images.append(key)
with open("missing_imgs.txt",'w') as f:
    json.dump(missing_images, f)