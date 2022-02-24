import smbclient
import pandas as pd
import numpy as np
import cv2

server_address = '192.168.178.35'
user_name = ''
pw = ''

base_image_path = '192.168.178.35/DataCStore/HeinrichZille/Bilddateien/'

# Optional - register the server with explicit credentials
smbclient.register_session(server_address, username=user_name, password=pw)

# Create a directory (only the first request needs credentials)
# smbclient.mkdir(r"192.168.178.35/DataCStore/HeinrichZille/Bilddateien/klein", username=user_name, password=pw)

# load image meta data
df = pd.read_csv('../data/parsed_image_meta_data.csv')

all_image_ids = df['id']

for image_id in all_image_ids:
    with smbclient.open_file(f'{base_image_path}{image_id}.jpg', 'rb') as fd:
        f = fd.read()
        jpeg_array = bytearray(f)
        img = cv2.imdecode(np.asarray(jpeg_array), cv2.IMREAD_COLOR)
        scaling_factor = 0.1
        image_resized = cv2.resize(img, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)
        author_prefix = df.iloc[[df.index[df['id'] == image_id].tolist()[0]]]['author'].values[0][0]
        new_image_path = f'HZille_small/{author_prefix}/{image_id}.jpg'
        print(new_image_path)
        cv2.imwrite(new_image_path, image_resized)
