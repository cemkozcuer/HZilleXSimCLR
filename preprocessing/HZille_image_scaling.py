import sys
import pandas as pd
import cv2

base_image_path = '/volumes/512GB 5200U TM/datasets/HeinrichZille/Bilddateien/'

# load image meta data
df = pd.read_csv('../data/parsed_image_meta_data.csv')

img_count = 0
all_images = len(df)

for image_id in df['id']:

    image_path = f'{base_image_path}{image_id}.jpg'
    img = cv2.imread(image_path)

    if img is None:
        sys.exit(f'Couldn´t read {image_path} .')
    else:
        print(f'Did load {image_path}')

    scaling_factor = 0.1
    print(f'Scaling by {scaling_factor}')
    image_resized = cv2.resize(img, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)

    new_image_path = f'data/HZille_small/all/{image_id}.jpg'
    print(f'Writing to {new_image_path}')
    cv2.imwrite(new_image_path, image_resized)

    img_count += 1
    print(f'{all_images - img_count} images left...')
