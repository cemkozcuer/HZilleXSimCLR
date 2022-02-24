"""
This script downloads the dew (Date Estimation in the Wild) dataset.

Images only until the year 1950 are downloaded.

The script checks which of the images are already downloaded,
collects bad response requests and timeouts when requesting.

In that regard the script can continue it´s work when restarted
after any kind of exception or shut down.
"""


import pandas as pd
import requests
import time
import os
import sys


def append_to_txt(txt_name, str_to_append):
    with open(txt_name, 'a') as _f:
        _f.write(str_to_append)
        _f.write('\n')


def get_left_images(_df: pd.DataFrame, _download_folder: str, _bad_responses_txt: str) -> pd.DataFrame:

    print('Computing left images to download...')

    print('...counting already downloaded images...')
    downloaded_images = os.listdir(_download_folder)
    print(f'...found {len(downloaded_images)} downloaded images...')

    downloaded_ids = []
    for image_path in downloaded_images:

        file_name = image_path.split('.')[0]

        if file_name != '':
            _img_id = int(file_name.split('_')[0])  # first part before "_" is image id in dataset
            downloaded_ids.append(_img_id)

    # filter out bad responses from last attempt
    with open(_bad_responses_txt) as _f:
        _bad_responses = [line.rstrip('\n') for line in _f]

    print(f'...found {len(_bad_responses)} bad responses...')

    print('before remove', len(_df))

    _remaining_image_ids = _df[~_df['url'].isin(_bad_responses)]
    print('after removing bad responses', len(_remaining_image_ids))

    # filter out already downloaded images
    _remaining_image_ids = _remaining_image_ids[~_remaining_image_ids['img_id'].isin(downloaded_ids)]['img_id']
    print('after removing downloaded', len(_remaining_image_ids), '\n')

    return _remaining_image_ids


try:
    print()

    until = 1950
    download_folder = '/volumes/512GB 5200U TM/datasets/dew/until_1950/'
    bad_responses_txt = 'bad_responses.txt'

    df = pd.read_csv('../data/dew/meta.csv')
    df_selected = df[df['GT'] <= until]

    remaining_image_ids = get_left_images(df_selected, download_folder, bad_responses_txt)

    successful_downloads = []
    errors = []
    bad_responses = []
    timeouts = []

    total_images = len(remaining_image_ids)
    count = 0
    total_time = 0

    for img_id in remaining_image_ids:
        start_time = time.time()
        count += 1

        row = df_selected[df_selected['img_id'] == img_id].iloc[0]

        url = row['url']
        print(f'Downloading no.{count}:', url)

        img_file_name = str.split(url, '/')[-1]
        try:
            response = requests.get(url, timeout=5)
            print(response.status_code, response.headers['content-type'])

            if response.headers['content-type'] == 'image/jpeg':
                f = open(f'{download_folder}/{img_file_name}', 'wb')
                f.write(response.content)
                f.close()

                successful_downloads.append(url)
                print('download successful')
            else:
                bad_responses.append(url)
                append_to_txt('bad_responses.txt', url)

                print('bad response')

        except requests.exceptions.Timeout:
            timeouts.append(url)
            append_to_txt('timeouts.txt', url)
            print('timeout during downloading:', url)

        except IOError:
            errors.append(url)
            append_to_txt('errors.txt', url)
            print('failed downloading:', url)

        end_time = time.time()
        took_time = end_time - start_time
        total_time += took_time
        left_images = total_images - count
        avg_time_per_image = total_time / count

        print()
        print('took:', round(took_time, 1), 'secs')
        print('time total:', round(total_time, 1), 'secs')
        print('avg time:', round(avg_time_per_image, 1), 'secs')
        print('left:', left_images, 'est:', round(avg_time_per_image * left_images / 60, 0), 'mins', '\n')

        # print(i, url, file_name)
        # if len(succesful_downloads) == 50:
        # if count == 10:
        #     break

    print(f'Downloaded {len(successful_downloads)} images, {len(errors)} errors, {len(bad_responses)} bad responses, {len(timeouts)} timeouts.')

except KeyboardInterrupt:

    print('Interrupted')
    print(f'Downloaded {len(successful_downloads)} images, {len(errors)} errors, {len(bad_responses)} bad responses, {len(timeouts)} timeouts.')
    sys.exit(0)
