"""
This is a small UI app that is made in order to crop the Heinrich Zille images.

The images are scanned with color and contrast cards in order
to have a reference in manual image processing and printing.

These are not needed and counter productive for the following machine learning
and can be removed with this app.

The app opens each image one by one (and spare already processed ones).
Left mouse click sets the left-top rectangle corner. Clicking second time sets the right-bottom corner.
Right mouse click will remove set corner (lifo).
Space bar will crop if both corners are set, store the image and load the next one.
"""


import smbclient
import pandas as pd
import numpy as np
import cv2
import io
import time
import os


def print_all_callback_params(event, x, y, flags, param):
    """
    This is basically an example of the callback API of opencv
    UI apps.
    """
    # print('---')
    # print(event)
    # print(f'x: {x} / y: {y}')
    # print(flags)
    # print(param)
    # print('---')
    pass


def get_opencv_img_from_file(file_obj):

    byte_obj = file_obj.read()
    jpeg_array = bytearray(byte_obj)
    _image = cv2.imdecode(np.asarray(jpeg_array), cv2.IMREAD_COLOR)

    return _image


def get_opencv_img_from_path(path):
    img = cv2.imread(path)

    return img


def process_mouse_click(event, x, y, _app_state):

    if event == cv2.EVENT_LBUTTONUP:
        # add coordinates
        if _app_state[rectangle_top_left_key] is False:
            _app_state[rectangle_top_left_key] = (x, y)
            print('Did set top-left:', (x, y))
        elif _app_state[rectangle_bottom_right_key] is False:

            _x1, _y1 = app_state[rectangle_top_left_key]

            if _x1 < x and _y1 < y:
                _app_state[rectangle_bottom_right_key] = (x, y)
                print('Did set bottom-right:', (x, y))
            else:
                print('Invalid 2nd point: needs to be bottom right.')
        else:
            print('doing nothing - rectangle set already...')
            # pass

    elif event == cv2.EVENT_RBUTTONUP:
        # remove set coordinates
        if _app_state[rectangle_bottom_right_key] is not False:
            _app_state[rectangle_bottom_right_key] = False
            print('removed bottom-right')
        elif _app_state[rectangle_top_left_key] is not False:
            _app_state[rectangle_top_left_key] = False
            print('removed top-left')
        else:
            print('doing nothing - no rectangle set...')
            # pass

    else:
        # print('unused event:', event)
        pass
    # print('app state out:', app_state)


def set_current_x_y(event, x, y, _app_state):

    if event == cv2.EVENT_MOUSEMOVE:
        _app_state[current_x_y_key] = (x, y)


def mouse_callback_creator(_app_state):

    def mouse_callback(event, x, y, flags, param):
        print_all_callback_params(event, x, y, flags, param)
        process_mouse_click(event, x, y, _app_state)
        set_current_x_y(event, x, y, _app_state)

    return mouse_callback


# todo: add credentials here in order to us an SMB client
# server_address = '192.168.178.35'
# user_name = ''
# pw = ''

# base_image_path = '192.168.178.35/DataCStore/Data_Sets/HeinrichZille/Bilddateien/'
# base_writing_path = '192.168.178.35/DataCStore/Data_Sets/HeinrichZille/WORK/Bilddateien_ausgeschnitten/'
base_image_path = '/volumes/512GB 5200U TM/datasets/HeinrichZille/Bilddateien/'
base_writing_path = '/volumes/512GB 5200U TM/datasets/HeinrichZille/Bilddateien_ausgeschnitten/'

# in order to assure everything is either on smb or normal file system
assert base_image_path[0:3] == base_writing_path[0:3]
operating_on_smb = base_image_path[0:3] == '192'

# Optional - register the server with explicit credentials
# smbclient.register_session(server_address, username=user_name, password=pw)

# load image meta data
df = pd.read_csv('data/parsed_image_meta_data.csv')

# todo remove already processed ids
# already_processed_images = smbclient.listdir(base_writing_path)
already_processed_images = os.listdir(base_writing_path)
image_ids = [image_path.split('.')[0] for image_path in already_processed_images]
image_ids = [entry for entry in image_ids if entry != '']
image_ids = [int(entry) for entry in image_ids]

all_image_ids = df['id']
print('before remove', len(all_image_ids))

all_image_ids = df[~df['id'].isin(image_ids)]['id']
print('after remove', len(all_image_ids), '\n')

window_name = 'CROP'
top_left_is_set_key = 'top_left_is_set'
bottom_right_is_set_key = 'bottom_right_is_set'
rectangle_top_left_key = 'rectangle_top_left',
rectangle_bottom_right_key = 'rectangel_bottom_right'
current_x_y_key = 'current_x_y'

app_state = {
    top_left_is_set_key: False,
    bottom_right_is_set_key: False,
    rectangle_top_left_key: False,
    rectangle_bottom_right_key: False,
    current_x_y_key: False
}

total_time = 0

# process all images
for i, image_id in enumerate(all_image_ids):
    start_time = time.time()

    cv2.namedWindow(window_name)

    img_path = f'{base_image_path}{image_id}.jpg'
    print('-->', img_path)

    if operating_on_smb:
        # with smbclient.open_file(img_path, 'rb') as image_as_file:  # normally this way it´s more robust...
        image_as_file = smbclient.open_file(img_path, 'rb')
        image = get_opencv_img_from_file(image_as_file)
        image_as_file.close()
    else:
        image = cv2.imread(img_path)

    # scale images down before processing (otherwise app can be very laggy)
    scaling_factor = 0.1
    image_resized = cv2.resize(image, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)

    # keep vanilla image for refreshing
    vanilla_image = image_resized.copy()

    # lisen to UI
    cv2.setMouseCallback(window_name, mouse_callback_creator(app_state))

    last_point_a = False
    last_point_b = False

    using_vanilla_image = True
    pink = (128, 0, 255)
    rectangle_thickness = int(0.0015 * image_resized.shape[1])

    # drawing loop
    while True:

        # UI callback can set rectangle coordinates...
        has_set_rectangle_coordinates = app_state[rectangle_top_left_key] is not False

        # only update draw if the rectangle is set... otherwise there is no need to reload the image in every loop
        if has_set_rectangle_coordinates:
            using_vanilla_image = False

            point_a = app_state[rectangle_top_left_key]

            second_point_is_set = app_state[rectangle_bottom_right_key] is not False

            if second_point_is_set:
                point_b = app_state[rectangle_bottom_right_key]
            else:
                point_b = app_state[current_x_y_key]

            pink = (128, 0, 255)

            rectangle_has_changed = point_a != last_point_a or point_b != last_point_b

            # redraw image with rectangle, otherwise rectangle gets drawn multiple times on image (every drawing loop)
            if rectangle_has_changed:
                image_resized = vanilla_image.copy()
                cv2.rectangle(
                    image_resized,
                    pt1=point_a,
                    pt2=point_b,
                    color=pink,
                    thickness=rectangle_thickness
                )

            last_point_a = point_a
            last_point_b = point_b
        # do not redraw
        else:
            if using_vanilla_image is False:
                image_resized = vanilla_image.copy()
                using_vanilla_image = True

        cv2.imshow(window_name, image_resized)

        # finish cropping (results in actual cropping, storing and going to next image
        if cv2.waitKey(33) & 0xFF == 32:  # await SPACE key to exit/go to next image. #27 would be ESC key
            break

    cv2.destroyAllWindows()

    x1, y1 = app_state[rectangle_top_left_key]
    x2, y2 = app_state[rectangle_bottom_right_key]

    # todo: sanity check for top-left before bottom-right

    print()
    print(f'cropping at: {x1},{y1} / {x2},{y2}')
    cropped_image = vanilla_image[y1:y2, x1:x2]
    new_image_path = f'{base_writing_path}{image_id}.jpg'

    if operating_on_smb:
        with smbclient.open_file(new_image_path, mode='wb') as fd:
            is_success, image_buffer = cv2.imencode('.jpg', cropped_image)
            io_buf = io.BytesIO(image_buffer)
            fd.write(io_buf.getbuffer())
            fd.close()
    else:
        cv2.imwrite(new_image_path, cropped_image, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

    # reset app state for next image
    app_state = {
        top_left_is_set_key: False,
        bottom_right_is_set_key: False,
        rectangle_top_left_key: False,
        rectangle_bottom_right_key: False,
        current_x_y_key: False
    }

    # finish time measurements
    end_time = time.time()
    took_time = end_time - start_time
    total_time += took_time

    # calculate estimated time statistics
    left_images = len(all_image_ids) - (i + 1)
    print()
    print('took:', round(took_time, 1), 'secs', 'est:', round(took_time * left_images / 60, 0))
    print('time total:', round(total_time, 1), 'secs')
    print('avg time:', round(total_time / (i + 1), 1), 'secs')
    print('left:', left_images, 'est:', round(total_time / (i + 1) * left_images / 60, 0), 'mins', '\n')
    # break
