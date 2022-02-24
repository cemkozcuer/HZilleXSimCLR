"""
This is just a scratch book in order to convert the Heinrich Zille metadata XML file
into JSON. Eventually it turned out this step is not needed.

The final parsing of the XML file is done in the ipython notebook: "HZille_meta_data_parsing_XML_to_CSV.ipynb".
"""


import numpy as np
import cv2
import pandas as pd
import xmltodict
import json


import sys
# import smbclient

# smbclient.ClientConfig(username='', password='')


# smbclient.mkdir(r"\\server\share\directory", username="user", password="pass")
#
# with smbclient.open_file(r"\\server\share\directory\file.txt", mode="w") as fd:
#     fd.write(u"file contents")


# smbclient.listdir('smb://192.168.178.35/DataCStore/HeinrichZille/Bilddateien')
#
# smbclient.readlink()
import smbclient
from smbclient import (
    listdir,
    mkdir,
    register_session,
    rmdir,
    scandir,
)

from smbclient.path import (
    isdir,
)

server_address = '192.168.178.35'

# Optional - register the server with explicit credentials
register_session(server_address, username='', password='')

# Create a directory (only the first request needs credentials)
# mkdir(r"\\server\share\directory", username="user", password="pass")
#
# # Remove a directory
# rmdir(r"\\server\share\directory")
#
# # Checking whether a file is a directory
# d_filename = r"\\server\share\directory"
# print("Is file {} dir?: {}".format(d_filename, isdir(d_filename)))

# List the files/directories inside a dir
# for filename in listdir(r"192.168.178.35/DataCStore/HeinrichZille/Bilddateien"):
#     print(filename)
#     with smbclient.open_file(f'192.168.178.35/DataCStore/HeinrichZille/Bilddateien/{filename}', 'rb') as fd:
#         f = fd.read()
#         jpeg_array = bytearray(f)
#         img = cv2.imdecode(np.asarray(jpeg_array), cv2.IMREAD_COLOR)
#         cv2.imshow('Image display', img)
#         cv2.waitKey(delay=0)


    # f = fobj.read()
    # jpeg_array = bytearray(f)
    # img = cv2.imdecode(np.asarray(jpeg_array), cv2.IMREAD_COLOR)
    # cv2.imshow('Image display', img)
    # break
    # with open("img.png", "rb") as image:
    #     f = image.read()
    #     b = bytearray(f)py
    #     print
    #     b[0]


# READ meta data from XML

# open meta data as file object from SMB server
meta_data_address = f'{server_address}/DataCStore/HeinrichZille/Metadaten/BG_LIDO_Zille_CdV_20170816.xml'

with smbclient.open_file(meta_data_address, 'r') as meta_data_file:
    # meta_data = pd.read_xml(meta_data_file)
    # print(meta_data)
    xml_as_dict = xmltodict.parse(meta_data_file.read())

    with open('../data/HZille_meta_data.json', 'w') as json_file:
        json.dump(xml_as_dict, json_file, indent=4)
