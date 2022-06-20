from urllib.request import urlretrieve
import requests
import socket
import os


def urllib_download(img_url,file_name):
    urlretrieve(img_url,file_name)
    print(file_name+'finished')

with open('img.tsv','r') as f:
    next(f)
    for line in f:
        line = line.strip('\n').split('\t')
        if 'true' in line[0]:
            file_name = "./true_"
        else:
            file_name = "./fake_"
        if line[1] == '0':
            file_name = file_name+"light/"
        else:
            file_name = file_name+"dark/"
        file_name = file_name + line[0][-8:]
        urllib_download(line[0], file_name)
        


