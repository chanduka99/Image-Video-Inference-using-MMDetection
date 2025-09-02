import os
import requests


def download_weights(url,file_save_name):
    """
    Download weights for any model.

    :param url: Dwonload url for the weight file.
    :param file_save_name:  String name to save the file on to the disk.
    """
    # Make checkpoint directory if not present
    if not os.path.exists('checkpoint'):
        os.makedirs('checkpoint')

    # Download the file if not present
    if not os.path.exists(os.path.join('checkpoint',file_save_name)):
        file =  requests.get(url)
        open(
            os.path.join('checkpoint',file_save_name),'wb'
        ).write(file.content)


