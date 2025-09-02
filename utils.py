import os
import requests
import glob
import yaml

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



def parse_meta_file():
    """
    Function to parse all the model files inside 'mmdetection/configs'
    and return the download URLs for all the available models.

    Returns:
        weight_list: List containing URLs for all the downloadble models.
    """
    root_meta_file_path='mmdetection/configs'
    all_meta_file_paths=glob.glob(os.path.join(root_meta_file_path,"*","metafile.yml"),recursive=True)
    weights_list = []

    for meta_file_path in all_meta_file_paths:
        with open(meta_file_path,'r') as f:
            yaml_file = yaml.safe_load(f)

            for i in range(len(yaml_file['Models'])):
                try:
                    weights_list.append(yaml_file['Models'][i]['Weights'])
                except:
                    for k,v in yaml_file['Models'][i]['Results'][0]['Metrics'].items():
                        if k== 'Weights':
                            weights_list.append(yaml_file['Models'][i]['Results'][0]['Metrics']['Weights'])
    

    return weights_list