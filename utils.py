import os
import requests
import glob
import yaml

from mmdet.apis import init_detector

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


def get_model(weights_name):
    """
    Either downloads a model or loads one from local path if already
    downloaded using the weight file name ('weights_name') provided.

    :param weights_name: Name of the weight file. Most like in the format:
    **'retinanet_ghm_r50_fpn_1x_coco'**. See **'weights.txt'** to know weight file name formats and downloadable URL formats.

    Returns:
        model: The loaded detection model.
    """
    # Get the list containing all the weight file download URLs.
    weights_list = parse_meta_file()

    download_url = None
    for weights in weights_list:
        if weights_name in weights:
            print(f'Found weights: {weights}\n')
            download_url = weights
            break
        

    assert download_url != None, f"{weights_name} weight file not found!!!"
    
    # Download the checkpoint file.(will only donwload if the checkpoint file is not present locally)
    download_weights(url=download_url,file_save_name=download_url.split('/')[-1])

    checkpoint_file = os.path.join('checkpoint',download_url.split('/')[-1])

    # Build the model using the configuration file.
    config_file = os.path.join(
        'mmdetection/configs',
        download_url.split('/')[-3],
        download_url.split('/')[-2],
        '.py')
    
    model = init_detector(config_file,checkpoint_file)

    return model

def write_weights_txt_file():
    """
    Write all the model URLs to 'weights.txt' to have complete list and choose one of them.add()

    EXECUTE 'utils.py' if 'weights.txt' not already present.
    'python utils.py' command will generate the latest 'weights.txt'
    file according to the cloned mmdetection repository.
    """

    # Get the list containing all the weight file download URLs
    weight_list = parse_meta_file()
    with open('weights.txt','w') as f:
        for weights in weight_list:
            f.writelines(f"{weights}\n")
        f.close()


if __name__ == '__main__':
    write_weights_txt_file()
    weights_list = parse_meta_file()
    print(weights_list[:3])