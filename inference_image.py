from mmdet.apis import inference_detector
from utils import get_model

import mmcv
import argparse


# Construct the argument parser.
parser = argparse.ArgumentParser(description="tool for object recognition in images")

parser.add_argument(
    '-i','--input',default='mmdetection/demo/demo.jpg',
    help='path to the input file'
)

parser.add_argument(
    '-w','--weights',default='yolov3_mobilenetv2_320_300e_coco',
    help='weight file name'
)

parser.add_argument(
    '-t','--threshold',default=0.5,type=float,
    help='detection threshold for bounding box visualization'
)

args = vars(parser.parse_args()) # will take the input from the commandline and construct a dictionary out of it.


# Build the model.
model= get_model(args['weights'])

img_path = args['input']
image = mmcv.imread(img_path)

# Carry out the infernce.
result = inference_detector(model,image)

# Show the reuslts.
frame = model.show_result(image,result,score_thr=args['threshold'])
mmcv.imshow(frame)

# Initialize a file to save the result
save_name = f"{args['input'].split('/')[-1].split('.')[0]}_{args['weights']}"
mmcv.imwrite(frame,f"ouputs/{save_name}.jpg")