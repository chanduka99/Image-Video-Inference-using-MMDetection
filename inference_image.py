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
    '-t','--treshold',default=0.5,type=float,
    help='detection treshold for bounding box visualization'
)

args = vars(parser.parse_args) # will take the input from the commandline and construct a dictionary out of it.
