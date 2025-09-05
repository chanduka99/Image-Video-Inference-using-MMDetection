from mmdet.apis import inference_detector
from mmdet.utils import register_all_modules
from mmdet.visualization import DetLocalVisualizer
from utils import get_model
import time
import mmcv
import argparse
import os

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


# required in MMDET 3.x
register_all_modules()

# Build the model.
print("\nLoading model...\n")
model= get_model(args['weights'])

img_path = args['input']
image = mmcv.imread(img_path)

# Carry out the infernce.
print("\nInferencing...\n")
d_start_time = time.time()
result = inference_detector(model,image)
d_end_time = time.time()

# # Show the reuslts.
# # frame = model.show_result(image,result,score_thr=args['threshold'])
# frame = show_result_pyplot(model, image, result, score_thr=args['threshold'])
# mmcv.imshow(frame)


# Initialize a file to save reuslts
os.makedirs("outputs",exist_ok=True)
save_name = f"{args['input'].split('/')[-1].split('.')[0]}_{args['weights']}"

# visualize and save 
visualizer = DetLocalVisualizer()
visualizer.dataset_meta = model.dataset_meta

converted_image = mmcv.imconvert(image,'bgr','rgb')
visualizer.add_datasample(
    name="result",
    image=converted_image,
    data_sample=result,
    draw_gt=False,
    show=False,  
    wait_time=0,
    out_file=f"outputs/{save_name}.jpg" 
)

d_exec_time = d_end_time-d_start_time
print(f"detection time: {d_exec_time} seconds")
