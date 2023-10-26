from ultralyticsplus import YOLO, render_result
import argparse
import os 

# load model
model = YOLO('keremberke/yolov8n-building-segmentation')

#initialize parser and set model parameters
parser = argparse.ArgumentParser(description='Query Replacer')
parser.add_argument('--model', type=str, default='YOLO')
parser.add_argument('--conf', type=float, default=0.25) # NMS confidence threshold
parser.add_argument('--iou', type=float, default=0.45) # NMS IoU threshold
parser.add_argument('--agnostic_nms', action='store_false') # NMS class-agnostic
parser.add_argument('--max_det', type=int, default=1000) # maximum number of detections per image

args = parser.parse_args()

# set model parameters
# model.overrides['conf'] = args.conf   # NMS confidence threshold
# model.overrides['iou'] = args.iou # NMS IoU threshold
# model.overrides['agnostic_nms'] = args.agnostic_nms # NMS class-agnostic
# model.overrides['max_det'] = args.max_det# maximum number of detections per image

# set image
# image = 'https://datasets-server.huggingface.co/assets/keremberke/satellite-building-segmentation/--/full/train/2/image/image.jpg'
directory = '/Users/ridhaalkhabaz/Documents/mlds/test'
for filename in os.listdir(directory):
    f = os.path.join(directory, filename)
    # checking if it is a file
    if os.path.isfile(f):
        img = f 
        # perform detection 
        results = model.predict(img)
        print(len(results[0].boxes))
        # please only run the following on subset of the data for visualization 
        render = render_result(model=model, image=img, result=results[0])
        render.show()
        


# img2 = './trial_tw.jpg'
# # perform inference

# results = model.predict(img2)

# # observe results
# print(results[0].boxes)
# print(results[0].masks)
# render = render_result(model=model, image=img2, result=results[0])
# render.show()