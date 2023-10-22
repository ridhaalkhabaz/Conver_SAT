from ultralyticsplus import YOLO, render_result

# load model
model = YOLO('keremberke/yolov8n-building-segmentation')

# set model parameters
model.overrides['conf'] = 0.25  # NMS confidence threshold
model.overrides['iou'] = 0.45  # NMS IoU threshold
model.overrides['agnostic_nms'] = False  # NMS class-agnostic
model.overrides['max_det'] = 1000  # maximum number of detections per image

# set image
# image = 'https://datasets-server.huggingface.co/assets/keremberke/satellite-building-segmentation/--/full/train/2/image/image.jpg'
img2 = './trial_tw.jpg'
# perform inference

results = model.predict(img2)

# observe results
print(results[0].boxes)
print(results[0].masks)
render = render_result(model=model, image=img2, result=results[0])
render.show()