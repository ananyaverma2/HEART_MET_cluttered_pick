a = [{'boxes': ([[248.3356, 116.8230, 370.2843, 273.9492],
        [121.2937,  64.1053, 189.4675, 115.8273],
        [135.6489,  96.2212, 256.1018, 250.5652],
        [132.1856, 100.4806, 257.2313, 251.5852],
        [135.1593, 103.4020, 257.4217, 239.4028],
        [328.5329, 248.3300, 440.3280, 361.3508],
        [319.8617, 258.7409, 441.7427, 359.0014],
        [123.7075,  29.6284, 160.1677,  75.3327],
        [119.3406,  27.7742, 182.1990, 123.9923],
        [251.6660, 117.4839, 372.5842, 262.3557],
        [254.8138, 114.8885, 374.2295, 265.4638],
        [358.9767,   4.0159, 483.4856, 211.1230]]), 
        'labels': ([47,  3, 47, 70, 51,  3, 70,  3,  3, 70, 44,  8]), 
        'scores': ([0.8283, 0.5669, 0.5024, 0.3657, 0.1697, 0.1490, 0.1392, 0.1121, 0.0989,
        0.0912, 0.0728, 0.0500])}]

COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

print("Total number of categories: ", len(COCO_INSTANCE_CATEGORY_NAMES))

detected_object_list = []
detected_object_score = []
print("---------------------------")
print("Name of the object, Score\n")
for idx, value in enumerate(a[0]['labels']):
    object_name = COCO_INSTANCE_CATEGORY_NAMES[value]
    score = a[0]['scores'][idx]
    detected_object_list.append(object_name)
    detected_object_score.append(score)
    print("{}, {}".
        format(object_name, score))


