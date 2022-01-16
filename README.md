# nms-simd Benchmark

This benchmark is done in python using google benchmark. 

We are benchmarking results from two different models ([yolov5](https://github.com/ultralytics/yolov5) and [FasterRCNN](https://github.com/onnx/models/tree/master/vision/object_detection_segmentation/faster-rcnn)), six different images ([dog.jpg](https://github.com/pjreddie/darknet/blob/master/data/dog.jpg    ), [eagle.jpg  ](https://github.com/pjreddie/darknet/blob/master/data/eagle.jpg  ), [giraffe.jpg](https://github.com/pjreddie/darknet/blob/master/data/giraffe.jpg), [horses.jpg ](https://github.com/pjreddie/darknet/blob/master/data/horses.jpg ), [kite.jpg](https://github.com/pjreddie/darknet/blob/master/data/kite.jpg   ), [person.jpg ](https://github.com/pjreddie/darknet/blob/master/data/person.jpg )) and six different algorithms (nms-simd, tensorflow, torchvision, [faster-nms](https://www.pyimagesearch.com/faster-non-maximum-suppression-python/), opencv, mmcv). Current results are taken from a Ryzen 2600 machine.

# Requirements

google-benchmark
onnx-runtime
tensorflow
torchvision
opencv
mmcv
numpy

# How to Use

## Weights

Weights should be in root project folder. Download [FasterRCNN](https://github.com/onnx/models/tree/master/vision/object_detection_segmentation/faster-rcnn).
Clone [yolov5](https://github.com/ultralytics/yolov5) whereever you want then export as onnx. Default weights for yolov5 is yolov5x6.  

## Running

First run one of the object detection scripts (yolov5.py or fasterrcnn.py). Then run benchmark.py. Then run the other object detection script, rinse and repeat.

