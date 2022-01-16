from typing import List, Tuple
import onnxruntime as ort
import cv2
import numpy as np
import logging
from input_const import IMAGE_PATH, OD_IMAGE_PATHS, OD_OUTPUT_PATHS, OD_IMAGE_FILES
import time
def preprocess():
    batch = np.ndarray(shape=(len(OD_IMAGE_PATHS), 3, 1280, 1280), dtype=np.float32)
    ratios = np.ndarray(shape=len(OD_IMAGE_PATHS), dtype=np.float32)
    for img_idx, img_name in enumerate(OD_IMAGE_PATHS):
        image = cv2.imread(img_name.__str__())
        ratio = 1280 / max(image.shape[1], image.shape[0])
        ratios[img_idx] = 1 / ratio
        image = cv2.resize(image, (int(ratio * image.shape[1]), int(ratio * image.shape[0])))
        image = image.astype('float32')

        # HWC -> CHW
        image: np.ndarray = np.transpose(image, (2, 0, 1))[::-1]
        image = np.ascontiguousarray(image)

        # Normalize
        # for i in range(image.shape[0]):
        #     image[i, :, :] = image[i, :, :] - mean_vec[i]

        padded_image = np.zeros((3, 1280, 1280), dtype=np.float32)
        padded_image[:, :image.shape[1], :image.shape[2]] = image
        batch[img_idx, :, :, :] = padded_image

    batch /= 255
    return (batch, ratios)

def infer(input: np.ndarray):
    detection = ort.InferenceSession("yolov5x6.onnx", providers=["CPUExecutionProvider"])
    output_names = ["output"]

    output_list = []

    for i in range(input.shape[0]):
        logging.info(f"inferring {OD_IMAGE_FILES[i]}")
        output_list.append(detection.run(output_names, {"images": input[i:i+1]})[0])
    return output_list
    
def postprocess(output_list: List[np.ndarray], img_ratios: np.ndarray):
    boxes_list = []
    scores_list = []
    for output, ratio in zip(output_list, img_ratios):
        output = output[0]
        boxes = np.ndarray(shape=(output.shape[0], 4), dtype=np.float32)
        scores = np.ndarray(shape=(output.shape[0]), dtype=np.float32)
        for ele_idx, element in enumerate(output):
            x1y1x2y2 = np.ndarray(shape=4, dtype=np.float32)
            boxes[ele_idx, 0] = element[0] - element[2] / 2
            boxes[ele_idx, 1] = element[1] - element[3] / 2
            boxes[ele_idx, 2] = element[0] + element[2] / 2
            boxes[ele_idx, 3] = element[1] + element[3] / 2
            boxes[ele_idx] *= ratio
            scores[ele_idx] = element[4]

        boxes_list.append(boxes)
        scores_list.append(scores)
    return (boxes_list, scores_list)

def save(boxes_list: List[np.ndarray], scores_list: List[np.ndarray]):
    for paths, boxes, scores in zip(OD_OUTPUT_PATHS, boxes_list, scores_list):
        np.save(paths["scores"], scores)
        np.save(paths["boxes"], boxes)
        
def main():
    input, ratios = preprocess()
    output_list = infer(input)
    boxes_list, scores_list = postprocess(output_list, ratios)
    save(boxes_list, scores_list)

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    main()