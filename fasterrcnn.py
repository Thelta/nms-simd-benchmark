from typing import List, Tuple
import onnxruntime as ort
import cv2
import numpy as np
import logging
from input_const import OD_IMAGE_PATHS, OD_OUTPUT_PATHS, OD_IMAGE_FILES
def preprocess():
    mean_vec = np.array([102.9801, 115.9465, 122.7717])

    batch = np.ndarray(shape=(len(OD_IMAGE_PATHS), 3, 800, 800), dtype=np.float32)
    ratios = np.ndarray(shape=len(OD_IMAGE_PATHS), dtype=np.float32)
    for img_idx, img_name in enumerate(OD_IMAGE_PATHS):
        image = cv2.imread(img_name.__str__())
        ratio = 800.0 / max(image.shape[1], image.shape[0])
        ratios[img_idx] = 1 / ratio
        image = cv2.resize(image, (int(ratio * image.shape[1]), int(ratio * image.shape[0])))
        image = image.astype('float32')

        # HWC -> CHW
        image = np.transpose(image, (2, 0, 1))

        # Normalize
        for i in range(image.shape[0]):
            image[i, :, :] = image[i, :, :] - mean_vec[i]

        padded_image = np.zeros((3, 800, 800), dtype=np.float32)
        padded_image[:, :image.shape[1], :image.shape[2]] = image
        batch[img_idx, :, :, :] = padded_image
    return (batch, ratios)

def infer(input: np.ndarray):
    detection = ort.InferenceSession("FasterRCNN-10.onnx", providers=["CPUExecutionProvider"])
    output_names = [x.name for x in detection.get_outputs()]

    detection.get_outputs()
    boxes_list = []
    scores_list = []

    for i in range(0, input.shape[0]):
        logging.info(f"inferring {OD_IMAGE_FILES[i]}")
        (boxes, _, scores) = detection.run(output_names, {"image": input[i]})
        boxes_list.append(boxes)
        scores_list.append(scores)

    return (boxes_list, scores_list)

def postprocess(boxes_list: List[np.ndarray], img_ratios: np.ndarray):
    for boxes, ratio in zip(boxes_list, img_ratios):
        boxes *= ratio

def save(boxes_list: List[np.ndarray], scores_list: List[np.ndarray]):
    for paths, boxes, scores in zip(OD_OUTPUT_PATHS, boxes_list, scores_list):
        np.save(paths["scores"], scores)
        np.save(paths["boxes"], boxes)
        
def main():
    input, ratios = preprocess()
    boxes_list, scores_list = infer(input)
    postprocess(boxes_list, ratios)
    save(boxes_list, scores_list)



if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    main()