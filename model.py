import os
import sys
import json
import numpy as np
import time
from PIL import Image, ImageDraw

# Root directory of the project
ROOT_DIR = os.path.abspath('../../')

# Import mrcnn libraries
sys.path.append(ROOT_DIR) 
import mrcnn.utils as utils
import mrcnn.model as modellib
from mrcnn.config import Config
from mrcnn import visualize

# Path to trained weights file
BALLOON_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_balloon.h5")

# Directory to save logs and model checkpoints
DEFAULT_LOGS_DIR = os.path.abspath("./logs/")

# Directory to images and annotations 
DATASET_DIR = os.path.abspath("./dataset/png/")

class CustomConfig(Config):
    NAME = "building"
    BACKBONE = "resnet50"
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 2
    IMAGE_MIN_DIM = 640
    IMAGE_MAX_DIM = 640
    TRAIN_ROIS_PER_IMAGE = 32
    USE_MINI_MASK = True
    STEPS_PER_EPOCH = 500
    VALIDATION_STEPS = 50
    MAX_GT_INSTANCES = 500
    DETECTION_MAX_INSTANCES = 500

config = CustomConfig()

class CustomDataset(utils.Dataset):
    
    def load_data(self, annotation_path, images_path):
        """ Load the coco-like dataset from json
        Args:
            annotation_path: The path to the coco annotations json file
            images_path: The directory holding the images referred to by the json file
        """
        # Load json from file
        annotations_json = json.load(open(os.path.join(DATASET_DIR, annotation_path)))
        
        # Add the class names using the base method from utils.Dataset
        source_name = "building"
        for category in annotations_json['categories']:
            class_id = category['id']
            class_name = category['name']
            if class_id > 0:
                self.add_class(source_name, class_id, class_name)
        
        # Get all annotations
        annotations = {}
        for annotation in annotations_json['annotations']:
            image_id = annotation['image_id']
            if image_id not in annotations:
                annotations[image_id] = []
            annotations[image_id].append(annotation)
        
        # Get all images and add them to the dataset
        seen_images = {}
        for image in annotations_json['images']:
            image_id = image['id']
            if image_id in seen_images:
                print("Warning: Skipping duplicate image id: {}".format(image))
            else:
                seen_images[image_id] = image
                try:
                    image_file_name = image['file_name']
                    image_width = image['width']
                    image_height = image['height']
                except KeyError as key:
                    print("Warning: Skipping image (id: {}) with missing key: {}".format(image_id, key))
                
                image_path = os.path.abspath(os.path.join(DATASET_DIR, images_path, image_file_name))
                image_annotations = annotations[image_id]
                
                # Add the image using the base method from utils.Dataset
                self.add_image(
                    source=source_name,
                    image_id=image_id,
                    path=image_path,
                    width=image_width,
                    height=image_height,
                    annotations=image_annotations
                )
                
    def load_mask(self, image_id):
        """ Load instance masks for the given image.
        MaskRCNN expects masks in the form of a bitmap [height, width, instances].
        Args:
            image_id: The id of the image to load masks for
        Returns:
            masks: A bool array of shape [height, width, instance count] with
                one mask per instance.
            class_ids: a 1D array of class IDs of the instance masks.
        """
        image_info = self.image_info[image_id]
        annotations = image_info['annotations']
        instance_masks = []
        class_ids = []
        
        for annotation in annotations:
            class_id = annotation['category_id']
            mask = Image.new('1', (image_info['width'], image_info['height']))
            mask_draw = ImageDraw.ImageDraw(mask, '1')
            for segmentation in annotation['segmentation']:
                mask_draw.polygon(segmentation, fill=1)
                bool_array = np.array(mask) > 0
                instance_masks.append(bool_array)
                class_ids.append(class_id)

        mask = np.dstack(instance_masks)
        class_ids = np.array(class_ids, dtype=np.int32)
        
        return mask, class_ids

dataset_train = CustomDataset()
dataset_train.load_data('train/train_annotations.json', 'train/images/')
dataset_train.prepare()

dataset_val = CustomDataset()
dataset_val.load_data('val/val_annotations.json', 'val/images/')
dataset_val.prepare()

model = modellib.MaskRCNN(mode="training", config=config,
                          model_dir=DEFAULT_LOGS_DIR)

init_with = "balloon"

if init_with == "imagenet":
    model.load_weights(model.get_imagenet_weights(), by_name=True)
elif init_with == "balloon":
    model.load_weights(BALLOON_WEIGHTS_PATH, by_name=True)
elif init_with == "last":
    model.load_weights(model.find_last(), by_name=True)

start_train = time.time()
model.train(dataset_train, dataset_val, 
            learning_rate=config.LEARNING_RATE / 10, 
            epochs=10, 
            layers='all')
end_train = time.time()
minutes = round((end_train - start_train) / 60, 2)
print(f'Training took {minutes} minutes')

class InferenceConfig(CustomConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    IMAGE_MIN_DIM = 640
    IMAGE_MAX_DIM = 640
    DETECTION_MIN_CONFIDENCE = 0.85
    

inference_config = InferenceConfig()

model = modellib.MaskRCNN(mode="inference", config=inference_config,
                          model_dir=DEFAULT_LOGS_DIR)

model_path = model.find_last()
model.load_weights(model_path, by_name=True)
model.keras_model.save('./output/maskrcnn_building.h5')