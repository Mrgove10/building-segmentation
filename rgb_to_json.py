import glob
import argparse
from create_annotations import *

# Label ids of the dataset
category_ids = {
    "road":0,
    "lane": 1,
    "undrivable": 2,
    "movable": 3,
    "my car": 4
}

# Define which colors match which categories in the images
category_colors = {
    '(64, 32, 32)': 0, # road 
    '(255, 0, 0)': 1, # lane markings
    '(128, 128, 96)': 2, # undrivable
    '(0, 255, 102)': 3, # movable 
    '(204, 0, 255)': 4 # my car
}

# Define the ids that are a multiplolygon. In our case: wall, roof and sky
multipolygon_ids = [0, 1, 2, 3, 4]

# Get "images" and "annotations" info 
def images_annotations_info(maskpath):
    # This id will be automatically increased as we go
    annotation_id = 0
    image_id = 0
    annotations = []
    images = []
    
    for mask_image in glob.glob(maskpath + "*.png"):
        # The mask image is *.png but the original image is *.jpg.
        # We make a reference to the original file in the COCO JSON file
        original_file_name = os.path.basename(mask_image)

        # Open the image and (to be sure) we convert it to RGB
        mask_image_open = Image.open(mask_image).convert("RGB")
        w, h = mask_image_open.size

        # "images" info 
        print(original_file_name)
        image = create_image_annotation(original_file_name, w, h, image_id)
        images.append(image)
        
        sub_masks = create_sub_masks(mask_image_open, w, h)
        for color, sub_mask in sub_masks.items():
            category_id = category_colors[color]

            # "annotations" info
            polygons, segmentations = create_sub_mask_annotation(sub_mask)

            # Check if we have classes that are a multipolygon
            if category_id in multipolygon_ids:
                # Combine the polygons to calculate the bounding box and area
                multi_poly = MultiPolygon(polygons)
                                
                annotation = create_annotation_format(multi_poly, segmentations, image_id, category_id, annotation_id)

                annotations.append(annotation)
                annotation_id += 1
            else:
                for i in range(len(polygons)):
                    # Cleaner to recalculate this variable
                    segmentation = [np.array(polygons[i].exterior.coords).ravel().tolist()]
                    
                    annotation = create_annotation_format(polygons[i], segmentation, image_id, category_id, annotation_id)
                    
                    annotations.append(annotation)
                    annotation_id += 1
        
        image_id += 1
    
    return images, annotations, annotation_id

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='A test program.')
    parser.add_argument("maskpath", help="Prints the supplied argument.")
    args = parser.parse_args()

    print(args.maskpath)

    # Get the standard COCO JSON format
    coco_format = get_coco_json_format()
    
    for keyword in ["train", "val"]: #, "test"]:
        mask_path = f"{args.maskpath}{keyword}_mask/"
        print(mask_path)

        # Create category section
        coco_format["categories"] = create_category_annotation(category_ids)
    
        # Create images and annotations sections
        coco_format["images"], coco_format["annotations"], annotation_cnt = images_annotations_info(mask_path)

        with open(f"{keyword}.json","w") as outfile:
            json.dump(coco_format, outfile)
        
        print(f"Created {annotation_cnt} annotations for images in folder: {mask_path}")