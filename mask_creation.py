import os
import json
import cv2
import numpy as np

# ----------------------- Configuration -----------------------
# Update these paths with your actual directory paths.
legend_output_dir = '/media/pro/28d83ea8-54ca-4f9d-b09d-604d9edec39f/probir/img_seg/data' # Where to save the legend image

train_image_dir = '/media/pro/28d83ea8-54ca-4f9d-b09d-604d9edec39f/probir/img_seg/data/train/images'         # Directory containing input images
train_json_dir = '/media/pro/28d83ea8-54ca-4f9d-b09d-604d9edec39f/probir/img_seg/data/train/json_file'           # Directory containing corresponding JSON files
train_mask_output_dir = '/media/pro/28d83ea8-54ca-4f9d-b09d-604d9edec39f/probir/img_seg/data/train/mask' # Where to save the generated mask images


test_image_dir = '/media/pro/28d83ea8-54ca-4f9d-b09d-604d9edec39f/probir/img_seg/data/test/images'         # Directory containing input images
test_json_dir = '/media/pro/28d83ea8-54ca-4f9d-b09d-604d9edec39f/probir/img_seg/data/test/json_file'           # Directory containing corresponding JSON files
test_mask_output_dir = '/media/pro/28d83ea8-54ca-4f9d-b09d-604d9edec39f/probir/img_seg/data/test/mask' # Where to save the generated mask images

val_image_dir = '/media/pro/28d83ea8-54ca-4f9d-b09d-604d9edec39f/probir/img_seg/data/val/images'         # Directory containing input images
val_json_dir = '/media/pro/28d83ea8-54ca-4f9d-b09d-604d9edec39f/probir/img_seg/data/val/json_file'           # Directory containing corresponding JSON files
val_mask_output_dir = '/media/pro/28d83ea8-54ca-4f9d-b09d-604d9edec39f/probir/img_seg/data/val/mask' # Where to save the generated mask images

# Create output directories if they don't exist.
os.makedirs(train_mask_output_dir, exist_ok=True)
os.makedirs(test_mask_output_dir, exist_ok=True)
os.makedirs(val_mask_output_dir, exist_ok=True)
os.makedirs(legend_output_dir, exist_ok=True)

# ----------------- Define Label Groups and Grayscale Values -----------------
# Each group is mapped to a set of labels (all in lowercase for normalization).
large_vehicle_labels = {'bus', 'truck', 'truckgroup', 'car', 'cargroup', 'caravan', 'trailer'}
two_wheeler_labels   = {'motorcycle', 'motorcyclegroup', 'rider', 'ridergroup', 'bicycle', 'bicyclegroup'}
people_labels        = {'person', 'persongroup'}

# Choose distinct grayscale pixel values for each class.
# (These values are arbitrary but must be distinct and visually differentiable.)
GRAYSCALE_LARGE_VEHICLE = 85
GRAYSCALE_TWO_WHEELER   = 170
GRAYSCALE_PEOPLE        = 255

# Priority order is: People > Two Wheeler > Large Vehicle
# So we will draw in that order (first large vehicle, then two wheeler, and lastly people).

# ----------------- Function Definitions -----------------
def process_image_and_annotations(image_path, json_path, mask_output_path):
    """
    Process one image and its corresponding JSON file to create a segmentation mask.
    Only annotations for large vehicles, two wheelers, and people are considered.
    Overlapping polygons are merged and the final mask is drawn using the priority:
    People > Two Wheeler > Large Vehicle.
    """
    # Load image and get its dimensions.
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image {image_path}")
        return
    height, width = image.shape[:2]
    
    # Initialize a blank grayscale mask.
    mask = np.zeros((height, width), dtype=np.uint8)
    
    # Load JSON annotation.
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error reading JSON {json_path}: {e}")
        return
    
    # Prepare lists to accumulate polygons per group.
    large_vehicle_polys = []
    two_wheeler_polys = []
    people_polys = []
    
    # Iterate through each annotated object.
    for obj in data.get('objects', []):
        label = obj.get('label', '').lower().strip()
        # Ignore annotations not in our target groups.
        if label in large_vehicle_labels:
            poly = obj.get('polygon', [])
            if poly:
                pts = np.array(poly, dtype=np.int32).reshape((-1, 1, 2))
                large_vehicle_polys.append(pts)
        elif label in two_wheeler_labels:
            poly = obj.get('polygon', [])
            if poly:
                pts = np.array(poly, dtype=np.int32).reshape((-1, 1, 2))
                two_wheeler_polys.append(pts)
        elif label in people_labels:
            poly = obj.get('polygon', [])
            if poly:
                pts = np.array(poly, dtype=np.int32).reshape((-1, 1, 2))
                people_polys.append(pts)
        else:
            continue  # Ignore labels not in our defined groups.
    
    # Draw polygons in order of increasing priority:
    # First, draw large vehicle regions.
    if large_vehicle_polys:
        cv2.fillPoly(mask, large_vehicle_polys, GRAYSCALE_LARGE_VEHICLE)
    # Then, draw two wheeler regions over any overlapping large vehicle areas.
    if two_wheeler_polys:
        cv2.fillPoly(mask, two_wheeler_polys, GRAYSCALE_TWO_WHEELER)
    # Finally, draw people regions over everything.
    if people_polys:
        cv2.fillPoly(mask, people_polys, GRAYSCALE_PEOPLE)
    
    # Save the resulting mask image.
    cv2.imwrite(mask_output_path, mask)
    print(f"Saved mask: {mask_output_path}")

def create_legend_image(legend_output_path):
    """
    Create and save a legend image that visually maps the grayscale shades to the class labels.
    The legend consists of colored (gray) boxes with text overlay.
    """
    # Create a blank white image; here we choose a size that fits three labels.
    legend = np.ones((200, 400, 3), dtype=np.uint8) * 255
    
    # Draw rectangles with the respective grayscale values.
    # Large Vehicle rectangle.
    cv2.rectangle(legend, (20, 20), (80, 60), (GRAYSCALE_LARGE_VEHICLE, GRAYSCALE_LARGE_VEHICLE, GRAYSCALE_LARGE_VEHICLE), -1)
    cv2.putText(legend, "Large Vehicle", (90, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    
    # Two Wheeler rectangle.
    cv2.rectangle(legend, (20, 80), (80, 120), (GRAYSCALE_TWO_WHEELER, GRAYSCALE_TWO_WHEELER, GRAYSCALE_TWO_WHEELER), -1)
    cv2.putText(legend, "Two Wheeler", (90, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    
    # People rectangle.
    cv2.rectangle(legend, (20, 140), (80, 180), (GRAYSCALE_PEOPLE, GRAYSCALE_PEOPLE, GRAYSCALE_PEOPLE), -1)
    cv2.putText(legend, "People", (90, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    
    # Save the legend image.
    cv2.imwrite(legend_output_path, legend)
    print(f"Saved legend image: {legend_output_path}")

def main():
    # Iterate over images in the image directory.
    for filename in os.listdir(test_image_dir):
        if filename.lower().endswith(('.png')):
            image_path = os.path.join(test_image_dir, filename)
            base_name = os.path.splitext(filename)[0]
            json_filename = base_name + '.json'
            json_path = os.path.join(test_json_dir, json_filename)
            if not os.path.exists(json_path):
                print(f"JSON file not found for {filename}. Skipping.")
                continue
            mask_output_path = os.path.join(test_mask_output_dir, base_name + '.png')
            process_image_and_annotations(image_path, json_path, mask_output_path)
    for filename in os.listdir(train_image_dir):
        if filename.lower().endswith(('.png')):
            image_path = os.path.join(train_image_dir, filename)
            # Derive the base filename (e.g., aachen_000119_000019) and form the JSON filename.
            base_name = os.path.splitext(filename)[0]
            json_filename = base_name + '.json'
            json_path = os.path.join(train_json_dir, json_filename)
            
            # Check whether the corresponding JSON exists.
            if not os.path.exists(json_path):
                print(f"JSON file not found for {filename}. Skipping.")
                continue
            
            # Define the output mask filename.
            mask_output_path = os.path.join(train_mask_output_dir, base_name + '.png')
            process_image_and_annotations(image_path, json_path, mask_output_path)
    for filename in os.listdir(val_image_dir):
        if filename.lower().endswith(('.png')):
            image_path = os.path.join(val_image_dir, filename)
            # Derive the base filename (e.g., aachen_000119_000019) and form the JSON filename.
            base_name = os.path.splitext(filename)[0]
            json_filename = base_name + '.json'
            json_path = os.path.join(val_json_dir, json_filename)
            
            # Check whether the corresponding JSON exists.
            if not os.path.exists(json_path):
                print(f"JSON file not found for {filename}. Skipping.")
                continue
            
            # Define the output mask filename.
            mask_output_path = os.path.join(val_mask_output_dir, base_name + '.png')
            process_image_and_annotations(image_path, json_path, mask_output_path)
    # Create a legend image and save it.
    legend_output_path = os.path.join(legend_output_dir, "legend.png")
    create_legend_image(legend_output_path)

if __name__ == '__main__':
    main()
