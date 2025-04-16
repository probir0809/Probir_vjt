# Probir_vjt

# Task-1 (Cityscapes Mask Generation)
Processes Cityscapes data: organizes downloaded annotations and images, counts unique classes, and creates segmentation masks with a legend.  
- Download annotations from [Cityscapes Annotations](https://www.cityscapes-dataset.com/file-handling/?packageID=1) and images from [Cityscapes Images](https://www.cityscapes-dataset.com/file-handling/?packageID=3).  
- Update directory paths in `make_data.py`, `count_unique_classes.py`, and `mask_creation.py` as needed.  
- Run `python make_data.py` to copy and rename files, creating a final data directory containing images, JSON files, and (after running) masks.  
- Run `python count_unique_classes.py` to list unique class labels.  
- Run `python mask_creation.py` to generate masks and a legend image.  

# Task-2
Once the dataset is prepared using Task-1, train a semantic segmentation model (UNet) by running `task_2.ipynb`. 
