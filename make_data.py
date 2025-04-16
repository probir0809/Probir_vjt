

import os
import shutil

def copy_files(source_dir, destination_dir, extension):
    """
    Copy all files with the specified file extension from the source directory (and its subdirectories)
    to the destination directory.

    Parameters:
        source_dir (str): The root directory to search for files.
        destination_dir (str): The target directory where the files will be copied.
        extension (str): The file extension to look for (e.g., ".json", ".png").
    """
    os.makedirs(destination_dir, exist_ok=True)
    
    # Walk through source_dir recursively
    for root, _, files in os.walk(source_dir):
        for file in files:
            # Check file extension (case-insensitive)
            if file.lower().endswith(extension.lower()):
                src_file = os.path.join(root, file)
                dst_file = os.path.join(destination_dir, file)
                shutil.copy(src_file, dst_file)
                print(f"Copied: {src_file} -> {dst_file}")

def rename_files(directory, target_suffix, substring_to_remove):
    """
    Rename files in the specified directory by removing a given substring from files that end with a target suffix.

    For example, if files end with '_leftImg8bit.png', calling
    rename_files(directory, '_leftImg8bit.png', '_leftImg8bit')
    will rename "example_leftImg8bit.png" to "example.png".

    Parameters:
        directory (str): The directory containing files to rename.
        target_suffix (str): Only files that end with this suffix will be processed.
        substring_to_remove (str): The substring to remove from the filename.
    """
    # Iterate over all files in the given directory
    for filename in os.listdir(directory):
        if filename.endswith(target_suffix):
            new_filename = filename.replace(substring_to_remove, "")
            src_path = os.path.join(directory, filename)
            dst_path = os.path.join(directory, new_filename)
            os.rename(src_path, dst_path)
            print(f"Renamed: {filename} -> {new_filename}")

if __name__ == "__main__":
    # Example usage:
    
    # Define the directories based on your current setup.
    train_json_source_dir = '/media/pro/28d83ea8-54ca-4f9d-b09d-604d9edec39f/probir/img_seg/gtFine_trainvaltest/gtFine/train'
    train_json_destination_dir = '/media/pro/28d83ea8-54ca-4f9d-b09d-604d9edec39f/probir/img_seg/data/train/json_file'
    train_image_source_dir = '/media/pro/28d83ea8-54ca-4f9d-b09d-604d9edec39f/probir/img_seg/leftImg8bit_trainvaltest/leftImg8bit/train'
    train_image_destination_dir = '/media/pro/28d83ea8-54ca-4f9d-b09d-604d9edec39f/probir/img_seg/data/train/image'

    test_json_source_dir = '/media/pro/28d83ea8-54ca-4f9d-b09d-604d9edec39f/probir/img_seg/gtFine_trainvaltest/gtFine/test'
    test_json_destination_dir = '/media/pro/28d83ea8-54ca-4f9d-b09d-604d9edec39f/probir/img_seg/data/test/json_file'
    test_image_source_dir = '/media/pro/28d83ea8-54ca-4f9d-b09d-604d9edec39f/probir/img_seg/leftImg8bit_trainvaltest/leftImg8bit/test'
    test_image_destination_dir = '/media/pro/28d83ea8-54ca-4f9d-b09d-604d9edec39f/probir/img_seg/data/test/image'

    val_json_source_dir = '/media/pro/28d83ea8-54ca-4f9d-b09d-604d9edec39f/probir/img_seg/gtFine_trainvaltest/gtFine/val'
    val_json_destination_dir = '/media/pro/28d83ea8-54ca-4f9d-b09d-604d9edec39f/probir/img_seg/data/val/json_file'
    val_image_source_dir = '/media/pro/28d83ea8-54ca-4f9d-b09d-604d9edec39f/probir/img_seg/leftImg8bit_trainvaltest/leftImg8bit/val'
    val_image_destination_dir = '/media/pro/28d83ea8-54ca-4f9d-b09d-604d9edec39f/probir/img_seg/data/val/image'
    
    #train
    copy_files(train_json_source_dir , train_json_destination_dir, '.json')
    copy_files(train_image_source_dir, train_image_destination_dir, '.png')

    #train
    copy_files(test_json_source_dir, test_json_destination_dir, '.json')
    copy_files(test_image_source_dir, test_image_destination_dir, '.png')

    #train
    copy_files(val_json_source_dir, val_json_destination_dir, '.json')
    copy_files(val_image_source_dir, val_image_destination_dir, '.png')
    
    # rename images, json file:
    rename_files(train_image_destination_dir, '_leftImg8bit.png', '_leftImg8bit')
    rename_files(train_json_destination_dir, '_gtFine_polygons.json', '_gtFine_polygons')

    rename_files(test_image_destination_dir, '_leftImg8bit.png', '_leftImg8bit')
    rename_files(test_json_destination_dir, '_gtFine_polygons.json', '_gtFine_polygons')

    rename_files(val_image_destination_dir, '_leftImg8bit.png', '_leftImg8bit')
    rename_files(val_json_destination_dir, '_gtFine_polygons.json', '_gtFine_polygons')

