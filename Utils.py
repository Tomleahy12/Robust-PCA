import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
from os import chdir
import cv2
import torch
import hashlib
"""
This file contains utility functions used to handle and shape the images into datasets for test. 

Note functions in here may not be used but are left for future references.

"""

def image_standardize(data):
    # Calculate the mean (along the 0-dimension, i.e., for each feature/column)
    mean = data.mean(dim=0, keepdim=True)
    # Standardize the dataset
    standardized_data = data / 255
    return standardized_data

def build_data_set(image_folder, start=0, end=None, standardized=False):
    """
    Takes a range of video frames in image set and combines into a large torch tensor.

    Parameters:
    image_folder (str): The directory path that contains the image files.
    start (int): Starting index of the images to combine. Defaults to 0.
    end (int): Ending index of the images to combine (exclusive). If None, all images from start to the end of the folder will be used.
    standardized (bool): If True, standardizes the data. Default is False.
    """
    # Get the list of image file names
    image_files = sorted([f for f in os.listdir(image_folder) if f.endswith(('.jpg', '.png', '.jpeg', '.bmp'))])
    # Check start and end boundaries
    if start < 0:
        start = 0
    if end is None or end > len(image_files):
        end = len(image_files)

    image_tensors = []

    for image_file in image_files[start:end]:
        image_path = os.path.join(image_folder, image_file)
        with Image.open(image_path) as image:
            image_greyed = np.asarray(image.convert('L'))
            image_tensor = torch.tensor(image_greyed, dtype=torch.float32).flatten()
            image_tensors.append(image_tensor)

    # Stack all image tensors
    data_set = torch.stack(image_tensors)

    if standardized:
        # Placeholder for the standardize_data function; define it according to your requirements
        data_set = image_standardize(data_set)

    return data_set

def md5(fname):
    hash_md5 = hashlib.md5()
    with open(fname, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def check_for_duplicates(image_folder):
    image_files = sorted([f for f in os.listdir(image_folder) if f.endswith(('.jpg', '.png', '.jpeg', '.bmp'))])

    seen_files = {}
    duplicates = []

    for filename in image_files:
        file_path = os.path.join(image_folder, filename)

        file_size = os.path.getsize(file_path)

        if file_size in seen_files:
            original_hash = md5(seen_files[file_size])
            current_hash = md5(file_path)

            if original_hash == current_hash:
                duplicates.append((file_path, seen_files[file_size]))
            else:
                seen_files[file_size] = file_path
        else:
            seen_files[file_size] = file_path

    return duplicates

def rand_noise_uni_int(img_tensor, reduction=200):
    noise = torch.randint(0,reduction,size=(img_tensor.size()))    
    corrupted_img = img_tensor +  noise 
    return corrupted_img

def rand_noise_normal(img_tensor, scale = 1):
    noise = torch.randn_like(img_tensor.float())
    #noise = noise.int()*scale
    corrupted_img =img_tensor + noise * scale
    return corrupted_img,noise

def standardize_data(data):
    # Calculate the mean (along the 0-dimension, i.e., for each feature/column)
    mean = data.mean(dim=0, keepdim=True)
    # Calculate the standard deviation (along the 0-dimension)
    std = data.std(dim=0, keepdim=True)
    # Standardize the dataset
    standardized_data = data / 252
    return standardized_data

def build_corrupted_imageset(image_folder, method, corrupt_param, start=0, end=None, standardized=False):
    """
    Create a dataset of corrupted images.

    :param image_folder: Path to the folder containing image files.
    :param method: 'Remove' or 'Noise'. Determines the corruption method.
    :param corrupt_param: Determines the percentage of pixels to remove 
                          or the scale of noise to add.
    :param start: Index of the first image to process.
    :param end: Index of the last image to process (exclusive).
    :param standardized: Whether to standardize pixel values.
    
    :return: data_set - A torch.Tensor containing the corrupted images.
    """

    image_files = sorted([f for f in os.listdir(image_folder) 
                          if f.endswith(('.jpg', '.png', '.jpeg', '.bmp'))])
    if start < 0:
        start = 0
    if end is None or end > len(image_files):
        end = len(image_files)

    image_tensors = []
    for image_file in image_files[start:end]:
        image_path = os.path.join(image_folder, image_file)

        with Image.open(image_path) as image:
            image_greyed = np.asarray(image.convert('L'))
            image_tensor = torch.tensor(image_greyed, dtype=torch.float32).flatten()

            if method == 'Remove':
                corrupted_image = pixel_remove(image_tensor, corrupt_param)

            elif method == 'Noise':
                corrupted_image = pixel_val_change(image_tensor, scale=corrupt_param)
            else:
                raise ValueError("Invalid method. Use 'Remove' or 'Noise'.")

            image_tensors.append(corrupted_image.unsqueeze(0)) 

    data_set = torch.cat(image_tensors, dim=0)
    if standardized:
        data_set = image_standardize(data_set)

    return data_set
    
def pixel_remove(image: torch.tensor, removal_cutoff: float):
    """
    args: 
    image: 
    removal_cutoff:

    """
    if removal_cutoff >= 1 or removal_cutoff <= 0:
        raise ValueError("removal_cutoff should be a value between 0 and 1 exclusive.")
    indices_matrix = torch.rand_like(image)
    mask = indices_matrix > removal_cutoff
    corrupted_image = torch.where(mask, torch.zeros_like(image), image)
    
    return corrupted_image

    
def pixel_val_change(image: torch.tensor, scale: float):
    """
    :params image: image tensor to be corrupted
    :params removal_cutoff: scale of values to be changed.

    """
    noise = torch.randn_like(image)
    #noise = noise.int()*scale
    corrupted_img =image + noise * scale
    return corrupted_img
    


def create_animated_gif(sparse_matrix, output_filename: str, cut_off = .5, frame_duration=5, image_size=(120, 160)):
    # for some reason, repeatedly running PIL wont work unless imported each time. 
    from PIL import Image
    
    """
    Create an animated GIF from a 2D numpy array where each row is a flattened frame. Image is converted to numpy. Values < 0 are set to 0. Rescale by 255
    
    :param sparse_matrix: 2D numpy array or PyTorch tensor with each row as a flattened image frame
    :param output_filename: Filename to save the GIF
    :param frame_duration: Duration for each frame in the animation in milliseconds
    :param image_size: Size of each frame in the animation
    :return: None
    """
    
    if hasattr(sparse_matrix, 'detach') and hasattr(sparse_matrix, 'cpu'):
        sparse_matrix = sparse_matrix.detach().cpu().numpy()
    elif not isinstance(sparse_matrix, np.ndarray):
        sparse_matrix = sparse_matrix.numpy()
    sparse_matrix = np.abs(sparse_matrix)
    sparse_matrix = np.where(sparse_matrix < cut_off, 0, 255)
    rows, _ = sparse_matrix.shape
    image_sequence = []
    
    for i in range(rows):
        new_image = np.reshape(sparse_matrix[i], image_size).astype(np.uint8) 
        picture = Image.fromarray(new_image)
        image_sequence.append(picture)
    
    
    if len(image_sequence) > 1:
        image_sequence[0].save(
            output_filename,
            save_all=True,
            append_images=image_sequence[1:],
            duration=frame_duration,
            loop=0
        )
        print("GIF saved successfully.")
    else:
        print("Not enough images to create a GIF.")

def return_animated_gif(sparse_matrix, output_filename: str, cut_off = .5, frame_duration=5, image_size=(120, 160)):
    # for some reason, repeatedly running PIL wont work unless imported each time. 
    from PIL import Image
    
    """
    Create an animated GIF from a 2D numpy array where each row is a flattened frame. Image is converted to numpy. Values < 0 are set to 0. Rescale by 255
    
    :param sparse_matrix: 2D numpy array or PyTorch tensor with each row as a flattened image frame
    :param output_filename: Filename to save the GIF
    :param frame_duration: Duration for each frame in the animation in milliseconds
    :param image_size: Size of each frame in the animation
    :return: None
    """
    
    if hasattr(sparse_matrix, 'detach') and hasattr(sparse_matrix, 'cpu'):
        sparse_matrix = sparse_matrix.detach().cpu().numpy()
    elif not isinstance(sparse_matrix, np.ndarray):
        sparse_matrix = sparse_matrix.numpy()
    sparse_matrix = sparse_matrix * 255
    rows, _ = sparse_matrix.shape
    image_sequence = []
    
    for i in range(rows):
        new_image = np.reshape(sparse_matrix[i], image_size).astype(np.uint8) 
        picture = Image.fromarray(new_image)
        image_sequence.append(picture)
    
    
    if len(image_sequence) > 1:
        image_sequence[0].save(
            output_filename,
            save_all=True,
            append_images=image_sequence[1:],
            duration=frame_duration,
            loop=0
        )
        print("GIF saved successfully.")
    else:
        print("Not enough images to create a GIF.")

def percentile_animated_gif(sparse_matrix, output_filename: str, percentile_cutoff=.9, frame_duration=50, image_size=(120, 160)):
    """
    Create an animated GIF from a 2D numpy array where each row is a flattened frame.
    Only the top percentile of the pixel values will be kept.
    
    This means that, e.g., if `percentile_cutoff` is set to 90, only the top 10% of pixel
    intensities will be included in the GIF (thresholded at the 90th percentile).
    
    :param sparse_matrix: 2D numpy array or PyTorch tensor with each row as a flattened image frame
    :param output_filename: Filename to save the GIF
    :param percentile_cutoff: The percentile to cut off pixel values (e.g., 90 for top 10%)
    :param frame_duration: Duration for each frame in the animation in milliseconds
    :param image_size: Size of each frame in the animation
    :return: None
    """
    
    # Convert PyTorch tensors to numpy arrays if needed
    if hasattr(sparse_matrix, 'detach') and hasattr(sparse_matrix, 'cpu'):
        sparse_matrix = sparse_matrix.detach().cpu().numpy()
    elif not isinstance(sparse_matrix, np.ndarray):
        sparse_matrix = sparse_matrix.numpy()

    flattened_frames = np.abs(sparse_matrix.flatten())
    value_at_percentile = np.percentile(flattened_frames, percentile_cutoff)
    sparse_matrix[sparse_matrix < value_at_percentile] = 0
    sparse_matrix[sparse_matrix >= value_at_percentile] = 255
    sparse_matrix = sparse_matrix.astype(np.uint8)

    frames = []
    for i in range(sparse_matrix.shape[0]):
        frame = np.reshape(sparse_matrix[i], image_size)
        image = Image.fromarray(frame, mode='L') 
        frames.append(image)
    if frames:
        frames[0].save(
            output_filename,
            save_all=True,
            append_images=frames[1:],
            duration=frame_duration,
            loop=0
        )
        print(f"GIF saved successfully as {output_filename}.")
    else:
        print("Sparse matrix is empty, no frames to create GIF.")