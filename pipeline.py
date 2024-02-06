import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tempfile
from sklearn.model_selection import train_test_split
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
import torchvision
import torch.nn.functional as F
from torch.autograd import Variable

from PIL import Image
import cv2
import albumentations as A
import pydicom
from matplotlib.figure import Figure


import time
import os
from tqdm.notebook import tqdm

from torchsummary import summary
import segmentation_models_pytorch as smp

model_path = 'Unet-vt.pt'
input_path = 'baiyongsheng.dcm'

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

model = torch.load(model_path, map_location=device)

def apply_clahe(image):
    # Create a CLAHE object (Arguments are optional)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(image)

def process_image(image):
    # Convert to grayscale
    # Read the image in color
    color_img = image  # cv2.imread(image_path, cv2.IMREAD_COLOR)
    # Convert to grayscale
    gray_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)
    # Apply Gaussian Blur with a 9x9 kernel
    blurred_img = cv2.GaussianBlur(gray_img, (9, 9), 0)
    # Apply Canny Edge Detection
    edges = cv2.Canny(blurred_img, 100, 200)
    # Overlay edges in red on the original image
    color_img[edges > 0] = [255, 0, 0]
    return color_img

def process_dcm_image(dcm_path):
    ds = pydicom.dcmread(dcm_path)
    img = ds.pixel_array.astype(np.float32)

    img_normalized = img / np.max(img)
    mean = np.mean(img_normalized)
    std = np.std(img_normalized)
    img_standardized = (img_normalized - mean) / std

    # Apply CLAHE
    img_clahe = apply_clahe((img_standardized * 255).astype(np.uint8))
    img_denoised = cv2.GaussianBlur(img_clahe, (5, 5), 0)

    # Convert to color image for edge detection
    color_img = cv2.cvtColor(img_denoised, cv2.COLOR_GRAY2BGR)
    img_processed = process_image(color_img)

    return img_processed

class DicomTest:
    def __init__(self, img_path, transform=None):
        self.img_path = img_path
        self.transform = transform

    def get_data(self):
        img = process_dcm_image(self.img_path)
        # img = cv2.imread(self.img_path)

        if self.transform is not None:
            # Update this line
            aug = self.transform(image=img)
            img = Image.fromarray(aug['image'])
        else:
            img = Image.fromarray(img)

        return img


def calculate_dicom_image_area(dicom_file_path):
    """
    Calculate the area of a DICOM image in square centimeters.

    Parameters:
    dicom_file_path (str): The file path to the DICOM image.

    Returns:
    float: The area of the DICOM image in square centimeters.
    """
    # Load the DICOM file
    ds = pydicom.dcmread(dicom_file_path)

    # Ensure Pixel Spacing attribute is present
    if 'PixelSpacing' not in ds:
        raise ValueError("DICOM file does not contain 'Pixel Spacing' information.")

    # Get pixel spacing values (returns a list [row spacing, column spacing] in mm)
    pixel_spacing = ds.PixelSpacing

    # Convert pixel spacing from mm to cm
    pixel_spacing_cm = [float(spacing) / 10 for spacing in pixel_spacing]

    # Get the number of rows and columns in the image
    num_rows = ds.Rows
    num_columns = ds.Columns

    # Calculate the width and height in cm
    width_cm = num_columns * pixel_spacing_cm[1]  # Column spacing for width
    height_cm = num_rows * pixel_spacing_cm[0]  # Row spacing for height

    # Calculate and return the area in square cm
    area_cm2 = width_cm * height_cm
    return area_cm2

# calculate the hu value
def calculate_hu_values_by_segment(dicom_path, segmented_array):
    """
    Calculate the average HU values for each segment in a DICOM image.
    
    Parameters:
    - dicom_path: Path to the DICOM file.
    - segmented_array: A 2D numpy array with segmentation results.
    
    Returns:
    A dictionary with segment values as keys and average HU values as values.
    """
    # Load the DICOM file
    ds = pydicom.dcmread(dicom_path)
    
    # Ensure the DICOM file has pixel data
    if 'PixelData' not in ds:
        raise ValueError("DICOM file does not contain pixel data.")
    
    # Convert pixel data to a numpy array
    image_array = ds.pixel_array.astype(np.float64)
    
    # Apply the rescale slope and intercept to convert to HU
    rescale_slope = ds.RescaleSlope if 'RescaleSlope' in ds else 1
    rescale_intercept = ds.RescaleIntercept if 'RescaleIntercept' in ds else 0
    hu_image = image_array * rescale_slope + rescale_intercept
    
    # Calculate average HU values for each segment
    unique_segments = np.unique(segmented_array)
    average_hu_values = {}
    
    for segment in unique_segments:
        mask = segmented_array == segment
        segment_hu_values = hu_image[mask]
        
        # Optional: Filter or handle unexpected HU values
        # For example, exclude HU values outside expected range if necessary
        # segment_hu_values = segment_hu_values[(segment_hu_values >= HU_MIN) & (segment_hu_values <= HU_MAX)]
        
        average_hu = np.mean(segment_hu_values) if segment_hu_values.size > 0 else 'N/A'
        average_hu_values[segment] = average_hu
    
    return average_hu_values


# Usage example
t_test = A.Resize(704, 1056, interpolation=cv2.INTER_NEAREST)

single_test = DicomTest(input_path, transform=t_test)
image = single_test.get_data()
image_array = np.array(image)

def predict_image_mask_miou(model, image, mean=[0.485], std=[0.224]):
    #print(image)
    model.eval()
    t = T.Compose([T.ToTensor()])
    image = t(image)
    model.to(device); image=image.to(device)
    with torch.no_grad():

        image = image.unsqueeze(0)

        output = model(image)

        masked = torch.argmax(output, dim=1)
        masked = masked.cpu().squeeze(0)
    return masked

def pixel_accuracy(output, mask):
    with torch.no_grad():
        output = torch.argmax(F.softmax(output, dim=1), dim=1)
        correct = torch.eq(output, mask).int()

        accuracy = float(correct.sum()) / float(correct.numel())
    return accuracy

def predict_image_mask_pixel(model, image, mean=[0.485], std=[0.224]):
    model.eval()
    t = T.Compose([T.ToTensor()])
    image = t(image)
    model.to(device); image=image.to(device)
    with torch.no_grad():

        image = image.unsqueeze(0)
        output = model(image)
        masked = torch.argmax(output, dim=1)
        masked = masked.cpu().squeeze(0)
    return masked

def calculate_percentages(matrix):

    total_counts = {0: 0, 1: 0, 2: 0, 3: 0, 'others': 0}
    total_elements = 0

    for row in matrix:
        # Check if the row contains the number 1 to avoid IndexError
        ones_indices = np.where(row == 0)[0]
        if ones_indices.size > 0:
            first_one_index = ones_indices[0]
            last_one_index = ones_indices[-1]

            # Count the elements between the first and last one
            for element in row[first_one_index + 1:last_one_index]:
                if element in total_counts:
                    total_counts[element] += 1
                else:
                    total_counts['others'] += 1
                total_elements += 1

    # Calculate and print the percentages with color representation
    color_map = {0: "Blue", 1: "Red", 2: "Yellow", 3: "Green", 'others': "Not Sketched"}
    accuracies = []
    for key in total_counts:
        percentage = (total_counts[key] / total_elements) * 100 if total_elements else 0
        accuracies.append(percentage)
        print(f"{color_map[key]} percentage: {percentage:.2f}%")
    return accuracies

def overlay_mask_on_image(image, mask):
    """
    Overlay a mask onto an image.

    :param image: Original image
    :param mask: 2-D mask array
    :return: Image with mask overlaid
    """
    color_map = {
        0: [255, 0, 0],    # Blue
        1: [0, 0, 255],    # Red
        2: [0, 255, 255],  # Yellow
        3: [0, 255, 0],    # Green
    }

    if len(image.shape) == 2:  # Check if the image is grayscale
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    colored_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    for value, color in color_map.items():
        colored_mask[mask == value] = color

    overlaid_image = cv2.addWeighted(image, 0.7, colored_mask, 0.3, 0)
    return overlaid_image


def plot_segmented_mask_and_ground_truth(test_set, model, top_n=10):
    """
    Plot segmented mask and ground truth for top N cases.

    :param test_set: Test dataset
    :param model: Trained model
    :param top_n: Number of top cases to plot
    :return: Figure object and percentages list
    """
    fig = Figure(figsize=(12, 6))
    ax = fig.subplots()
    percentages_list = []

    for i in range(top_n):
        image = test_set[i]
        pred_mask = predict_image_mask_miou(model, image)
        pred_percentage = calculate_percentages(pred_mask.detach().cpu().numpy())
        percentages_list.append(pred_percentage)

        # Overlay the masks onto the image
        result = overlay_mask_on_image(np.array(image), pred_mask)
        result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

        ax.imshow(result_rgb)
        ax.set_title('Predicted Mask')
        ax.axis('off')

    # Return the figure object and percentages
    return fig, percentages_list[0] if percentages_list else []


# 更新process_and_predict函数
def process_and_predict(dcm_path):
    t_test = A.Resize(704, 1056, interpolation=cv2.INTER_NEAREST)
    single_test = DicomTest(dcm_path, transform=t_test)
    image = single_test.get_data()
    
    # calculate the image total area in cm
    dicom_area = calculate_dicom_image_area(dcm_path)
    # 这里仅需要返回百分比和图像，不需要保存图像
    fig, pred_percentage = plot_segmented_mask_and_ground_truth([image], model, top_n=1)
    
    # calculate the area in cm for each class
    pred_area = []
    for cla in pred_percentage:
        pred_area.append(cla * dicom_area)
    
    
    # the segmented_array s 512 * 512 pred mask
    #hu_value = calculate_hu_values_by_segment(dcm_path, segmented_array)

    return fig, pred_percentage, pred_area #, hu_value


# pred_mask = predict_image_mask_miou(model, image)

# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
# ax1.imshow(image)
# ax1.set_title('Original.png')


# ax2.imshow(pred_mask)
# ax2.set_title('Masked Picture')
# ax2.set_axis_off()

# pred = plot_segmented_mask_and_ground_truth([image], model, top_n=1)
