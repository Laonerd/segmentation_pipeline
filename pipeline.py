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

    # 这里仅需要返回百分比和图像，不需要保存图像
    fig, pred_percentage = plot_segmented_mask_and_ground_truth([image], model, top_n=1)

    return fig, pred_percentage


# pred_mask = predict_image_mask_miou(model, image)

# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
# ax1.imshow(image)
# ax1.set_title('Original.png')


# ax2.imshow(pred_mask)
# ax2.set_title('Masked Picture')
# ax2.set_axis_off()

# pred = plot_segmented_mask_and_ground_truth([image], model, top_n=1)
