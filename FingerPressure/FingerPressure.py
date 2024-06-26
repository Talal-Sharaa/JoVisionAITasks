import cv2
import numpy as np
from openpyxl import Workbook
import glob

# Path to the folder containing images
image_folder = r'./Assets/task3'
image_paths = glob.glob(f"{image_folder}/*.jpg")
threshold_folder = r'./Assets/threshold'

# Initialize the workbook
wb = Workbook()
ws = wb.active

# Add headers to the Excel sheet
headers = ["Image", "Thumb_Right", "Index_Right", "Middle_Right", "Ring_Right", "Pinky_Right"]
ws.append(headers)

for image_path in image_paths:
    # Read the image
    img = cv2.imread(image_path)
    height, width, _ = img.shape

    # Extract the pressure data part (right half of the image)
    pressure_data = img[:, width//2:, :]

    # Convert to grayscale
    gray = cv2.cvtColor(pressure_data, cv2.COLOR_BGR2GRAY)

    # Determine the threshold
    _, thresh = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)

    # Define the regions for each finger
    finger_regions = {
        "Pinky_Right": gray[0:25, 0:129],
        "Ring_Right": gray[48:71, 0:129],
        "Middle_Right": gray[80:104, 0:129],
        "Index_Right": gray[120:145, 0:129],
        "Thumb_Right": gray[144:255, 200:235]
    }

    # Analyze the pressure for each finger
    finger_pressures = {}
    for finger, region in finger_regions.items():
        if np.mean(region) > 56:
            finger_pressures[finger] = 1
        else:
            finger_pressures[finger] = 0

    # Append the results to the Excel sheet
    row = [image_path.split('\\')[-1]] + [finger_pressures[f] for f in headers[1:]]
    ws.append(row)

# Save the workbook
output_path = r'./task3.xlsx'
wb.save(output_path)
