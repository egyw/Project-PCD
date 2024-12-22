import cv2
import numpy as np
import os
import shutil

# Base folder setup
base_folder = "Nomer-2"
input_folder = os.path.join(base_folder, "Image")
output_folder = os.path.join(base_folder, "Output")

# Clear the output folder before saving new results
if os.path.exists(output_folder):
    shutil.rmtree(output_folder)  # Remove all existing files in the output folder
os.makedirs(output_folder, exist_ok=True)

# Prompt the user to enter the image name
print("Available images:")
for filename in os.listdir(input_folder):
    print(f"- {filename}")

input_file = input("Enter the name of the image file (e.g., 'image1.png'): ").strip()
input_path = os.path.join(input_folder, input_file)

# Validate if the file exists
if not os.path.exists(input_path):
    raise FileNotFoundError(f"Image '{input_file}' does not exist in folder '{input_folder}'.")

# Load the input image
image = cv2.imread(input_path)
if image is None:
    raise FileNotFoundError(f"Image '{input_file}' could not be loaded. Please check the file format and integrity.")

# Get the dimensions of the image
height, width, _ = image.shape

# Define the four quadrants
quadrants = [
    image[0:height // 2, 0:width // 2],
    image[0:height // 2, width // 2:width],
    image[height // 2:height, 0:width // 2],
    image[height // 2:height, width // 2:width]
]

# Define the target orange range in HSV
orange_hue_range = (10, 20)  # Approximate hue range for orange
saturation_range = (150, 255)  # Saturation range for bright orange
value_range = (150, 255)  # Value range for bright orange

# Function to calculate the average hue of non-background pixels in a quadrant
def calculate_average_hue(quadrant, quadrant_index):
    # Convert to HSV color space
    hsv = cv2.cvtColor(quadrant, cv2.COLOR_BGR2HSV)
    # Create a mask for orange color
    mask = cv2.inRange(
        hsv,
        (orange_hue_range[0], saturation_range[0], value_range[0]),
        (orange_hue_range[1], saturation_range[1], value_range[1])
    )
    # Save the mask for debugging
    mask_output_path = os.path.join(output_folder, f'{input_file}_quadrant_{quadrant_index}_mask.png')
    cv2.imwrite(mask_output_path, mask)
    # Get the hue values of the masked area
    hue_values = hsv[:, :, 0][mask > 0]
    if len(hue_values) > 0:
        return np.mean(hue_values)
    else:
        return float('inf')  # Return a high value if no orange pixels are found

# Analyze each quadrant and calculate the average hue
average_hues = [calculate_average_hue(quad, i + 1) for i, quad in enumerate(quadrants)]

# Handle the case where no orange pixels are detected in any quadrant
if all(hue == float('inf') for hue in average_hues):
    closest_quadrant_index = -1  # Indicate no orange shape is found
else:
    # Find the quadrant with the average hue closest to the orange hue (~15)
    closest_quadrant_index = np.argmin(average_hues) + 1  # Adding 1 to match position numbering

# Save each quadrant for reference
for i, quad in enumerate(quadrants, start=1):
    cv2.imwrite(os.path.join(output_folder, f'{input_file}_quadrant_{i}.png'), quad)

# Save the result
result_file = os.path.join(output_folder, f'{input_file}_result.txt')
with open(result_file, 'w') as f:
    if closest_quadrant_index == -1:
        f.write("No orange shape detected in the image.\n")
    else:
        f.write(f"The orange shape is in position: {closest_quadrant_index}\n")

print(f"Processing complete. Results saved in '{output_folder}'.")
