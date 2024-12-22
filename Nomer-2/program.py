import cv2
import numpy as np
import os
import shutil

base_folder = "Nomer-2"
input_folder = os.path.join(base_folder, "Image")
output_folder = os.path.join(base_folder, "Output")

if os.path.exists(output_folder):
    shutil.rmtree(output_folder)  
os.makedirs(output_folder, exist_ok=True)


print("Available images:")
for filename in os.listdir(input_folder):
    print(f"- {filename}")

input_file = input("Enter the name of the image file (e.g., 'image1.png'): ").strip()
input_path = os.path.join(input_folder, input_file)


if not os.path.exists(input_path):
    raise FileNotFoundError(f"Image '{input_file}' does not exist in folder '{input_folder}'.")

image = cv2.imread(input_path)
if image is None:
    raise FileNotFoundError(f"Image '{input_file}' could not be loaded. Please check the file format and integrity.")

height, width, _ = image.shape

quadrants = [
    image[0:height // 2, 0:width // 2],
    image[0:height // 2, width // 2:width],
    image[height // 2:height, 0:width // 2],
    image[height // 2:height, width // 2:width]
]


orange_hue_range = (10, 20)  
saturation_range = (150, 255)  
value_range = (150, 255)  


def calculate_average_hue(quadrant, quadrant_index):
    hsv = cv2.cvtColor(quadrant, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(
        hsv,
        (orange_hue_range[0], saturation_range[0], value_range[0]),
        (orange_hue_range[1], saturation_range[1], value_range[1])
    )
    mask_output_path = os.path.join(output_folder, f'{input_file}_quadrant_{quadrant_index}_mask.png')
    cv2.imwrite(mask_output_path, mask)

    hue_values = hsv[:, :, 0][mask > 0]
    if len(hue_values) > 0:
        return np.mean(hue_values)
    else:
        return float('inf')  


average_hues = [calculate_average_hue(quad, i + 1) for i, quad in enumerate(quadrants)]


if all(hue == float('inf') for hue in average_hues):
    closest_quadrant_index = -1  
else:
    closest_quadrant_index = np.argmin(average_hues) + 1  


for i, quad in enumerate(quadrants, start=1):
    cv2.imwrite(os.path.join(output_folder, f'{input_file}_quadrant_{i}.png'), quad)


result_file = os.path.join(output_folder, f'{input_file}_result.txt')
with open(result_file, 'w') as f:
    if closest_quadrant_index == -1:
        f.write("No orange shape detected in the image.\n")
    else:
        f.write(f"The orange shape is in position: {closest_quadrant_index}\n")

print(f"Processing complete. Results saved in '{output_folder}'.")
