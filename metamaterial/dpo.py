import cv2
import numpy as np
from PIL import Image
import random
import os
import csv
import eval_abaqus
import shutil
import matplotlib.pyplot as plt

def mirror_image(image_path, output_path):
    original = Image.open(image_path)
    mirrored_horizontal = original.transpose(Image.FLIP_LEFT_RIGHT)
    mirrored_vertical = original.transpose(Image.FLIP_TOP_BOTTOM)

    new_width = original.width * 2
    new_height = original.height * 2
    new_image = Image.new('L', (new_width, new_height))
    new_image.paste(original, (0, 0))  # Top left
    new_image.paste(mirrored_horizontal, (original.width, 0))  # Top right
    new_image.paste(mirrored_vertical, (0, original.height))  # Bottom left
    new_image.paste(mirrored_horizontal.transpose(Image.FLIP_TOP_BOTTOM), (original.width, original.height))  # Bottom right
    new_image.save(output_path)

def add_circle_on_border(image_path, output_path, circle_color="white", radius=3):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Find the contours of the white shape
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        raise ValueError("No white shape found in the image.")

    # Take the main contour (assuming there is a single white shape)
    contour = max(contours, key=len)

    # Get the dimensions of the image
    height, width = image.shape

    # Filter the contour points to exclude those on the edges of the image
    filtered_contour = [pt[0] for pt in contour if 0 < pt[0][0] < width - 1 and 0 < pt[0][1] < height - 1]

    if len(filtered_contour) == 0:
        raise ValueError("No valid points on the contour that are not on the edges of the image.")

    # Choose a random point from the filtered points
    random_point = random.choice(filtered_contour)

    # Define the color of the circle
    if circle_color == "white":
        color = 255  # White for grayscale images
    else:
        color = 0  # Black for grayscale images

    # Add the circle to the image
    output_image = image.copy()
    cv2.circle(output_image, tuple(random_point), radius=radius, color=color, thickness=-1)

    # Save the resulting image
    cv2.imwrite(output_path, output_image)

# Main
main_path = "/path/to/wd/metamaterial"
full_image_path = f"{main_path}/AbaqusTmp/image_step24.png"
perturbed_path = f"{main_path}/AbaqusTmp/tmp_perturbed"
abaqus_image_path = f"{main_path}/AbaqusTmp/tmp_abaqus_image"
min_mse_image_path = f"{main_path}/AbaqusTmp/min_mse_image"
process_path = f"{main_path}/AbaqusTmp/process"
target_csv = f"{main_path}/TargetCurves/target_1.csv"
curve_output_path = f"{main_path}/curve.png"
samples_path = f"{main_path}/AbaqusTmp/runs"
rec_file = f"{samples_path}/abaqus_eval_sample_0/abaqus1.rec"
stress_values_n = 51

# Load the image as a tensor using PIL and torch
image_path = f"{full_image_path[:-4]}_1qt.png"
with Image.open(full_image_path).convert("L") as img:
    if img.size[0] > 100:
        width, height = img.size
        new_width = width // 2
        new_height = height // 2

        upper_left_quarter = img.crop((0, 0, new_width, new_height))
        upper_left_quarter.save(image_path)

# Target curve
with open(target_csv, 'r') as file:
    reader = csv.reader(file)
    stress_target = np.array(next(reader), dtype=float)
    
os.makedirs(perturbed_path, exist_ok=True)
os.makedirs(abaqus_image_path, exist_ok=True)
os.makedirs(min_mse_image_path, exist_ok=True)
os.makedirs(process_path, exist_ok=True)

# Original shape's stress curve
mirror_image(image_path, f"{main_path}\original.png")
print(f"Processing original image")
os.chdir(main_path)
eval_abaqus.main(image_path = f"{main_path}\original.png", stress_strain_csv_path = "", samples_path = samples_path)
if os.path.exists(rec_file):
    os.remove(rec_file)

if not os.path.exists(f"{samples_path}/abaqus_eval_sample_0/csv/stress_strain.csv"):
    stress_original = np.zeros(stress_values_n)
    x_column = np.linspace(0, 0.2, stress_values_n) 
    with open(f"{samples_path}/abaqus_eval_sample_0/csv/stress_strain.csv", mode='w', newline='') as file:
        writer = csv.writer(file)
        for row in zip(x_column, stress_original):
            writer.writerow(row)
    stress_predicted_best = stress_original
    mse_old = 1e6
    print(f"    Error in Abaqus evaluation. Skipping original image")
else:
    with open(f"{samples_path}/abaqus_eval_sample_0/csv/stress_strain.csv", 'r') as f:
        reader = csv.reader(f)
        stress_original = np.array([row[1] for row in reader], dtype=float)
    stress_predicted_best = stress_original
    mse_old = np.mean((stress_target - stress_original) ** 2)
    print(f"    MSE original: {mse_old}")

# Particle swarm loop
iters = 0
min_mse = 1000
radius = 7
radius_reduce_step = 1
num_perturbations = 2
max_iters = 5
mse_threshold = 5

while min_mse > mse_threshold and iters < max_iters:
    print(f"Iter n° {iters}")

    # Add a white circle on the border
    shutil.rmtree(perturbed_path)
    shutil.rmtree(abaqus_image_path)
    os.makedirs(perturbed_path)
    os.makedirs(abaqus_image_path)
    for seed in range(num_perturbations):
        add_circle_on_border(image_path, f"{perturbed_path}\{seed}_w.png", circle_color="white", radius=radius)
        add_circle_on_border(image_path, f"{perturbed_path}\{seed}_b.png", circle_color="black", radius=radius)

    mse_dict = {}
    with open(f"{samples_path}/abaqus_eval_sample_0/csv/stress_strain.csv", 'r') as f:
        reader = csv.reader(f)
        stress_predicted_old = np.array([row[1] for row in reader], dtype=float)

    for file in os.listdir(perturbed_path):
        if file.endswith(".png"):
            mirror_image(f"{perturbed_path}\{file}", f"{abaqus_image_path}\{file}")
            #print(f"Image {file} mirrored")

            # Abaqus evaluation
            print(f"Processing sample {file}")
            os.chdir(main_path)
            eval_abaqus.main(f"{abaqus_image_path}\{file}", stress_strain_csv_path = "", samples_path=samples_path)
            if os.path.exists(rec_file):
                os.remove(rec_file)
            # Extract results
            with open(f"{samples_path}/abaqus_eval_sample_0/csv/stress_strain.csv", 'r') as f:
                reader = csv.reader(f)
                stress_predicted = np.array([row[1] for row in reader], dtype=float)

            #print(stress_predicted)

            # Compute MSE
            if not np.array_equal(stress_predicted, stress_predicted_old) and stress_predicted[-1] != 0:
                mse_dict[file] = (np.mean((stress_target - stress_predicted) ** 2), stress_predicted)
                stress_predicted_old = stress_predicted
                print(f"    MSE {file}: {mse_dict[file][0]}")
            else:
                print(f"    Error in Abaqus evaluation. Skipping {file}")

            
            if os.path.exists(rec_file):
                os.remove(rec_file)

    # Get image minimum mse
    if mse_dict == {}:
        print(f"None of the perturbed images has a valid stress curve")
        min_mse = mse_old
        radius += 2
        print(f"Radius increased to {radius}")
    else:
        image_min_mse, best_mse = min(mse_dict.items(), key=lambda item: item[1][0])
        min_mse, stress_predicted_tmp = best_mse

    if min_mse < mse_old:
        print(f"Image with minimum MSE: {image_min_mse}")
        shutil.copy(f"{perturbed_path}\{image_min_mse}", f"{min_mse_image_path}\{iters}_{image_min_mse}")
        mirror_image(f"{min_mse_image_path}\{iters}_{image_min_mse}", f"{process_path}\{iters}.png")
        image_path = f"{min_mse_image_path}\{iters}_{image_min_mse}"
        stress_predicted_best = stress_predicted_tmp
        mse_old = min_mse

    else:
        print(f"None of the preturbed images has a lower MSE")
        if iters == 0:
            shutil.copy(f"{abaqus_image_path}\{file}", f"{process_path}\{iters}.png")
        else:
            shutil.copy(f"{process_path}\{iters-1}.png", f"{process_path}\{iters}.png")

    # Save CSV
    csv_file_path = f"{process_path}\{iters}_curve.csv"
    np.savetxt(csv_file_path, stress_predicted_best, delimiter=",", comments="")

    # Save curve plot
    x = np.linspace(0, .2, len(stress_predicted_best))

    plt.figure(figsize=(10, 6))
    plt.plot(x, stress_predicted_best, label='Predicted', linestyle='--', color='g')
    plt.plot(x, stress_original, label='Original', linestyle='--', color='r')
    plt.plot(x, stress_target, label='Target', linestyle='-', color='b')
    plt.xlabel('Strain')
    plt.ylabel('Stress')
    plt.title('Predicted vs Target Stress-Strain Curve')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{process_path}\{iters}_curve.png")
    plt.close()
    
    if (iters+1) % radius_reduce_step == 0 and radius > 2:
        radius -= 1
        print(f"Radius decreased to {radius}")
        
    iters += 1

shutil.copy(f"{process_path}\{iters-1}.png", f"{full_image_path[:-4]}_aligned.png")
with Image.open(f"{full_image_path[:-4]}_aligned.png") as img:
    width, height = img.size
    new_width = width // 2
    new_height = height // 2

    upper_left_quarter = img.crop((0, 0, new_width, new_height))
    upper_left_quarter.save(f"{full_image_path[:-4]}_aligned_1qt.png")

plt.figure(figsize=(10, 10))

x = np.linspace(0, 0.2, num=len(stress_original))
plt.figure(figsize=(10, 6))
plt.plot(x, stress_predicted_best, label='Predicted', linestyle='--', color='g')
plt.plot(x, stress_original, label='Original', linestyle='--', color='r')
plt.plot(x, stress_target, label='Target', linestyle='-', color='b')
plt.xlabel('Strain')
plt.ylabel('Stress')
plt.title('Predicted vs Target Stress-Strain Curve')
plt.legend()
plt.grid(True)
plt.savefig(curve_output_path)
plt.close()