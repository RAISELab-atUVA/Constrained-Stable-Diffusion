import os
import os.path
import numpy as np
from abaqus.abq_utils import *
from pathlib import Path
import imageio
import csv
from PIL import Image
#import matplotlib.pyplot as plt
import sys

def process_image_to_csv(image_path, output_csv):
    """
    Processes a PNG image by resizing to 96x96, rotating it by -90 degrees,
    and extracting the top-right quadrant.

    Parameters:
        image_path (str): Path to the input PNG image.
        output_csv (str): Path to save the output CSV file.

    Returns:
        None
    """
    # Open the image and convert to grayscale
    img = Image.open(image_path).convert("L")

    # Resize to 96x96
    img = img.resize((96, 96))

    # Rotate -90 degrees (reverse columns and transpose)
    rotated_image = img.transpose(method=Image.ROTATE_270)

    # Get the top-right quadrant of the rotated image
    width, height = rotated_image.size
    quadrant = rotated_image.crop((width // 2, 0, width, height // 2))


    # Convert to binary (1 for white, 0 for black)
    binary_data = np.array(quadrant) >= 128  # Threshold at 128
    binary_data = binary_data.astype(int)  # Convert to 0 and 1

    # Flatten the binary data into a single line
    flattened_data = binary_data.flatten().tolist()

    # Save to CSV
    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(flattened_data)

def main(image_path = None, stress_strain_csv_path = "", sample_index = 0, coarseness = 0.2, samples_path="AbaqusTmp/runs"):

    process_image_to_csv(image_path, f'{samples_path}/geometries.csv')
    image_name = os.path.basename(image_path) # SZ
    image_base, image_ext = os.path.splitext(image_name)  # SZ
    image_ext = image_ext.lstrip('.')

    assert Path(samples_path, 'geometries.csv').is_file(), 'geometries.csv not found in samples_path'

    sample_grf = False # use to sample random grf instead of given geometry
    create_gifs = False # create gif of abaqus simulation
    #gif_reference_frame = 'eulerian' # 'eulerian', 'lagrangian' # store gif in Eulerian or Lagrangian reference frame SZ
    gif_reference_frame = 'lagrangian' # 'eulerian', 'lagrangian' # store gif in Eulerian or Lagrangian reference frame

    pixels = 96//2 # since we only consider one quarter

    # grf sampling properties
    if sample_grf:
        grf_alpha = 6
        pixel_threshold_rel = 0.1
        grf_threshold_rel = 0.5
        grf_geometry = generate_geometry(grf_alpha, pixels, pixel_threshold_rel, grf_threshold_rel)
        samples_path = 'grf_sample/'
        sample_index = 0
        os.makedirs(samples_path, exist_ok=True)
        np.savetxt(os.path.join(samples_path, 'geometries.csv'), grf_geometry, delimiter=',')

    # change dir to abaqus_path to store abaqus output more conveniently
    abaqus_path = os.path.join(samples_path, 'abaqus_eval_sample_{}'.format(sample_index)) 
    original_dir = os.getcwd()
    #original_dir = "metamaterial"
    os.makedirs(abaqus_path, exist_ok=True)
    os.chdir(abaqus_path)

    script_path = os.path.join(original_dir, 'abaqus', 'abaqus_script.py')
    rel_samples_path = os.path.relpath(samples_path, abaqus_path)

    if create_gifs:
        store_frames = True
    else:
        store_frames = False

    # run abaqus script
    if stress_strain_csv_path != "":
        command = (
            "abaqus cae noGUI={} -- "
            "--samples_path {} "
            "--sample_index {} "
            "--store_frames {} "
            "--pixels {} "
            "--coarseness {} "
            "--stress_strain_csv_path {} "
            "--image_base {} "
            "--image_ext {} "
            .format(script_path, rel_samples_path, sample_index, store_frames, pixels, coarseness, stress_strain_csv_path, image_base, image_ext)
        )
    else:
        command = (
            "abaqus cae noGUI={} -- "
            "--samples_path {} "
            "--sample_index {} "
            "--store_frames {} "
            "--pixels {} "
            "--coarseness {} "
            "--image_base {} "
            "--image_ext {} "
            .format(script_path, rel_samples_path, sample_index, store_frames, pixels, coarseness, image_base, image_ext)
        )

    #print(command)
    

    """ command = (
        "abaqus cae noGUI={} -- "
        "--samples_path {} "
        "--sample_index {} "
        "--store_frames {} "
        "--pixels {} "
        "--coarseness {} "
        .format(script_path, rel_samples_path, sample_index, store_frames, pixels, coarseness)
    ) 
    print(command) """ # SZ

    os.system(command)
    #print('abaqus simulation finished')

    if create_gifs:
        if Path('csv', 'geometry_frames_eul.csv').is_file(): # check if abaqus evaluation was successful
            gif_pixels = int(2*pixels)
            if gif_reference_frame == 'eulerian':
                geom_frames = np.genfromtxt(os.path.join('csv', 'geometry_frames_eul.csv'), delimiter=',').reshape(-1,gif_pixels,gif_pixels)
                s_mises_frames = np.genfromtxt(os.path.join('csv', 's_mises_frames_eul.csv'), delimiter=',').reshape(-1,gif_pixels,gif_pixels)
                s_22_frames = np.genfromtxt(os.path.join('csv', 's_22_frames_eul.csv'), delimiter=',').reshape(-1,gif_pixels,gif_pixels)
                strain_energy_frames = np.genfromtxt(os.path.join('csv', 'strain_energy_dens_frames_eul.csv'), delimiter=',').reshape(-1,gif_pixels,gif_pixels)

                # convert data to uint8 and store scaling
                max_s_mises = np.max(s_mises_frames)
                min_s_22 = np.min(s_22_frames)
                max_s_22 = np.max(s_22_frames)
                max_strain_energy = np.max(strain_energy_frames)
                frame_range = np.array([max_s_mises, min_s_22, max_s_22, max_strain_energy])
                frame_range_header = ['max_s_mises', 'min_s_22', 'max_s_22', 'max_strain_energy']

                # rescale data to [0,1]
                if not frame_range.any() == 0:
                    s_mises_frames = s_mises_frames / max_s_mises
                    s_22_frames = (s_22_frames - min_s_22) / (max_s_22 - min_s_22)
                    strain_energy_frames = strain_energy_frames / max_strain_energy

                geom_frames = (geom_frames * 255).astype(np.uint8)
                s_mises_frames = (s_mises_frames * 255).astype(np.uint8)
                s_22_frames = (s_22_frames * 255).astype(np.uint8)
                strain_energy_frames = (strain_energy_frames * 255).astype(np.uint8)

                # stack all frames at the end for consistent gif creation
                full_frames = np.stack((geom_frames, s_mises_frames, s_22_frames, strain_energy_frames), axis=-1).astype(np.uint8)

            elif gif_reference_frame == 'lagrangian':
                    
                u_1_frames = np.genfromtxt(os.path.join('csv', 'u_1_frames_lagr.csv'), delimiter=',').reshape(-1,gif_pixels,gif_pixels)
                u_2_frames = np.genfromtxt(os.path.join('csv', 'u_2_frames_lagr.csv'), delimiter=',').reshape(-1,gif_pixels,gif_pixels)
                s_mises_frames = np.genfromtxt(os.path.join('csv', 's_mises_frames_lagr.csv'), delimiter=',').reshape(-1,gif_pixels,gif_pixels)
                s_22_frames = np.genfromtxt(os.path.join('csv', 's_22_frames_lagr.csv'), delimiter=',').reshape(-1,gif_pixels,gif_pixels)
                strain_energy_frames = np.genfromtxt(os.path.join('csv', 'strain_energy_dens_frames_lagr.csv'), delimiter=',').reshape(-1,gif_pixels,gif_pixels)

                # convert data to uint8 and store scaling
                min_u_1 = np.min(u_1_frames)
                max_u_1 = np.max(u_1_frames)
                min_u_2 = np.min(u_2_frames)
                max_u_2 = np.max(u_2_frames)
                max_s_mises = np.max(s_mises_frames)
                min_s_22 = np.min(s_22_frames)
                max_s_22 = np.max(s_22_frames)
                max_strain_energy = np.max(strain_energy_frames)
                frame_range = np.array([min_u_1, max_u_1, min_u_2, max_u_2, max_s_mises, min_s_22, max_s_22, max_strain_energy])
                frame_range_header = ['min_u_1', 'max_u_1', 'min_u_2', 'max_u_2', 'max_s_mises', 'min_s_22', 'max_s_22', 'max_strain_energy']

                # rescale data to [0,1]
                if not frame_range.any() == 0:
                    u_1_frames = (u_1_frames - min_u_1) / (max_u_1 - min_u_1)
                    u_2_frames = (u_2_frames - min_u_2) / (max_u_2 - min_u_2)
                    s_mises_frames = s_mises_frames / max_s_mises
                    s_22_frames = (s_22_frames - min_s_22) / (max_s_22 - min_s_22)

                u_1_frames = (u_1_frames * 255).astype(np.uint8)
                u_2_frames = (u_2_frames * 255).astype(np.uint8)
                s_mises_frames = (s_mises_frames * 255).astype(np.uint8)
                s_22_frames = (s_22_frames * 255).astype(np.uint8)

                # stack all frames at the end for consistent gif creation
                full_frames = np.stack((u_1_frames, u_2_frames, s_mises_frames, s_22_frames), axis=-1).astype(np.uint8)

            # save as gif
            os.chdir(original_dir)
            no_frames = 11
            gif_save_dir = os.path.join(abaqus_path, 'gif/')
            os.makedirs(gif_save_dir, exist_ok=True)
            for j in range(0, full_frames.shape[-1]):
                images = []
                for k in range(no_frames):
                    images.append(full_frames[k,:,:,j])            
                imageio.mimsave(os.path.join(gif_save_dir, 'prediction_channel_' + str(j) + '.gif'), images, duration=0.2)

            np.savetxt(os.path.join(gif_save_dir, 'frame_range.csv'), np.array([frame_range]), delimiter=',', comments='', header=','.join(frame_range_header))
            print('gif creation successful')
        else:
            print('gif creation not successful')
    
if __name__ == "__main__": ### ONLY for standalone shape processing

    # Image path
    image_path = "/path/to/image"
    print(f"Image name: {os.path.basename(image_path)}")

    # Abaqus process
    main(image_path = image_path, sample_index =0, coarseness=.2, samples_path=f"{image_path}/AbaqusTmp/runs")

    # Print stress results
    with open(f"{image_path}/AbaqusTmp/runs/abaqus_eval_sample_0/csv/stress_strain.csv", 'r') as f:
        reader = csv.reader(f)
        stress_predicted = np.array([row[1] for row in reader], dtype=float)
    for stress in stress_predicted:
        print(stress)