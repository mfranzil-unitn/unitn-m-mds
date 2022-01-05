import os
import sys

from scipy.io import loadmat, savemat

folder = sys.argv[1]

print(f"This script's pwd is: {__file__}")
print(f"This script must be run from the root of the project"
      f"(so that noises are available under {folder}/[camera]/noise.mat).")
print(f"This script will create a new folder in the current directory called 'average_mat'.")
print(f"If all of this is fine, press ENTER to continue.")

_ = input()

print("Creating folder 'average_mat'...")

if not os.path.exists('average_mat'):
    os.mkdir('average_mat')

# Enumerate models
models = []

for camera in os.listdir(folder):
    if os.path.isdir(f'{folder}/{camera}'):
        models.append(camera)

for camera in models:
    print(f"Model: {camera}")
    # Prepare the output average noise matrix
    average_mat = []

    # Calculate noise
    for i, filename in enumerate(os.listdir(f'{folder}/{camera}')):
        if filename.endswith('.mat'):
            print(f"\t{i + 1}/{len(os.listdir(f'{folder}/{camera}'))}==>{filename}")
            # Load the noise
            noise = loadmat(f'{folder}/{camera}/{filename}')['noiseprint']
            # Add the noise to the average
            if len(average_mat) == 0:
                average_mat = noise
            else:
                average_mat += noise
        else:
            raise FileNotFoundError(f"File {filename} is not a .mat file.")

    # Divide by the number of photographs
    average_mat /= len(os.listdir(f'{folder}/{camera}'))

    # Save the average noise matrix
    print(f"Saving average noise matrix to 'average_mat/{camera}.mat'...")
    savemat(f'average_mat/{camera}.mat', {'noiseprint': average_mat})

    print(f"Done.")
