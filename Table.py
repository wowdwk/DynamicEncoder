import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from skimage.metrics import peak_signal_noise_ratio as psnr
from scipy.stats import entropy
from InPainting import Loader_maker_for_InPainting, InPaint
from params import *
from Selection import patch_importance

def calculate_edge_density(image):
    """Calculate edge density of an image."""
    image = image.squeeze().cpu().numpy()  # Convert to numpy array
    if image.ndim == 3:  # Check if the image has multiple channels (e.g., RGB)
        gradients = np.gradient(image, axis=(1, 2))  # Compute gradient along height and width
        edge_density = np.sum(np.sqrt(sum(g**2 for g in gradients)))
    else:  # Grayscale image
        dy, dx = np.gradient(image)
        edge_density = np.sum(np.sqrt(dx**2 + dy**2))
    return edge_density

def calculate_information_entropy(image):
    """Calculate information entropy of an image."""
    image = image.squeeze().cpu().numpy()  # Convert to numpy array
    histogram, _ = np.histogram(image, bins=256, range=(0, 1), density=True)
    return entropy(histogram)

def process_images(dim, snr) :
    # Load the trained model
    model_path  = f"inpaint_model/InPaint(DIM={dim}_SNR={snr}_PSNR=*.pt)"
    model_files = [f for f in os.listdir('inpaint_model') if f.startswith(f'InPaint(DIM={dim}_SNR={snr}')]

    if not model_files :
        raise FileNotFoundError(f"No model found for DIM={dim} and SNR={snr}")

    model_path = os.path.join('inpaint_model', model_files[-1])  # Use the latest model
    model = torch.load(model_path)
    model.eval()

    # Load the dataset
    masked_train_path = f"Masked_Train/{dim}/{snr}"
    dataset = Loader_maker_for_InPainting(root_dir=masked_train_path)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    # Prepare a dataframe to store results
    df = pd.DataFrame(columns=["Edge Density", "Information Entropy", "PSNR(Lv.1)", "PSNR(Lv.2)", "PSNR(Lv.3)", "PSNR(Lv.4)"])

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # Set the device
    model = model.to(device)  # Move the model to the device

    with torch.no_grad():
        for i, data in enumerate(dataloader):
            original_images = data['original_images'].squeeze()
            recon_0         = data['recon_0'].squeeze(1).to(device)  # Move to device
            recon_masked_23 = data['recon_masked_23'].squeeze(1).to(device)  # Move to device
            recon_masked_43 = data['recon_masked_43'].squeeze(1).to(device)  # Move to device
            recon_masked_60 = data['recon_masked_60'].squeeze(1).to(device)  # Move to device

            # Calculate edge density and information entropy
            edge_density = round(calculate_edge_density(original_images), 3)

            information_entropy = calculate_information_entropy(original_images)

            # Generate outputs and calculate PSNR for each
            outputs_0 = model(recon_0)
            outputs_23 = model(recon_masked_23)
            outputs_43 = model(recon_masked_43)
            outputs_60 = model(recon_masked_60)

            psnr_0 = round(psnr(original_images.squeeze().cpu().numpy(), outputs_0.squeeze().cpu().numpy()), 3)
            psnr_23 = round(psnr(original_images.squeeze().cpu().numpy(), outputs_23.squeeze().cpu().numpy()), 3)
            psnr_43 = round(psnr(original_images.squeeze().cpu().numpy(), outputs_43.squeeze().cpu().numpy()), 3)
            psnr_60 = round(psnr(original_images.squeeze().cpu().numpy(), outputs_60.squeeze().cpu().numpy()), 3)

            # Add the results to the dataframe
            df.loc[i] = [edge_density, information_entropy, psnr_0, psnr_23, psnr_43, psnr_60]

            if (i+1) % 1000 == 0:
                print(f"Processed {i+1}/{len(dataloader)} images")

    # Save the dataframe to an Excel file
    output_file = f"Table_DIM_{dim}_SNR_{snr}.xlsx"
    df.to_excel(output_file, index=False)
    print(f"Results saved to {output_file}")

if __name__ == "__main__":

    for dim_i in range(len(params['DIM'])) :
        for snr_i in range(len(params['SNR'])):

            process_images(params['DIM'][dim_i], params['SNR'][snr_i])
