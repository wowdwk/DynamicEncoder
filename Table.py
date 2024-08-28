import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from skimage.metrics import peak_signal_noise_ratio as psnr
from scipy.stats import entropy
from skimage.feature import canny
from skimage.filters import sobel, laplace
from skimage.transform import resize
from skimage.feature.texture import local_binary_pattern
from skimage.filters import gabor
from skimage import color
from skimage.measure import shannon_entropy
from scipy.fft import fft2
from skimage import morphology
from skimage.measure import label, regionprops
from InPainting import Loader_maker_for_InPainting, InPaint
from params import *
from skimage.color import rgb2gray

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
    # 이미지가 이미 numpy 배열이므로 변환할 필요 없음
    histogram, _ = np.histogram(image, bins=256, range=(0, 1), density=True)
    return entropy(histogram)


def calculate_canny_edge(image):
    """Calculate Canny edge detection."""
    return np.sum(canny(image))

def calculate_sobel_edge(image):
    """Calculate Sobel edge detection."""
    return np.sum(sobel(image))

def calculate_laplacian_edge(image):
    """Calculate Laplacian edge detection."""
    return np.sum(laplace(image))

def calculate_fractal_dimension(image):
    """Calculate fractal dimension of an image."""
    threshold = np.mean(image)
    binarized = (image < threshold)
    sizes = np.logspace(1, np.log(min(image.shape)), num=10, base=2).astype(int)  # Here is the change
    counts = [np.sum(morphology.square(i).sum(axis=0)) for i in sizes]
    coeffs = np.polyfit(np.log(sizes), np.log(counts), 1)
    return -coeffs[0]



def calculate_lbp_features(image):
    """Calculate LBP (Local Binary Pattern) features."""
    # LBP 함수를 사용하기 전에 이미지를 정수형으로 변환
    image = (image * 255).astype(np.uint8)  # 이미지를 0-255 정수형으로 변환
    lbp = local_binary_pattern(image, P=8, R=1, method='uniform')
    return np.histogram(lbp, bins=np.arange(257), density=True)[0].sum()

def calculate_gabor_features(image):
    """Calculate Gabor filter features."""
    gabor_result, _ = gabor(image, frequency=0.6)
    return np.sum(gabor_result)

def calculate_fourier_transform_features(image):
    """Calculate Fourier Transform features."""
    f_transform = fft2(image)
    return np.sum(np.abs(f_transform))

def calculate_object_count_and_density(image):
    """Calculate object count and density in an image."""
    label_img = label(image > np.mean(image))
    return len(regionprops(label_img))

def calculate_color_histogram(image):
    """Calculate color histogram features."""
    hist = np.histogram(image, bins=256, range=(0, 1))
    return hist[0].sum()





def process_images(dim, snr):
    # Load the trained model
    model_path  = f"inpaint_model/InPaint(DIM={dim}_SNR={snr}_PSNR=*.pt)"
    model_files = [f for f in os.listdir('inpaint_model') if f.startswith(f'InPaint(DIM={dim}_SNR={snr}')]

    if not model_files:
        raise FileNotFoundError(f"No model found for DIM={dim} and SNR={snr}")

    model_path = os.path.join('inpaint_model', model_files[-1])  # Use the latest model
    model = torch.load(model_path)
    model.eval()

    # Load the dataset
    masked_train_path = f"Masked_Train/{dim}/{snr}"
    dataset = Loader_maker_for_InPainting(root_dir=masked_train_path)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    # Prepare a dataframe to store results
    df = pd.DataFrame(columns=[
        "Avg PSNR", "PSNR(Lv.1)", "PSNR(Lv.2)", "PSNR(Lv.3)", "PSNR(Lv.4)",
        "Canny Edge", "Sobel Edge", "Laplacian Edge", "Fractal Dimension",
        "Entropy", "GLCM", "LBP", "Gabor", "Fourier Transform",
        "Object Count and Density", "Color Histogram"
    ])

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # Set the device
    model = model.to(device)  # Move the model to the device

    with torch.no_grad():
        for i, data in enumerate(dataloader):
            original_images = data['original_images'].squeeze()
            recon_0         = data['recon_0'].squeeze(1).to(device)  # Move to device
            recon_masked_33 = data['recon_masked_33'].squeeze(1).to(device)  # Move to device
            recon_masked_60 = data['recon_masked_60'].squeeze(1).to(device)  # Move to device
            recon_masked_75 = data['recon_masked_75'].squeeze(1).to(device)  # Move to device

            # Generate outputs and calculate PSNR for each
            outputs_0 = model(recon_0)
            outputs_33 = model(recon_masked_33)
            outputs_60 = model(recon_masked_60)
            outputs_75 = model(recon_masked_75)

            psnr_0 = psnr(original_images.squeeze().cpu().numpy(), outputs_0.squeeze().cpu().numpy())
            psnr_33 = psnr(original_images.squeeze().cpu().numpy(), outputs_33.squeeze().cpu().numpy())
            psnr_60 = psnr(original_images.squeeze().cpu().numpy(), outputs_60.squeeze().cpu().numpy())
            psnr_75 = psnr(original_images.squeeze().cpu().numpy(), outputs_75.squeeze().cpu().numpy())

            avg_psnr = (psnr_0 + psnr_33 + psnr_60 + psnr_75) / 4

            # 함수 호출 전에 numpy 배열로 변환
            original_images_np = original_images.squeeze().cpu().numpy()

            # 2차원(그레이스케일) 이미지로 변환
            if original_images_np.ndim == 3:  # RGB 이미지인 경우
                # (Channels, Height, Width) -> (Height, Width, Channels)로 변환
                original_images_np = np.transpose(original_images_np, (1, 2, 0))
                original_images_np = rgb2gray(original_images_np)

            # 엣지 밀도 및 특징 계산
            canny_edge = calculate_canny_edge(original_images_np)
            sobel_edge = calculate_sobel_edge(original_images_np)
            laplacian_edge = calculate_laplacian_edge(original_images_np)
            fractal_dimension = calculate_fractal_dimension(original_images_np)
            entropy_val = calculate_information_entropy(original_images_np)
            lbp_val = calculate_lbp_features(original_images_np)
            gabor_val = calculate_gabor_features(original_images_np)
            fourier_val = calculate_fourier_transform_features(original_images_np)
            object_count_density = calculate_object_count_and_density(original_images_np)
            color_histogram = calculate_color_histogram(original_images_np)

            # 각 열과 일치하는 값들을 추가하도록 수정된 리스트
            row_data = [
                avg_psnr, psnr_0, psnr_33, psnr_60, psnr_75,
                canny_edge, sobel_edge, laplacian_edge, fractal_dimension,
                entropy_val, None, lbp_val, gabor_val, fourier_val,
                object_count_density, color_histogram
            ]

            # 데이터프레임에 행 추가
            df.loc[i] = row_data

            if (i + 1) % 1000 == 0:
                print(f"Processed {i + 1}/{len(dataloader)} images")

    # Sort by average PSNR in ascending order
    df.sort_values("Avg PSNR", inplace=True)

    # Save the dataframe to an Excel file
    output_file = f"Table_DIM_{dim}_SNR_{snr}.xlsx"
    df.to_excel(output_file, index=False)
    print(f"Results saved to {output_file}")

if __name__ == "__main__":

    for dim_i in range(len(params['DIM'])):
        for snr_i in range(len(params['SNR'])):
            process_images(params['DIM'][dim_i], params['SNR'][snr_i])
