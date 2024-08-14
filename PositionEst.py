import os
import torch
from torch.utils.data import DataLoader
from utils import *
from Selection import Loader_maker_for_Transmission
from Transmission import Autoencoder
from tqdm import tqdm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def plot_image(image):
    # Permute the tensor to bring the channels to the last dimension
    image = image.permute(1, 2, 0).cpu().numpy()
    plt.imshow(image)

    plt.axis('off')
    plt.show()

def reconstruct_image_from_patches(unmasked_patches, patch_indices, image_shape, patch_size):

    B, C, H, W = image_shape

    #plot_image(unmasked_patches[0])

    split_tensors = torch.split(unmasked_patches, patch_size, dim=2)

    unmasked_patches = torch.cat(split_tensors, dim=3)

    reconstructed_images = torch.zeros((B, C, H, W)).to(unmasked_patches.device)

    for b in range(B):
        patches = []
        for i in range(len(patch_indices[b])):
            patches.append(unmasked_patches[b, :, :, patch_size*i : patch_size * (i + 1)])

        for idx, linear_idx in enumerate(patch_indices[b]):
            i = torch.div(linear_idx, (W // patch_size), rounding_mode='floor')

            j = linear_idx % (W // patch_size)

            patch = patches[idx]
            reconstructed_images[b, :, i * patch_size:(i + 1) * patch_size, j * patch_size:(j + 1) * patch_size] = patch

        #plot_image(reconstructed_images[0])

    return reconstructed_images


def process_data_with_model(data_loader, model, SNR):
    processed_data = []
    for data in tqdm(data_loader):
        original_images = data['original_images'].to(device)
        reshaped_23 = data['reshaped_23'].to(device)
        reshaped_43 = data['reshaped_43'].to(device)
        reshaped_60 = data['reshaped_60'].to(device)
        index_23 = data['index_23']
        index_43 = data['index_43']
        index_60 = data['index_60']
        image_shape = original_images.shape

        with torch.no_grad():
            recon_0  = model(original_images, SNRdB=SNR, channel=params['channel'])
            recon_23 = model(reshaped_23, SNRdB=SNR, channel=params['channel'])
            recon_43 = model(reshaped_43, SNRdB=SNR, channel=params['channel'])
            recon_60 = model(reshaped_60, SNRdB=SNR, channel=params['channel'])

        recon_masked_23 = reconstruct_image_from_patches(recon_23, index_23, image_shape, patch_size=1)
        recon_masked_43 = reconstruct_image_from_patches(recon_43, index_43, image_shape, patch_size=1)
        recon_masked_60 = reconstruct_image_from_patches(recon_60, index_60, image_shape, patch_size=1)

        processed_data.append((original_images.cpu(), recon_0.cpu(), recon_masked_23.cpu(), recon_masked_43.cpu(), recon_masked_60.cpu()))

    return processed_data

def save_processed_data(processed_data, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    for i, (original_images, recon_0, recon_masked_23, recon_masked_43, recon_masked_60) in enumerate(processed_data):
        torch.save({
            'original_images': original_images,
            'recon_0'        : recon_0,
            'recon_masked_23': recon_masked_23,
            'recon_masked_43': recon_masked_43,
            'recon_masked_60': recon_masked_60
        }, os.path.join(save_dir, f'Data_{i}.pt'))

def Make_Data(DIM):
    for snr_i in range(len(params['SNR'])):
        SNR = params['SNR'][snr_i]

        model_path = f"trained_Transmission/DynamicEncoder(DIM={DIM}_SNR={SNR}).pt"
        print(f"Loading model from {model_path}")

        model = torch.load(model_path).to(device)
        model.eval()

        Processed_train_path = "Selected_Train"
        Processed_test_path = "Selected_Test"

        traindataset = Loader_maker_for_Transmission(root_dir=Processed_train_path)
        trainloader = DataLoader(traindataset, batch_size=1, shuffle=False, num_workers=4, drop_last=False)

        testdataset = Loader_maker_for_Transmission(root_dir=Processed_test_path)
        testloader = DataLoader(testdataset, batch_size=1, shuffle=False, num_workers=4, drop_last=False)

        processed_train = process_data_with_model(trainloader, model, SNR=SNR)
        save_processed_data(processed_train, save_dir=f"Masked_Train/{DIM}/{SNR}")

        processed_test = process_data_with_model(testloader, model, SNR=SNR)
        save_processed_data(processed_test, save_dir=f"Masked_Test/{DIM}/{SNR}")




