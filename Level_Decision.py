import os
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.optim as optim
import pickle
from sklearn.preprocessing import StandardScaler
from torchvision import transforms
from torch.utils.data import DataLoader
from InPainting import Loader_maker_for_InPainting
from tqdm import tqdm
from utils import *

def plot_images(original, recon_0, recon_33, recon_60, recon_75, idx):
    fig, axes = plt.subplots(1, 5, figsize=(15, 5))
    titles = ['Original', 'Reconstructed 0%', 'Reconstructed 33%', 'Reconstructed 60%', 'Reconstructed 75%']
    images = [original, recon_0, recon_33, recon_60, recon_75]

    for ax, img, title in zip(axes, images, titles):
        ax.imshow(img.permute(1, 2, 0).cpu().numpy())  # Channel을 마지막으로 옮겨서 imshow와 호환되도록 함
        ax.set_title(title)
        ax.axis('off')

    plt.suptitle(f"Image Set {idx}")
    plt.show()
def compute_fft(image):
    fft_image = np.fft.fft2(image, axes=(0, 1))
    fft_image_shifted = np.fft.fftshift(fft_image, axes=(0, 1))
    magnitude_spectrum = np.abs(fft_image_shifted)
    return np.log(magnitude_spectrum + 1)
def psnr(original, recon):
    mse = np.mean((original - recon) ** 2)
    if mse == 0:
        return 100
    max_pixel = 1.0
    psnr_value = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr_value
class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        original_images = sample[0]
        recon_0 = sample[1]
        recon_33 = sample[2]
        recon_60 = sample[3]
        recon_75 = sample[4]
        psnr_0 = sample[5]
        psnr_33 = sample[6]
        psnr_60 = sample[7]
        psnr_75 = sample[8]

        # Compute Fourier Transform of the image
        x_f = compute_fft(original_images.numpy())
        x_f = torch.tensor(x_f, dtype=torch.float32)

        return {
            'original_images': original_images,
            'recon_0' : recon_0,
            'recon_33': recon_33,
            'recon_60': recon_60,
            'recon_75': recon_75,
            'x_f': x_f,
            'psnr_values': torch.tensor([psnr_0, psnr_33, psnr_60, psnr_75], dtype=torch.float32)
        }
class fc_ResBlock(nn.Module):
    def __init__(self, Nin, Nout):
        super(fc_ResBlock, self).__init__()
        Nh = Nin * 2
        self.use_fc3 = False
        self.fc1 = nn.Linear(Nin, Nh)
        self.fc2 = nn.Linear(Nh, Nout)
        self.relu = nn.ReLU()
        if Nin != Nout:
            self.use_fc3 = True
            self.fc3 = nn.Linear(Nin, Nout)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        if self.use_fc3:
            x = self.fc3(x)
        out = out + x
        out = self.relu(out)
        return out
class PSNRPredictionResNet(nn.Module):
    def __init__(self, Nc_max):
        super(PSNRPredictionResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        self.conv1_f = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1)
        self.conv2_f = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.conv3_f = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv4_f = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        self.fc_resblock1 = fc_ResBlock(2049, Nc_max)
        self.fc_resblock2 = fc_ResBlock(Nc_max, Nc_max)
        self.fc_out = nn.Linear(Nc_max, 4)

        self.relu = nn.ReLU()

    def forward(self, x, x_f, snr):
        # Original Image CNN
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = x.view(x.size(0), -1)  # Flatten the output

        # Fourier Transformed Image CNN
        x_f = self.relu(self.conv1_f(x_f))
        x_f = self.relu(self.conv2_f(x_f))
        x_f = self.relu(self.conv3_f(x_f))
        x_f = self.relu(self.conv4_f(x_f))
        x_f = x_f.view(x_f.size(0), -1)  # Flatten the output

        # Concatenate flattened outputs and SNR
        snr_tensor = torch.tensor(snr, dtype=torch.float32, device=x.device).unsqueeze(0).repeat(x.size(0), 1)
        combined_features = torch.cat((x, x_f, snr_tensor), dim=1)

        # Fully Connected Residual Blocks
        x = self.fc_resblock1(combined_features)
        x = self.fc_resblock2(x)
        x = self.fc_out(x)

        return x


def train_LvDecision(dim, snr) :

    model_files = [f for f in os.listdir('inpaint_model') if f.startswith(f'InPaint(DIM={dim}_SNR={snr}')]
    if not model_files:
        raise FileNotFoundError(f"No model found for DIM={dim} and SNR={snr}")
    model_path = os.path.join('inpaint_model', model_files[-1])
    model = torch.load(model_path)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_data_file = f'prepared_traindata_{dim}_{snr}.pkl'
    test_data_file = f'prepared_testdata_{dim}_{snr}.pkl'

    if os.path.exists(train_data_file) and os.path.exists(test_data_file):
        with open(train_data_file, 'rb') as f:
            prepared_traindata = pickle.load(f)
        with open(test_data_file, 'rb') as f:
            prepared_testdata = pickle.load(f)
        print("Data loaded from files.")
    else:
        masked_train_path = f"Masked_Train/{dim}/{snr}"
        trainset = Loader_maker_for_InPainting(root_dir=masked_train_path)
        trainloader = DataLoader(trainset, batch_size=1, shuffle=True)

        masked_test_path = f"Masked_Test/{dim}/{snr}"
        testset = Loader_maker_for_InPainting(root_dir=masked_test_path)
        testloader = DataLoader(testset, batch_size=1, shuffle=False)

        prepared_traindata = []
        prepared_testdata = []

        with torch.no_grad():
            for i, data in tqdm(enumerate(trainloader), total=len(trainloader)):
                original_images = data['original_images'].squeeze()
                #print(original_images.shape)
                recon_0 = data['recon_0'].squeeze(1).to(device)
                recon_masked_33 = data['recon_masked_33'].squeeze(1).to(device)
                recon_masked_60 = data['recon_masked_60'].squeeze(1).to(device)
                recon_masked_75 = data['recon_masked_75'].squeeze(1).to(device)

                outputs_0 = model(recon_0)
                outputs_33 = model(recon_masked_33)
                outputs_60 = model(recon_masked_60)
                outputs_75 = model(recon_masked_75)

                psnr_0 = psnr(original_images.squeeze().cpu().numpy(), outputs_0.squeeze().cpu().numpy())
                psnr_33 = psnr(original_images.squeeze().cpu().numpy(), outputs_33.squeeze().cpu().numpy())
                psnr_60 = psnr(original_images.squeeze().cpu().numpy(), outputs_60.squeeze().cpu().numpy())
                psnr_75 = psnr(original_images.squeeze().cpu().numpy(), outputs_75.squeeze().cpu().numpy())



                prepared_traindata.append(
                    [original_images.cpu(), recon_0.cpu(), recon_masked_33.cpu(),recon_masked_60.cpu(), recon_masked_75.cpu(), psnr_0, psnr_33, psnr_60, psnr_75])
                '''
                if i == 0:
                    plot_images(original_images.squeeze(0), recon_0.squeeze(0), recon_masked_33.squeeze(0), recon_masked_60.squeeze(0), recon_masked_75.squeeze(0), idx=i)
                '''
            for i, data in tqdm(enumerate(testloader), total=len(testloader)):
                original_images = data['original_images'].squeeze()
                recon_0 = data['recon_0'].squeeze(1).to(device)
                recon_masked_33 = data['recon_masked_33'].squeeze(1).to(device)
                recon_masked_60 = data['recon_masked_60'].squeeze(1).to(device)
                recon_masked_75 = data['recon_masked_75'].squeeze(1).to(device)

                outputs_0 = model(recon_0)
                outputs_33 = model(recon_masked_33)
                outputs_60 = model(recon_masked_60)
                outputs_75 = model(recon_masked_75)

                psnr_0 = psnr(original_images.squeeze().cpu().numpy(), outputs_0.squeeze().cpu().numpy())
                psnr_33 = psnr(original_images.squeeze().cpu().numpy(), outputs_33.squeeze().cpu().numpy())
                psnr_60 = psnr(original_images.squeeze().cpu().numpy(), outputs_60.squeeze().cpu().numpy())
                psnr_75 = psnr(original_images.squeeze().cpu().numpy(), outputs_75.squeeze().cpu().numpy())

                prepared_testdata.append(
                    [original_images.cpu(), recon_0.cpu(), recon_masked_33.cpu(),recon_masked_60.cpu(), recon_masked_75.cpu(), psnr_0, psnr_33, psnr_60, psnr_75])



        with open(train_data_file, 'wb') as f:
            pickle.dump(prepared_traindata, f)
        with open(test_data_file, 'wb') as f:
            pickle.dump(prepared_testdata, f)
        print("Data saved to files.")

    train_loader = DataLoader(CustomDataset(prepared_traindata), batch_size=32, shuffle=True)
    test_loader = DataLoader(CustomDataset(prepared_testdata), batch_size=32, shuffle=False)

    model = PSNRPredictionResNet(Nc_max=128).to(device)
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    num_epochs = 5000
    early_stopping_counter = 0

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True) # 원래는 factor=0.5, patience=15
    min_cost = 100000
    previous_best_model_path = None

    for epoch in range(num_epochs):
        trainloss = 0.0
        model.train()
        for data in train_loader:

            images = data['original_images'].float().to(device)
            x_f = data['x_f'].float().to(device)
            targets = data['psnr_values'].to(device)

            outputs = model(images, x_f, snr)
            loss = criterion(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            trainloss += loss.item()
        traincost = trainloss / len(train_loader)

        testloss = 0.0
        model.eval()
        with torch.no_grad():
            for data in test_loader:
                images = data['original_images'].float().to(device)
                x_f = data['x_f'].float().to(device)
                targets = data['psnr_values'].to(device)

                outputs = model(images, x_f, snr)
                loss = criterion(outputs, targets)

                testloss += loss.item()
            testcost = testloss / len(test_loader)

        scheduler.step(testcost)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Train MSE: {traincost:.4f}    Test MSE: {testcost:.4f}")

        if testcost < min_cost:
            save_folder = 'Lvdecision'

            if not os.path.exists(save_folder):
                os.makedirs(save_folder)
            previous_cost = min_cost
            min_cost = testcost

            if previous_best_model_path is not None:
                os.remove(previous_best_model_path)
                print(f"Performance update!! {previous_cost:.4f} to {min_cost:.4f}")

            save_path = os.path.join(save_folder, f"Lv(DIM={dim}_SNR={snr}).pt")
            torch.save(model, save_path)
            print()

            previous_best_model_path = save_path

            with open('Transmission_peformance.txt', 'a', encoding='utf-8') as file:
                file.write(f"\nDIM:{dim}")
                file.write(f"\nSNR({snr}dB) : {testcost:.4f}")

            early_stopping_counter = 0

        else:
            early_stopping_counter += 1

        if early_stopping_counter >= 30:
            print("Early stopping triggered")
            break

    print("Training complete.")

if __name__ == '__main__':
    for dim_i in range(len(params['DIM'])) :
        for snr_i in range(len(params['SNR'])) :
            train_LvDecision(params['DIM'][dim_i], params['SNR'][snr_i])
