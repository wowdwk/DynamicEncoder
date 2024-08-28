from utils import *
import matplotlib.pyplot as plt

def plot_images(original_images, masked_0, masked_33, masked_60, masked_75, epoch):
    fig, axs = plt.subplots(1, 5, figsize=(15, 3))

    axs[0].imshow(original_images.permute(1, 2, 0).cpu().numpy())
    axs[0].set_title('Original')
    axs[0].axis('off')

    axs[1].imshow(masked_0.permute(1, 2, 0).cpu().numpy())
    axs[1].set_title('Masked 0')
    axs[1].axis('off')

    axs[2].imshow(masked_33.permute(1, 2, 0).cpu().numpy())
    axs[2].set_title('Masked 33')
    axs[2].axis('off')

    axs[3].imshow(masked_60.permute(1, 2, 0).cpu().numpy())
    axs[3].set_title('Masked 60')
    axs[3].axis('off')

    axs[4].imshow(masked_75.permute(1, 2, 0).cpu().numpy())
    axs[4].set_title('Masked 75')
    axs[4].axis('off')

    plt.suptitle(f'Epoch {epoch + 1}')
    plt.show()

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class Loader_maker_for_InPainting(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.file_paths = [os.path.join(root_dir, f) for f in os.listdir(root_dir) if f.endswith('.pt')]

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        data = torch.load(file_path)

        original_images = data['original_images']
        recon_0 = data['recon_0']
        recon_masked_33 = data['recon_masked_33']
        recon_masked_60 = data['recon_masked_60']
        recon_masked_75 = data['recon_masked_75']


        if self.transform:
            original_images = self.transform(original_images)
            recon_0         = self.transform(recon_0)
            recon_masked_33 = self.transform(recon_masked_33)
            recon_masked_60 = self.transform(recon_masked_60)
            recon_masked_75 = self.transform(recon_masked_75)

        return {
            'original_images': original_images,
            'recon_0' : recon_0,
            'recon_masked_33': recon_masked_33,
            'recon_masked_60': recon_masked_60,
            'recon_masked_75': recon_masked_75
        }

class InPaint(nn.Module):
    def __init__(self):
        super(InPaint, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=2, padding=1),
            nn.PReLU(),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.PReLU(),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1),
            nn.PReLU(),

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1),
            nn.PReLU(),

        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.PReLU(),

            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.PReLU(),

            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.PReLU(),

            nn.ConvTranspose2d(in_channels=32, out_channels=3, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


def InPaint_train(trainloader, testloader, DIM, SNR):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = InPaint().to(device)
    print("Model size : {}".format(count_parameters(model)))
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=params['LR'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.3, verbose=True)

    print("+++++ Finale(SNR = {}) Training Start! +++++\t".format(SNR))

    min_test_cost = float('inf')
    epochs_no_improve = 0
    n_epochs_stop = params['ES']

    max_psnr = 0
    previous_best_model_path = None

    for epoch in range(params['EP']):
        # ========================================== Train ==========================================
        train_loss_0 = 0.0
        train_loss_33 = 0.0
        train_loss_60 = 0.0
        train_loss_75 = 0.0

        model.train()
        timetemp = time.time()

        for data in trainloader:
            original_images = data['original_images'].squeeze(1)
            masked_0  = data['recon_0'].squeeze(1)
            masked_33 = data['recon_masked_33'].squeeze(1)
            masked_60 = data['recon_masked_60'].squeeze(1)
            masked_75 = data['recon_masked_75'].squeeze(1)

            #plot_images(original_images[0], masked_0[0], masked_33[0], masked_60[0], masked_75[0], epoch)

            original_images = original_images.to(device)
            optimizer.zero_grad()

            outputs_0  = model(masked_0.to(device))
            outputs_33 = model(masked_33.to(device))
            outputs_60 = model(masked_60.to(device))
            outputs_75 = model(masked_75.to(device))

            loss_0  = criterion(original_images, outputs_0)
            loss_33 = criterion(original_images, outputs_33)
            loss_60 = criterion(original_images, outputs_60)
            loss_75 = criterion(original_images, outputs_75)

            loss = loss_0 + loss_33 + loss_60 + loss_75

            loss.backward()
            optimizer.step()
            train_loss_0 += loss_0.item()
            train_loss_33 += loss_33.item()
            train_loss_60 += loss_60.item()
            train_loss_75 += loss_75.item()

        train_cost_0  = train_loss_0 / len(trainloader)
        train_cost_33 = train_loss_33 / len(trainloader)
        train_cost_60 = train_loss_60 / len(trainloader)
        train_cost_75 = train_loss_75 / len(trainloader)

        train_psnr_0  = round(10 * math.log10(1.0 / train_cost_0), 3)
        train_psnr_33 = round(10 * math.log10(1.0 / train_cost_33), 3)
        train_psnr_60 = round(10 * math.log10(1.0 / train_cost_60), 3)
        train_psnr_75 = round(10 * math.log10(1.0 / train_cost_75), 3)


        # ========================================== Test ==========================================
        test_loss_0 = 0.0
        test_loss_33 = 0.0
        test_loss_60 = 0.0
        test_loss_75 = 0.0
        model.eval()

        with torch.no_grad() :
            for data in testloader:
                original_images = data['original_images'].squeeze(1)
                masked_0 = data['recon_0'].squeeze(1)
                masked_33 = data['recon_masked_33'].squeeze(1)
                masked_60 = data['recon_masked_60'].squeeze(1)
                masked_75 = data['recon_masked_75'].squeeze(1)

                original_images = original_images.to(device)

                outputs_0  = model(masked_0.to(device))
                outputs_33 = model(masked_33.to(device))
                outputs_60 = model(masked_60.to(device))
                outputs_75 = model(masked_75.to(device))

                loss_0  = criterion(original_images, outputs_0)
                loss_33 = criterion(original_images, outputs_33)
                loss_60 = criterion(original_images, outputs_60)
                loss_75 = criterion(original_images, outputs_75)

                test_loss_0  += loss_0.item()
                test_loss_33 += loss_33.item()
                test_loss_60 += loss_60.item()
                test_loss_75 += loss_75.item()

            test_cost_0  = test_loss_0 / len(testloader)
            test_cost_33 = test_loss_33 / len(testloader)
            test_cost_60 = test_loss_60 / len(testloader)
            test_cost_75 = test_loss_75 / len(testloader)

            test_psnr_0 = round(10 * math.log10(1.0 / test_cost_0), 3)
            test_psnr_33 = round(10 * math.log10(1.0 / test_cost_33), 3)
            test_psnr_60 = round(10 * math.log10(1.0 / test_cost_60), 3)
            test_psnr_75 = round(10 * math.log10(1.0 / test_cost_75), 3)

            total_test_loss = test_cost_0 + test_cost_33 + test_cost_60 + test_cost_75

            if total_test_loss < min_test_cost:
                min_test_cost = total_test_loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if epochs_no_improve == n_epochs_stop:
                print("Early stopping!")
                break

            scheduler.step(total_test_loss)

            training_time = time.time() - timetemp

            Avg_train_psnr = (train_psnr_0 + train_psnr_33 + train_psnr_60 + train_psnr_75)/4
            Avg_test_psnr  = (test_psnr_0 + test_psnr_33 + test_psnr_60 + test_psnr_75)/4

            print(
                "[{:>3}-Epoch({:>5}sec.)] (Avg):{:>6.4f}/{:>6.4f}   (MR=0%):{:>6.4f}/{:>6.4f}   (MR=33%):{:>6.4f}/{:>6.4f}   (MR=60%):{:>6.4f}/{:>6.4f}   (MR=75%):{:>6.4f}/{:>6.4f}".format(
                    epoch + 1, round(training_time, 2), Avg_train_psnr, Avg_test_psnr, train_psnr_0, test_psnr_0, train_psnr_33, test_psnr_33, train_psnr_60, test_psnr_60, train_psnr_75, test_psnr_75))

            if Avg_test_psnr > max_psnr:
                save_folder = 'inpaint_model'

                if not os.path.exists(save_folder):
                    os.makedirs(save_folder)
                previous_psnr = max_psnr
                max_psnr = Avg_test_psnr

                if previous_best_model_path is not None:
                    os.remove(previous_best_model_path)
                    print(f"Performance update!! {previous_psnr} to {max_psnr}")

                save_path = os.path.join(save_folder, f"InPaint(DIM={DIM}_SNR={SNR}_PSNR={max_psnr}).pt")
                torch.save(model, save_path)
                print(f"Saved new best model at {save_path}")

                previous_best_model_path = save_path

    with open('Final_peformance.txt', 'a', encoding='utf-8') as file:

        file.write(f"\nDIM:{DIM}")
        file.write(f"\nSNR({SNR}dB) : (Avg):{max_psnr}   (Ori.):{test_psnr_0}   (MR=33%):{test_psnr_33}   (MR=65%):{test_psnr_60}   (MR=75%):{test_psnr_75}")