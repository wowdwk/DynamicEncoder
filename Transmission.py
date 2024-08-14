from utils import *
from params import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
class Encoder(nn.Module):
    def __init__(self, latent_dim):
        super(Encoder, self).__init__()
        self.latent_dim = latent_dim

        self.in1 = nn.Conv2d(3, 32, kernel_size=5, stride=1, padding=0)
        self.in2 = nn.Conv2d(3, 32, kernel_size=5, stride=1, padding=0)
        self.in3 = nn.Conv2d(3, 32, kernel_size=5, stride=1, padding=0)
        self.in4 = nn.Conv2d(3, 32, kernel_size=5, stride=1, padding=0)

        self.out1 = nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=1)
        self.out2 = nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=1)
        self.out3 = nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=1)
        self.out4 = nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=1)

        self.prelu = nn.PReLU()

        self.pool = nn.AdaptiveAvgPool2d((8, 8))

        self.flatten = nn.Flatten()
        self.linear = nn.Linear(2048, self.latent_dim)

        self.essen = nn.Conv2d(32, 32, kernel_size=5, stride=2, padding=1)


    def forward(self, x):

        height = x.shape[-1]

        if height == 32 :
            x = self.prelu(self.in1(x))
            x = self.prelu(self.out1(x))
            x = self.prelu(self.out2(x))
            x = self.prelu(self.out3(x))
            x = self.prelu(self.out4(x))

            x = self.prelu(self.essen(x))
            #print(x.shape)

        if height == 28 : # Selection Ratio : 0.765625
            x = self.prelu(self.in2(x))
            x = self.prelu(self.out2(x))
            x = self.prelu(self.out3(x))
            x = self.prelu(self.out4(x))

            x = self.prelu(self.essen(x))
            #print(x.shape)

        if height == 24 : # Selectio Ratio : 0.5625
            x = self.prelu(self.in3(x))
            x = self.prelu(self.out3(x))
            x = self.prelu(self.out4(x))
            x = self.prelu(self.essen(x))
            #print(x.shape)

        if height == 20 : # Selection Ratio : 0.390625
            x = self.prelu(self.in4(x))
            x = self.prelu(self.out4(x))
            x = self.prelu(self.essen(x))
            #print(x.shape)

        x = self.pool(x)
        #print(x.shape)
        x = self.flatten(x)
        encoded = self.linear(x)
        encoder_level = int(((32-height)/4) + 1)

        return encoded, encoder_level
class Decoder(nn.Module):
    def __init__(self, latent_dim):
        super(Decoder, self).__init__()
        self.latent_dim = latent_dim

        self.linear = nn.Linear(self.latent_dim, 2048)
        self.prelu = nn.PReLU()
        self.unflatten = nn.Unflatten(1, (32, 8, 8))

        self.essen = nn.ConvTranspose2d(32, 32, kernel_size=5, stride=2, padding=1, output_padding=1)

        self.in4 = nn.ConvTranspose2d(32, 32, kernel_size=5, stride=1, padding=1)
        self.in3 = nn.ConvTranspose2d(32, 32, kernel_size=5, stride=1, padding=1)
        self.in2 = nn.ConvTranspose2d(32, 32, kernel_size=5, stride=1, padding=1)
        self.in1 = nn.ConvTranspose2d(32, 32, kernel_size=5, stride=1, padding=0)

        self.out4 = nn.ConvTranspose2d(32, 3, kernel_size=5, stride=1, padding=2)
        self.out3 = nn.ConvTranspose2d(32, 3, kernel_size=5, stride=1, padding=1)
        self.out2 = nn.ConvTranspose2d(32, 3, kernel_size=5, stride=1, padding=0)
        self.out1 = nn.ConvTranspose2d(32, 3, kernel_size=5, stride=1, padding=0)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x, encoder_level):

        x = self.essen(self.unflatten(self.prelu(self.linear(x))))

        if encoder_level == 1:
            x = self.prelu(self.in4(x))
            x = self.prelu(self.in3(x))
            x = self.prelu(self.in2(x))
            x = self.prelu(self.in1(x))
            x = self.out1(x)


        elif encoder_level == 2:
            x = self.prelu(self.in4(x))
            x = self.prelu(self.in3(x))
            x = self.prelu(self.in2(x))
            x = self.out2(x)


        elif encoder_level == 3:
            x = self.prelu(self.in4(x))
            x = self.prelu(self.in3(x))
            x = self.out3(x)


        elif encoder_level == 4:
            x = self.prelu(self.in4(x))
            x = self.out4(x)

        decoded = self.sigmoid(x)

        return decoded
class Autoencoder(nn.Module):
    def __init__(self, latent_dim):
        super(Autoencoder, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)

    def Power_norm(self, z, P=1 / np.sqrt(2)):

        batch_size, z_dim = z.shape
        z_power = torch.sqrt(torch.sum(z ** 2, 1))
        z_M = z_power.repeat(z_dim, 1)

        return np.sqrt(P * z_dim) * z / z_M.t()
    def Power_norm_complex(self, z, P=1 / np.sqrt(2)):

        batch_size, z_dim = z.shape
        z_com = torch.complex(z[:, 0:z_dim:2], z[:, 1:z_dim:2])
        z_com_conj = torch.complex(z[:, 0:z_dim:2], -z[:, 1:z_dim:2])
        z_power = torch.sum(z_com * z_com_conj, 1).real
        z_M = z_power.repeat(z_dim // 2, 1)
        z_nlz = np.sqrt(P * z_dim) * z_com / torch.sqrt(z_M.t())
        z_out = torch.zeros(batch_size, z_dim).to(device)
        z_out[:, 0:z_dim:2] = z_nlz.real
        z_out[:, 1:z_dim:2] = z_nlz.imag

        return z_out

    def AWGN_channel(self, x, snr, P=1):
        batch_size, length = x.shape
        gamma = 10 ** (snr / 10.0)
        noise = np.sqrt(P / gamma) * torch.randn(batch_size, length).cuda()
        y = x + noise
        return y

    def Fading_channel(self, x, snr, P=1):

        gamma = 10 ** (snr / 10.0)
        [batch_size, feature_length] = x.shape
        K = feature_length // 2

        h_I = torch.randn(batch_size, K).to(device)
        h_R = torch.randn(batch_size, K).to(device)
        h_com = torch.complex(h_I, h_R)
        x_com = torch.complex(x[:, 0:feature_length:2], x[:, 1:feature_length:2])
        y_com = h_com * x_com

        n_I = np.sqrt(P / gamma) * torch.randn(batch_size, K).to(device)
        n_R = np.sqrt(P / gamma) * torch.randn(batch_size, K).to(device)
        noise = torch.complex(n_I, n_R)

        y_add = y_com + noise
        y = y_add / h_com

        y_out = torch.zeros(batch_size, feature_length).to(device)
        y_out[:, 0:feature_length:2] = y.real
        y_out[:, 1:feature_length:2] = y.imag

        return y_out

    def forward(self, x, SNRdB, channel):

        encoded, encoder_level = self.encoder(x)

        if channel == 'AWGN':
            normalized_x = self.Power_norm(encoded)
            channel_output = self.AWGN_channel(normalized_x, SNRdB)
        elif channel == 'Rayleigh':
            normalized_complex_x = self.Power_norm_complex(encoded)
            channel_output = self.Fading_channel(normalized_complex_x, SNRdB)

        decoded = self.decoder(channel_output, encoder_level)

        return decoded
def Transmission_train(trainloader, testloader, latent_dim):

    for snr_i in range(len(params['SNR'])):

        model = Autoencoder(latent_dim=latent_dim).to(device)
        print("Model size : {}".format(count_parameters(model)))
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=params['LR'])
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20, verbose=True)

        min_test_cost = float('inf')
        epochs_no_improve = 0
        n_epochs_stop = params['ES']

        print("+++++ SNR = {} Training Start! +++++\t".format(params['SNR'][snr_i]))

        max_psnr = 0
        previous_best_model_path = None

        for epoch in range(params['EP']):

            # ========================================== Train ==========================================
            train_loss_0 = 0.0
            train_loss_23 = 0.0
            train_loss_43 = 0.0
            train_loss_60 = 0.0

            model.train()
            timetemp = time.time()

            for data in trainloader:

                original_images = data['original_images']
                reshaped_23 = data['reshaped_23']
                reshaped_43 = data['reshaped_43']
                reshaped_60 = data['reshaped_60']

                inputs_0 = original_images.to(device)
                inputs_23 = reshaped_23.to(device)
                inputs_43 = reshaped_43.to(device)
                inputs_60 = reshaped_60.to(device)

                optimizer.zero_grad()
                outputs_0 = model(inputs_0, SNRdB=params['SNR'][snr_i], channel=params['channel'])
                outputs_23 = model(inputs_23, SNRdB=params['SNR'][snr_i], channel=params['channel'])
                outputs_43 = model(inputs_43, SNRdB=params['SNR'][snr_i], channel=params['channel'])
                outputs_60 = model(inputs_60, SNRdB=params['SNR'][snr_i], channel=params['channel'])

                loss_0 = criterion(inputs_0, outputs_0)
                loss_23 = criterion(inputs_23, outputs_23)
                loss_43 = criterion(inputs_43, outputs_43)
                loss_60 = criterion(inputs_60, outputs_60)

                loss = loss_0 + 0.9 * loss_23 + 0.6 * loss_43 + 0.4 * loss_60

                loss.backward()
                optimizer.step()
                train_loss_0 += loss_0.item()
                train_loss_23 += loss_23.item()
                train_loss_43 += loss_43.item()
                train_loss_60 += loss_60.item()

            train_cost_0 = train_loss_0 / len(trainloader)
            train_cost_23 = train_loss_23 / len(trainloader)
            train_cost_43 = train_loss_43 / len(trainloader)
            train_cost_60 = train_loss_60 / len(trainloader)

            train_psnr_0 = round(10 * math.log10(1.0 / train_cost_0), 3)
            train_psnr_23 = round(10 * math.log10(1.0 / train_cost_23), 3)
            train_psnr_43 = round(10 * math.log10(1.0 / train_cost_43), 3)
            train_psnr_60 = round(10 * math.log10(1.0 / train_cost_60), 3)

            # ========================================== Test ==========================================
            test_loss_0 = 0.0
            test_loss_23 = 0.0
            test_loss_43 = 0.0
            test_loss_60 = 0.0

            model.eval()
            with torch.no_grad():
                for data in testloader:
                    original_images = data['original_images']
                    reshaped_23 = data['reshaped_23']
                    reshaped_43 = data['reshaped_43']
                    reshaped_60 = data['reshaped_60']

                    inputs_0 = original_images.to(device)
                    inputs_23 = reshaped_23.to(device)
                    inputs_43 = reshaped_43.to(device)
                    inputs_60 = reshaped_60.to(device)

                    outputs_0 = model(inputs_0, SNRdB=params['SNR'][snr_i], channel=params['channel'])
                    outputs_23 = model(inputs_23, SNRdB=params['SNR'][snr_i], channel=params['channel'])
                    outputs_43 = model(inputs_43, SNRdB=params['SNR'][snr_i], channel=params['channel'])
                    outputs_60 = model(inputs_60, SNRdB=params['SNR'][snr_i], channel=params['channel'])

                    loss_0 = criterion(inputs_0, outputs_0)
                    loss_23 = criterion(inputs_23, outputs_23)
                    loss_43 = criterion(inputs_43, outputs_43)
                    loss_60 = criterion(inputs_60, outputs_60)



                    test_loss_0 += loss_0.item()
                    test_loss_23 += loss_23.item()
                    test_loss_43 += loss_43.item()
                    test_loss_60 += loss_60.item()

                test_cost_0 = test_loss_0 / len(testloader)
                test_cost_23 = test_loss_23 / len(testloader)
                test_cost_43 = test_loss_43 / len(testloader)
                test_cost_60 = test_loss_60 / len(testloader)

                test_psnr_0 = round(10 * math.log10(1.0 / test_cost_0), 3)
                test_psnr_23 = round(10 * math.log10(1.0 / test_cost_23), 3)
                test_psnr_43 = round(10 * math.log10(1.0 / test_cost_43), 3)
                test_psnr_60 = round(10 * math.log10(1.0 / test_cost_60), 3)

                total_test_loss = test_cost_0 + test_cost_23 + test_cost_43 + test_cost_60


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

            Avg_train_psnr = (train_psnr_0 + train_psnr_23 + train_psnr_43 + train_psnr_60) / 4
            Avg_test_psnr = (test_psnr_0 + test_psnr_23 + test_psnr_43 + test_psnr_60) / 4

            print(
                "[{:>3}-Epoch({:>5}sec.)] (Avg):{:>6.4f}/{:>6.4f}   (MR=0%):{:>6.4f}/{:>6.4f}   (MR=23%):{:>6.4f}/{:>6.4f}   (MR=43%):{:>6.4f}/{:>6.4f}   (MR=60%):{:>6.4f}/{:>6.4f}".format(
                    epoch + 1, round(training_time, 2), Avg_train_psnr, Avg_test_psnr, train_psnr_0, test_psnr_0,
                    train_psnr_23, test_psnr_23, train_psnr_43, test_psnr_43, train_psnr_60, test_psnr_60))

            if Avg_test_psnr > max_psnr:

                save_folder = 'trained_Transmission'

                if not os.path.exists(save_folder):
                    os.makedirs(save_folder)
                previous_psnr = max_psnr
                max_psnr = Avg_test_psnr

                # 이전 최고 성능 모델이 있다면 삭제
                if previous_best_model_path is not None:
                    os.remove(previous_best_model_path)
                    print(f"Performance update!! {previous_psnr} to {max_psnr}")

                save_path = os.path.join(save_folder,
                                         f"DynamicEncoder(DIM={latent_dim}_SNR={params['SNR'][snr_i]}).pt")
                torch.save(model, save_path)
                print(f"Saved new best model at {save_path}")

                previous_best_model_path = save_path
