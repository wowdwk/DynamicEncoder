import os
from utils import *
from Selection import preprocess_and_save_dataset, Loader_maker_for_Transmission
from InPainting import Loader_maker_for_InPainting
from Transmission import *
from PositionEst import Make_Data
from InPainting import InPaint_train

if __name__ == '__main__':

    ####################################### Selection Process at Tx #######################################

    Processed_train_path = 'Selected_Train'
    Processed_test_path  = 'Selected_Test'

    if not os.path.exists(Processed_train_path):
        transform = transforms.Compose([transforms.ToTensor()])
        train_cifar10 = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        preprocess_and_save_dataset(train_cifar10, Processed_train_path, patch_size=1,
                                    mask_ratio=[0.234375, 0.4375, 0.609375],
                                    importance_type='variance', how_many=1)
    if not os.path.exists(Processed_test_path):
        transform = transforms.Compose([transforms.ToTensor()])
        test_cifar10 = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
        preprocess_and_save_dataset(test_cifar10, Processed_test_path, patch_size=1,
                                    mask_ratio=[0.234375, 0.4375, 0.609375],
                                    importance_type='variance', how_many=1)

    ####################################### Transmission process with Rayleigh channel #######################################

    if not os.path.exists('trained_Transmission'):
        os.makedirs('trained_Transmission')

    for dim_i in range(len(params['DIM'])):

        # 수정된 부분: DIM 값만으로 파일이 있는지 확인
        transmission_model_dir = f"trained_Transmission/DynamicEncoder(DIM={params['DIM'][dim_i]})"
        if not any(fname.startswith(f"DynamicEncoder(DIM={params['DIM'][dim_i]}_SNR=") for fname in os.listdir('trained_Transmission')):
            traindataset = Loader_maker_for_Transmission(root_dir=Processed_train_path)
            testdataset  = Loader_maker_for_Transmission(root_dir=Processed_test_path)

            trainloader = DataLoader(traindataset, batch_size=params['BS'], shuffle=True, num_workers=4, drop_last=True)
            testloader  = DataLoader(testdataset, batch_size=params['BS'], shuffle=True, num_workers=4, drop_last=True)

            Transmission_train(trainloader, testloader, params['DIM'][dim_i])

    ####################################### Position Estimator at Rx #######################################

        masked_train_dir = f"Masked_Train/{params['DIM'][dim_i]}"
        if not os.path.exists(masked_train_dir):
            Make_Data(params['DIM'][dim_i])

    ####################################### Inpainting Process at Rx #######################################

    if not os.path.exists('inpaint_model'):
        os.makedirs('inpaint_model')

    for dim_i in range(len(params['DIM'])):

        inpaint_model_dir = f"inpaint_model/InPaint(DIM={params['DIM'][dim_i]})"
        if not any(fname.startswith(f"InPaint(DIM={params['DIM'][dim_i]}_SNR=") for fname in os.listdir('inpaint_model')):
            for snr_i in range(len(params['SNR'])):

                Masked_train_path = f"Masked_Train/{params['DIM'][dim_i]}/{params['SNR'][snr_i]}"
                Masked_test_path  = f"Masked_Test/{params['DIM'][dim_i]}/{params['SNR'][snr_i]}"

                traindataset = Loader_maker_for_InPainting(root_dir=Masked_train_path)
                testdataset = Loader_maker_for_InPainting(root_dir=Masked_test_path)

                trainloader = DataLoader(traindataset, batch_size=64, shuffle=True, num_workers=4, drop_last=True)
                testloader  = DataLoader(testdataset, batch_size=64, shuffle=True, num_workers=4, drop_last=True)

                InPaint_train(trainloader, testloader, params['DIM'][dim_i], params['SNR'][snr_i])
