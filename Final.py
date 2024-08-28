from utils import *
from InPainting import Loader_maker_for_InPainting
from Level_Decision import PSNRPredictionResNet, CustomDataset, psnr
import pickle


def final_performance(target_PSNR, dim, snr, num_images):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # PSNR 예측 모델 로드
    model = PSNRPredictionResNet(Nc_max=128).to(device)
    model_path = f"Lvdecision/Lv(DIM={dim}_SNR={snr}).pt"
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    inpaint_model_path = None
    for file in os.listdir('inpaint_model'):
        if file.startswith(f"InPaint(DIM={dim}_SNR={snr})"):
            inpaint_model_path = os.path.join('inpaint_model', file)
            break

    if inpaint_model_path is None:
        raise FileNotFoundError(f"No inpaint model found for DIM={dim} and SNR={snr}")

    inpaint_model = torch.load(inpaint_model_path)
    inpaint_model.eval()

    # 데이터 로드

    data_file = f'prepared_traindata_{dim}_{snr}.pkl'

    if os.path.exists(data_file) :
        with open(data_file, 'rb') as f:
            prepared_data = pickle.load(f)

    dataloader = DataLoader(CustomDataset(prepared_data), batch_size=1, shuffle=True)

    results = []
    for i, data in enumerate(dataloader):
        if i >= num_images:
            break

        original_image = data['original_images'].to(device)
        fft_image      = data['x_f'].to(device)

        with torch.no_grad():
            model.eval()
            psnr_predictions = model(original_image, fft_image, snr)
            psnr_33, psnr_60, psnr_75 = psnr_predictions[1], psnr_predictions[2], psnr_predictions[3]

        if psnr_75 >= target_PSNR:
            selected_image = data['recon_masked_75']
        elif psnr_60 >= target_PSNR:
            selected_image = data['recon_masked_60']
        elif psnr_33 >= target_PSNR:
            selected_image = data['recon_masked_33']
        else:
            selected_image = data['recon_masked_0']

        inpaint_model.eval()
        with torch.no_grad():
            inpainted_image = inpaint_model(selected_image)
            psnr_value = psnr(original_image, inpainted_image)
            results.append(psnr_value)
    avg_psnr = sum(results) / len(results)
    print(f"Average PSNR over {num_images} images: {avg_psnr:.2f}")
    return avg_psnr

# 함수 호출 예시
if __name__ == "__main__":
    final_performance(target_PSNR=30, dim=1024, snr=40, num_images=100)
