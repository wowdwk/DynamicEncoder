import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

# 엑셀 파일 경로 (실제 경로로 수정)
file_path = 'C:/Users/dowon/Desktop/CISL/DE/CBS(Lv4)/Table_DIM_1024_SNR_40.xlsx'

# 엑셀 파일 불러오기
df = pd.read_excel(file_path)

# 그래프를 그리기 위한 필요한 코드들
columns_to_plot = ['Avg PSNR', 'Canny Edge', 'Sobel Edge', 'Fourier Transform']
data_to_plot = df[columns_to_plot]

# 데이터 0~1로 스케일링
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data_to_plot)

# 스케일된 데이터를 데이터프레임으로 변환
scaled_df = pd.DataFrame(scaled_data, columns=columns_to_plot)

# 그래프 그리기
plt.figure(figsize=(50, 8))

# Avg PSNR을 먼저 그리고, linewidth를 키우기


# 나머지 그래프 그리기
for column in scaled_df.columns[1:]:
    plt.plot(scaled_df.index, scaled_df[column], label=column, linewidth=1)

plt.plot(scaled_df.index, scaled_df['Avg PSNR'], label='Avg PSNR', linewidth=10)

plt.xlabel('Image Index')
plt.ylabel('Scaled Values (0-1)')
plt.title('Scaled Features Plot')
plt.legend()
plt.show()
