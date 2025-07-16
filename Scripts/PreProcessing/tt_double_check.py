import numpy as np

# 1단계에서 생성된 '정답지' 파일을 불러옵니다.
file_path = 'results/preprocessed_data_20250716_094447.npy'
data = np.load(file_path)

print("--- 📝 정답지 파일 (test_output.npy) 분석 결과 ---")
print(f"데이터 형태: {data.shape}")

# 테스트 스크립트는 윈도우 크기를 12로 설정하여 테스트합니다.
# 따라서 데이터 형태의 중간 숫자는 12가 됩니다.

# 첫 번째 윈도우의 평균과 표준편차를 확인합니다.
first_window = data[0]
mean = np.mean(first_window)
std = np.std(first_window)

print(f"첫 번째 윈도우의 평균: {mean:.6f}")
print(f"첫 번째 윈도우의 표준편차: {std:.6f}")
print("-------------------------------------------------")