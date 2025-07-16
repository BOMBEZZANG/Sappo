import numpy as np
import os

def verify_npy_normalization(file_path):
    """
    지정된 .npy 파일이 0과 1 사이의 값으로 정규화되었는지 확인합니다.
    """
    if not os.path.exists(file_path):
        print(f"❌ 오류: 파일을 찾을 수 없습니다. 경로를 확인해주세요: {file_path}")
        return

    try:
        # 데이터 불러오기
        data = np.load(file_path)
        print(f"✅ 파일 로드 성공: {file_path}")
        print(f"데이터 형태 (shape): {data.shape}")

        # 전체 데이터에서 최솟값과 최댓값 찾기
        min_val = np.min(data)
        max_val = np.max(data)

        print("\n--- 🔬 정규화 검증 결과 ---")
        print(f"최솟값: {min_val:.6f}")
        print(f"최댓값: {max_val:.6f}")
        
        # 검증
        if 0.0 <= min_val <= 1.0 and 0.0 <= max_val <= 1.0:
            # 부동소수점 오차를 고려하여 0과 1에 매우 가까운지도 확인
            if np.isclose(min_val, 0.0) and np.isclose(max_val, 1.0):
                 print("\n👍 훌륭합니다! 데이터가 0과 1 사이로 완벽하게 정규화되었습니다.")
            else:
                 print("\n👍 좋습니다. 모든 데이터가 0과 1 사이의 범위 안에 있습니다.")
        else:
            print("\n⚠️ 경고: 데이터가 0과 1의 범위를 벗어났습니다. 전처리 과정을 다시 확인해주세요.")
        print("--------------------------")

    except Exception as e:
        print(f"❌ 오류 발생: 데이터를 처리하는 중 문제가 발생했습니다. {e}")

if __name__ == "__main__":
    # --- ⚠️ 여기에 확인하고 싶은 .npy 파일의 경로를 입력하세요 ---
    # 예시: "results/preprocessed_data_20250716_183000.npy"
    target_file = input("검증할 .npy 파일의 경로를 입력하세요: ")
    
    verify_npy_normalization(target_file)