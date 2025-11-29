import os
import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt

# =========================================================
# [설정] 데이터셋 압축 푼 폴더 경로를 여기에 맞춰주세요!
# 예: './data/train' 또는 절대 경로
BASE_PATH = r'C:\Users\hojun\Desktop\Recod.ai_SIFD\data' 
# (주의: 실제 폴더 구조를 보고 경로를 수정해야 할 수 있습니다)
# =========================================================

def show_images(img_path, mask_path=None, title="Image"):
    """이미지와 마스크를 나란히, 그리고 겹쳐서 보여주는 함수"""
    
    # 1. 이미지 읽기
    img = cv2.imread(img_path)
    if img is None:
        print(f"이미지를 찾을 수 없습니다: {img_path}")
        return
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Matplotlib용 변환

    # 2. 마스크 읽기 (없으면 까만색 빈 마스크 생성)
    if mask_path and os.path.exists(mask_path):
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask_status = "Forged (Mask Found)"
    else:
        # 마스크가 없으면(Authentic) 검은색(0)으로 채움
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        mask_status = "Authentic (No Mask)"

    # 3. 겹쳐서 보기 (Overlay) - 위조 부위를 빨간색으로 표시
    overlay = img.copy()
    # 마스크가 0보다 큰 부분(위조된 부분)을 빨간색으로 칠하기
    overlay[mask > 0] = [255, 0, 0] 
    
    # 투명도 조절해서 원본이랑 섞기
    final_img = cv2.addWeighted(overlay, 0.5, img, 0.5, 0)

    # 4. 시각화 출력
    plt.figure(figsize=(15, 5))
    
    # 원본
    plt.subplot(1, 3, 1)
    plt.imshow(img)
    plt.title(f"Original: {title}")
    plt.axis('off')

    # 마스크 (정답지)
    plt.subplot(1, 3, 2)
    plt.imshow(mask, cmap='gray')
    plt.title(f"Ground Truth Mask\n({mask_status})")
    plt.axis('off')

    # 오버레이 (겹친 것)
    plt.subplot(1, 3, 3)
    plt.imshow(final_img)
    plt.title("Overlay (Red = Forged)")
    plt.axis('off')

    plt.tight_layout()
    plt.show()

# --- 실행 예시 (폴더 구조에 따라 경로 수정 필요) ---

# 1. Authentic(진본) 예시 하나 찾아서 보기
# 보통 'pristine', 'authentic' 등의 폴더명을 씁니다.
authentic_list = glob.glob(os.path.join(BASE_PATH, '**', '*authentic*', '*.png'), recursive=True)
if authentic_list:
    print(f"--- Authentic 이미지 확인: {os.path.basename(authentic_list[0])} ---")
    show_images(authentic_list[0], title="Authentic Sample")
else:
    print("Authentic 이미지를 찾지 못했습니다. 경로를 확인해주세요.")

# 2. Forged(위조) 예시 하나 찾아서 보기
# 보통 'forged' 폴더에 이미지가 있고, 같은 이름으로 'ground_truth' 폴더에 마스크가 있습니다.
forged_list = glob.glob(os.path.join(BASE_PATH, '**', '*forged*', '*.png'), recursive=True)

if forged_list:
    target_img = forged_list[0]
    # 마스크 경로 추정 (이 부분은 실제 데이터셋 폴더 구조를 봐야 정확히 알 수 있습니다)
    # 예: forged/img1.png  <->  ground_truth/img1.png (또는 _mask.png)
    # 여기서는 단순히 폴더명만 바꿔서 시도해봅니다.
    target_mask = target_img.replace("forged", "ground_truth").replace("images", "masks") 
    
    print(f"\n--- Forged 이미지 확인: {os.path.basename(target_img)} ---")
    show_images(target_img, target_mask, title="Forged Sample")
else:
    print("Forged 이미지를 찾지 못했습니다. 경로를 확인해주세요.")