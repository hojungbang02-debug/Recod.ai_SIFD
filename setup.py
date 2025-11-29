import zipfile
import os

# 압축 파일 이름 (다운로드된 파일명 확인 필요)
zip_file = "recodai-luc-scientific-image-forgery-detection.zip"
extract_path = "./data"  # data 폴더에 풀기

# 폴더가 없으면 생성
os.makedirs(extract_path, exist_ok=True)

# 압축 풀기
with zipfile.ZipFile(zip_file, 'r') as zip_ref:
    zip_ref.extractall(extract_path)

print("압축 해제 완료!")