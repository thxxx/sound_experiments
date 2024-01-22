import pandas as pd
import numpy as np
import os

def split_and_save_dataset(csv_path, description, base_path='data'):
    # CSV 파일 읽기
    
    df = pd.read_csv(csv_path)

    # 'description' 열에서 "misc-" 문자열 제거
    df['description'] = df['description'].str.replace("misc-", "")

    # 지정된 description에 해당하는 행 필터링
    if description != 'general':
        filtered_df = df[df['description'] == description]
    else:
        filtered_df = df

    # 데이터를 무작위로 섞기
    filtered_df = filtered_df.sample(frac=1).reset_index(drop=True)

    # 데이터를 90% 훈련, 10% 평가로 분할
    split_index = int(0.9 * len(filtered_df))
    train_df = filtered_df[:split_index]
    eval_df = filtered_df[split_index:]

    # 디렉토리 생성
    target_path = os.path.join(base_path, description)
    os.makedirs(target_path, exist_ok=True)

    # CSV 파일로 저장
    train_df.to_csv(os.path.join(target_path, 'train_dataset.csv'), index=False)
    eval_df.to_csv(os.path.join(target_path, 'eval_dataset.csv'), index=False)

    return f"Datasets saved in {target_path}"


# 'description' 값을 추출하고 .txt 파일로 저장하는 함수
def save_descriptions_to_txt(csv_path, txt_file):
    df = pd.read_csv(csv_path)
    df['description'] = df['description'].str.replace("misc-", "")
    descriptions = df['description'].unique()
    with open(txt_file, 'w') as f:
        for desc in descriptions:
            f.write(f"{desc}\n")

# 모든 'description'에 대해 데이터셋을 분할하고 저장하는 함수
def process_all_descriptions(csv_path, txt_file, base_path='data'):
    os.makedirs(base_path, exist_ok=True)
    with open(txt_file, 'r') as f:
        descriptions = f.read().splitlines()
    for desc in descriptions:
        result = split_and_save_dataset(csv_path, desc, base_path)
        print(result)
    # 'general' 카테고리를 위해 전체 데이터셋 처리
    result = split_and_save_dataset(csv_path, 'general', base_path)
    print(result)

# 예시 사용법
if __name__== "__main__":
    csv_path = "./epidemic_under_1s.csv"
    txt_file = "descriptions.txt"
    
    save_descriptions_to_txt(csv_path, txt_file)
    process_all_descriptions(csv_path, txt_file)

