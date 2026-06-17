import os
import cv2
import numpy as np
import torch
from torchvision import transforms

# 导入你代码中的核心函数
from face_alignment.alignment import norm_crop
from face_detection.scrfd.detector import SCRFD
from face_recognition.arcface.model import iresnet_inference

# 配置（和你主代码一致）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
detector = SCRFD(model_file="face_detection/scrfd/weights/scrfd_2.5g_bnkps.onnx")
recognizer = iresnet_inference(
    model_name="r100", path="face_recognition/arcface/weights/arcface_r100.pth", device=device
)

# 路径配置（改用绝对路径，避免工作目录问题）
RAW_IMAGE_DIR = "./datasets/data"  # 原始人脸图片库
FEATURE_SAVE_PATH = "./datasets/face_features/feature"  # 特征库基础路径
FINAL_FEATURE_FILE = f"{FEATURE_SAVE_PATH}.npz"  # 最终生成的文件


# 人脸预处理（和你主代码一致）
def preprocess_face(face_img):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((112, 112)),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
    return transform(face_img).unsqueeze(0).to(device)


# 提取单张人脸的特征（增加异常处理）
def extract_face_feature(img_path):
    try:
        img = cv2.imread(img_path)
        if img is None:
            print(f"⚠️ {img_path} 图片读取失败，跳过")
            return None
        # 检测人脸并对齐
        bboxes, landmarks = detector.detect(image=img)
        if len(landmarks) == 0:
            print(f"⚠️ {img_path} 未检测到人脸，跳过")
            return None
        # 对齐人脸（和主代码逻辑一致）
        aligned_face = norm_crop(img, landmarks[0])
        # 提取特征
        with torch.no_grad():
            face_tensor = preprocess_face(aligned_face)
            emb = recognizer(face_tensor).cpu().numpy()
            emb = emb / np.linalg.norm(emb)  # 归一化
        return emb
    except Exception as e:
        print(f"⚠️ {img_path} 特征提取失败：{e}，跳过")
        return None


# 生成整个人脸库的特征（核心修改：键名改为images_name/images_emb）
def generate_features():
    # 创建特征库保存目录
    os.makedirs(os.path.dirname(FEATURE_SAVE_PATH), exist_ok=True)

    images_name = []  # 对应read_features的images_name键
    images_emb = []  # 对应read_features的images_emb键

    # 检查原始图片库是否存在
    if not os.path.exists(RAW_IMAGE_DIR):
        print(f"❌ 原始图片库目录不存在：{RAW_IMAGE_DIR}")
        # 生成空的有效文件，避免read_features返回None
        np.savez(FINAL_FEATURE_FILE, images_name=[], images_emb=np.array([]))
        print(f"✅ 生成空特征库文件：{FINAL_FEATURE_FILE}")
        return

    # 遍历原始图片库（按文件夹=人名）
    person_folders = [f for f in os.listdir(RAW_IMAGE_DIR) if os.path.isdir(os.path.join(RAW_IMAGE_DIR, f))]
    if len(person_folders) == 0:
        print(f"⚠️ 原始图片库中无有效人名文件夹：{RAW_IMAGE_DIR}")
        # 生成空的有效文件
        np.savez(FINAL_FEATURE_FILE, images_name=[], images_emb=np.array([]))
        print(f"✅ 生成空特征库文件：{FINAL_FEATURE_FILE}")
        return

    # 遍历每个人名的图片
    for person_name in person_folders:
        person_dir = os.path.join(RAW_IMAGE_DIR, person_name)
        print(f"🔄 处理 {person_name} 的人脸图片...")

        # 遍历该人的所有图片（只处理图片文件）
        img_files = [f for f in os.listdir(person_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if len(img_files) == 0:
            print(f"⚠️ {person_name} 文件夹中无有效图片，跳过")
            continue

        for img_name in img_files:
            img_path = os.path.join(person_dir, img_name)
            # 提取特征
            emb = extract_face_feature(img_path)
            if emb is not None:
                images_name.append(person_name)
                images_emb.append(emb)

    # 核心：保存时用images_name/images_emb作为键（和read_features完全匹配）
    if len(images_emb) > 0:
        # 拼接特征矩阵（n, 512）
        images_emb = np.concatenate(images_emb, axis=0)
        # 保存：键名严格对应read_features的要求
        np.savez(FINAL_FEATURE_FILE, images_name=images_name, images_emb=images_emb)
        print(f"✅ 特征库生成完成！")
        print(f"📊 共处理 {len(images_name)} 张人脸图片，涉及 {len(set(images_name))} 人")
        print(f"📁 特征库文件路径：{FINAL_FEATURE_FILE}")
    else:
        print(f"⚠️ 未提取到任何有效人脸特征")
        # 生成空的有效文件，避免read_features返回None
        np.savez(FINAL_FEATURE_FILE, images_name=[], images_emb=np.array([]))
        print(f"✅ 生成空特征库文件：{FINAL_FEATURE_FILE}")


if __name__ == "__main__":
    # 执行生成，并捕获全局异常
    try:
        generate_features()
    except Exception as e:
        print(f"❌ 特征库生成失败：{e}")
        # 兜底：生成空的有效文件
        np.savez(FINAL_FEATURE_FILE, images_name=[], images_emb=np.array([]))
        print(f"✅ 生成空特征库文件：{FINAL_FEATURE_FILE}")
