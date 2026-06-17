import io
import os
import time
import threading
import shutil
import cv2
import numpy as np
import torch
import torch.nn as nn  # 新增：用于兼容性补丁
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
from torchvision import transforms
from contextlib import asynccontextmanager  # 新增：用于现代 Lifespan 写法

from face_alignment.alignment import norm_crop
from face_detection.yolov5_face.detector import Yolov5Face
from face_recognition.arcface.model import iresnet_inference
from face_recognition.arcface.utils import compare_encodings, read_features

# ================= 🚀 PyTorch 2.x 兼容性补丁 =================
# 解决旧版 YOLOv5 在新版 PyTorch 中 Upsample 属性缺失的问题
nn.Upsample.recompute_scale_factor = None
# ==========================================================

cv2.setNumThreads(1)
torch.set_num_threads(1)
os.environ["OMP_NUM_THREADS"] = "1"

# ----------------- 全局变量与模型初始化 -----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
detector = None
recognizer = None
images_names = []
images_embs = []
feature_lock = threading.Lock()
FEATURE_PATH = "./datasets/face_features/feature"
DATASETS_DIR = "./datasets/data"

# 全局动态识别阈值 (默认 0.80)
global_threshold = 0.80

# 确保目录存在
os.makedirs(DATASETS_DIR, exist_ok=True)


# 使用现代的 lifespan 方式管理模型加载（替代已弃用的 on_event）
@asynccontextmanager
async def lifespan(app: FastAPI):
    global detector, recognizer, images_names, images_embs
    print(f"🔥 检测到运行设备: {device}")

    print("⏳ 正在加载 YOLOv5 人脸检测模型...")
    detector = Yolov5Face(model_file="face_detection/yolov5_face/weights/yolov5n-face.pt")

    print("⏳ 正在加载 ArcFace 特征识别模型...")
    recognizer = iresnet_inference(model_name="r100", path="face_recognition/arcface/weights/arcface_r100.pth",
                                   device=device)

    print("⏳ 正在加载人脸特征库...")
    os.makedirs(os.path.dirname(FEATURE_PATH), exist_ok=True)
    names, embs = read_features(feature_path=FEATURE_PATH)
    if names is not None and embs is not None:
        images_names = names
        images_embs = embs
        print(f"✅ 成功加载 {len(images_names)} 个人脸特征！")
    else:
        print("⚠️ 特征库为空，请先调用录入接口添加人脸。")
    yield
    # 清理逻辑可以在这里添加


app = FastAPI(title="FengAI Face Recognition Server", lifespan=lifespan)

# 挂载本地目录为静态文件服务
app.mount("/images", StaticFiles(directory=DATASETS_DIR), name="images")


# ----------------- 核心工具函数 -----------------
@torch.no_grad()
def get_feature(face_image):
    face_preprocess = transforms.Compose([
        transforms.ToTensor(), transforms.Resize((112, 112)),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
    face_tensor = face_preprocess(face_image).unsqueeze(0).to(device)
    emb = recognizer(face_tensor).cpu().numpy()
    return emb / np.linalg.norm(emb)


def read_imagefile(data) -> np.ndarray:
    npimg = np.frombuffer(data, np.uint8)
    frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    return frame


# ================= 业务接口区 =================

@app.post("/recognize")
async def recognize_face(file: UploadFile = File(...)):
    global global_threshold
    image_data = await file.read()
    frame = read_imagefile(image_data)

    if frame is None:
        return JSONResponse(status_code=400, content={"error": "图片解码失败"})

    bboxes, landmarks = detector.detect(image=frame)
    if len(bboxes) == 0:
        return {"faces": []}

    results = []
    with feature_lock:
        current_names = images_names
        current_embs = images_embs

    for i in range(len(bboxes)):
        x1, y1, x2, y2, det_score = bboxes[i]
        face_name = "UNKNOWN"
        face_score = 0.0

        if len(current_embs) > 0 and landmarks is not None:
            try:
                face_alignment = norm_crop(img=frame, landmark=landmarks[i])
                query_emb = get_feature(face_alignment)
                scores, id_min = compare_encodings(query_emb, current_embs)
                face_score = float(scores[0])

                if face_score >= global_threshold:
                    face_name = str(current_names[id_min])
            except Exception as e:
                print(f"提取特征失败: {e}")

        results.append({
            "box": [int(x1), int(y1), int(x2), int(y2)],
            "name": face_name,
            "score": round(face_score, 4)
        })
    return {"faces": results}


@app.post("/add_person")
async def add_person(name: str = Form(...), file: UploadFile = File(...)):
    image_data = await file.read()
    frame = read_imagefile(image_data)

    if frame is None:
        return JSONResponse(status_code=400, content={"error": "图片解码失败"})

    bboxes, landmarks = detector.detect(image=frame)
    if len(bboxes) == 0:
        return JSONResponse(status_code=400, content={"error": "未能在图片中检测到人脸，请换一张清晰的照片"})

    face_alignment = norm_crop(img=frame, landmark=landmarks[0])
    new_emb = get_feature(face_alignment)

    global images_names, images_embs
    with feature_lock:
        is_overwrite = False
        if len(images_embs) > 0 and name in images_names:
            is_overwrite = True
            mask = (images_names != name)
            images_names = images_names[mask]
            images_embs = images_embs[mask]

        if len(images_embs) == 0:
            images_names = np.array([name])
            images_embs = np.array([new_emb])
        else:
            images_names = np.hstack((images_names, [name]))
            images_embs = np.vstack((images_embs, new_emb))

        np.savez_compressed(f"{FEATURE_PATH}.npz", images_name=images_names, images_emb=images_embs)

    person_dir = os.path.join(DATASETS_DIR, name)
    if is_overwrite and os.path.exists(person_dir):
        shutil.rmtree(person_dir)

    os.makedirs(person_dir, exist_ok=True)
    save_path = os.path.join(person_dir, f"{int(time.time())}.jpg")
    cv2.imwrite(save_path, frame)

    action_str = "覆盖更新" if is_overwrite else "新增录入"
    return {"message": f"成功将 {name} {action_str} 到特征库！"}


@app.get("/get_faces")
async def get_faces():
    result = []
    with feature_lock:
        if len(images_names) == 0:
            return {"faces": []}
        unique_names = list(set(images_names))

    for name in unique_names:
        person_dir = os.path.join(DATASETS_DIR, str(name))
        img_url = None
        if os.path.exists(person_dir):
            imgs = [f for f in os.listdir(person_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            if imgs:
                img_url = f"/images/{name}/{imgs[0]}"
        result.append({"name": name, "image_url": img_url})
    return {"faces": result}


@app.post("/delete_person")
async def delete_person(name: str = Form(...)):
    global images_names, images_embs
    with feature_lock:
        if name not in images_names:
            return JSONResponse(status_code=404, content={"error": f"人脸库中不存在 {name}"})
        mask = (images_names != name)
        images_names = images_names[mask]
        images_embs = images_embs[mask]
        np.savez_compressed(f"{FEATURE_PATH}.npz", images_name=images_names, images_emb=images_embs)

    person_dir = os.path.join(DATASETS_DIR, name)
    if os.path.exists(person_dir):
        shutil.rmtree(person_dir)
    return {"message": f"成功删除 {name} 的所有数据"}


@app.post("/set_threshold")
async def set_threshold(threshold: float = Form(...)):
    global global_threshold
    if 0.0 <= threshold <= 1.0:
        global_threshold = threshold
        return {"message": f"全局识别阈值已成功修改为: {global_threshold}"}
    return JSONResponse(status_code=400, content={"error": "阈值必须在 0.0 到 1.0 之间"})


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)