# face_recog_core.py （完整重构后代码，直接替换你现有代码）
import threading
import time
import pickle
import os
from datetime import datetime

import cv2
import numpy as np
import torch
import yaml
from torchvision import transforms


# ====================== 新增：强制第三方库单线程运行（核心降CPU） ======================
# 关闭OpenCV多线程
cv2.setNumThreads(1)
# 关闭PyTorch CPU多线程（限制为1核，禁用交叉线程）
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
# 关闭numpy/BLAS多线程（禁用所有数值计算多线程）
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
# ======================================================================================

from face_alignment.alignment import norm_crop
from face_detection.yolov5_face.detector import Yolov5Face
from face_recognition.arcface.model import iresnet_inference
from face_recognition.arcface.utils import compare_encodings, read_features
from face_tracking.tracker.byte_tracker import BYTETracker
from face_tracking.tracker.visualize import plot_tracking

# -------------------------- 全局初始化（仅执行一次） --------------------------
# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 人脸检测器/识别器
detector = Yolov5Face(model_file="face_detection/yolov5_face/weights/yolov5n-face.pt")
recognizer = iresnet_inference(
    model_name="r100", path="face_recognition/arcface/weights/arcface_r100.pth", device=device
)
# 加载人脸库特征
images_names, images_embs = read_features(feature_path="./datasets/face_features/feature")

# 全局共享数据+线程锁（核心，接口和后台服务共享）
shared_data = {
    "raw_image": None,
    "tracking_ids": [],
    "detection_bboxes": [],
    "detection_landmarks": [],
    "tracking_bboxes": [],
    "latest_face_info": [],  # 存储{track_id, name}
    "latest_face_scores": [], # 新增：存储对应人脸的相似度，和info一一对应
    "latest_frame": None,
    "is_running": False,
}
data_lock = threading.Lock()
# 后台服务线程（避免重复启动）
service_thread = None

# -------------------------- 原有核心函数（完全保留，无修改） --------------------------
def load_config(file_name):
    try:
        with open(file_name, "r") as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"加载配置失败: {e}")
        return {"aspect_ratio_thresh": 1.6, "min_box_area": 10}

def process_tracking(frame, detector, tracker, args, frame_id, fps):
    outputs, img_info, bboxes, landmarks = detector.detect_tracking(image=frame)
    tracking_tlwhs = []
    tracking_ids = []
    tracking_bboxes = []
    if outputs is not None:
        online_targets = tracker.update(outputs, [img_info["height"], img_info["width"]], (128, 128))
        for t in online_targets:
            tlwh = t.tlwh
            tid = t.track_id
            if tlwh[2]*tlwh[3] > args["min_box_area"] and not (tlwh[2]/tlwh[3] > args["aspect_ratio_thresh"]):
                x1, y1, w, h = tlwh
                tracking_bboxes.append([x1, y1, x1+w, y1+h])
                tracking_tlwhs.append(tlwh)
                tracking_ids.append(tid)
    with data_lock:
        shared_data["raw_image"] = img_info["raw_img"]
        shared_data["detection_bboxes"] = bboxes if bboxes is not None else []
        shared_data["detection_landmarks"] = landmarks if landmarks is not None else []
        shared_data["tracking_ids"] = tracking_ids
        shared_data["tracking_bboxes"] = tracking_bboxes
        shared_data["latest_frame"] = img_info["raw_img"]
    return plot_tracking(img_info["raw_img"], tracking_tlwhs, tracking_ids, names={}, frame_id=frame_id+1, fps=fps)
    # return img_info["raw_img"]


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

def mapping_bbox(box1, box2):
    x_min_inter = max(box1[0], box2[0])
    y_min_inter = max(box1[1], box2[1])
    x_max_inter = min(box1[2], box2[2])
    y_max_inter = min(box1[3], box2[3])
    intersection_area = max(0, x_max_inter - x_min_inter + 1) * max(0, y_max_inter - y_min_inter + 1)
    area_box1 = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    area_box2 = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)
    union_area = area_box1 + area_box2 - intersection_area
    return intersection_area / union_area if union_area != 0 else 0.0

def recognition_thread():
    while True:
        with data_lock:
            is_running = shared_data["is_running"]
            raw_image = shared_data["raw_image"]
            det_bboxes = shared_data["detection_bboxes"].copy() if shared_data["detection_bboxes"] is not None else []
            det_landmarks = shared_data["detection_landmarks"].copy() if shared_data["detection_landmarks"] is not None else []
            track_ids = shared_data["tracking_ids"].copy()
            track_bboxes = shared_data["tracking_bboxes"].copy()
        if not is_running:
            break
        # 无脸/无帧时，仅清空数据+延时，无任何打印
        if raw_image is None or len(track_bboxes) == 0 or len(det_bboxes) == 0:
            with data_lock:
                shared_data["latest_face_info"] = []
                shared_data["latest_face_scores"] = []
            time.sleep(0.1)
            continue
        current_face_info = []
        current_face_scores = []
        for i in range(len(track_bboxes)):
            face_name = None
            face_score = 0.0
            track_id = track_ids[i] if i < len(track_ids) else None
            track_box = track_bboxes[i]
            match_idx = -1
            for j in range(len(det_bboxes)):
                if mapping_bbox(track_box, det_bboxes[j]) > 0.9:
                    match_idx = j
                    break
            if match_idx != -1 and det_landmarks is not None and match_idx < len(det_landmarks):
                try:
                    face_alignment = norm_crop(img=raw_image, landmark=det_landmarks[match_idx])
                    query_emb = get_feature(face_alignment)
                    scores, id_min = compare_encodings(query_emb, images_embs)
                    face_score = scores[0]
                    name = images_names[id_min]
                    if name is not None:
                        face_name = "UNKNOWN" if face_score < 0.25 else name
                    det_bboxes = np.delete(det_bboxes, match_idx, axis=0)
                    det_landmarks = np.delete(det_landmarks, match_idx, axis=0)
                except Exception as e:
                    print(f"识别单个人脸失败: {e}")  # 保留错误打印，方便排查问题
                    face_name = None
                    face_score = 0.0
            current_face_info.append({"track_id": track_id, "name": face_name})
            current_face_scores.append(face_score)
        with data_lock:
            shared_data["latest_face_info"] = current_face_info
            shared_data["latest_face_scores"] = current_face_scores
        time.sleep(0.1)

# -------------------------- 新增：后台服务核心循环（抽离自原main） --------------------------
def _face_recog_main_loop():
    """后台核心循环：摄像头采集+跟踪+绘制，仅由start_service调用"""
    config = load_config("./face_tracking/config/config_tracking.yaml")
    tracker = BYTETracker(args=config, frame_rate=30)
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    frame_id = 0
    fps = 0
    start_time = time.time()
    threading.Thread(target=recognition_thread, daemon=True).start()
    while True:
        with data_lock:
            is_running = shared_data["is_running"]
        if not is_running:
            break
        ret, frame = cap.read()
        if not ret:
            print("摄像头读取失败，退出后台服务")
            break
        tracking_image = process_tracking(frame, detector, tracker, config, frame_id, fps)
        frame_id += 1
        if frame_id % 30 == 0:
            fps = 30 / (time.time() - start_time)
            start_time = time.time()
        # 绘制识别结果【关键修改：读取相似度并绘制】
        with data_lock:
            face_info = shared_data["latest_face_info"].copy()
            face_scores = shared_data["latest_face_scores"].copy()  # 读取相似度
        for idx, info in enumerate(face_info):
            track_id = info["track_id"]
            name = info["name"] if info["name"] else "UNKNOWN"
            if name == "UNKNOWN":
                continue
                # 非UNKNOWN才执行后续绘制逻辑
            score = face_scores[idx] if idx < len(face_scores) else 0.0
            score_str = f"{score:.2f}"
            for i, tid in enumerate(shared_data["tracking_ids"]):
                if tid == track_id:
                    x1, y1, x2, y2 = shared_data["tracking_bboxes"][i]
                    cv2.rectangle(tracking_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    cv2.putText(tracking_image, f"{name}({score_str}) (ID:{track_id})",
                                (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.imshow("Face Recognition Service", tracking_image)
        # ============== 新增：按键捕获+退出触发（核心修复） ==============
        ch = cv2.waitKey(1) & 0xFF  # 接收按键信号（加0xFF兼容跨平台）
        if ch == 27 or ch == ord('q') or ch == ord('Q'):  # 匹配ESC/q/Q
            stop_face_recog_service()  # 调用封装的停止函数，置is_running=False
            break
        # ==============================================================
    cap.release()
    cv2.destroyAllWindows()
    print("人脸识别后台服务已停止，资源已释放")

# -------------------------- 新增：服务启动/停止函数（对外调用） --------------------------
def start_face_recog_service():
    """启动人脸识别后台服务（仅需调用一次）"""
    global service_thread
    with data_lock:
        if shared_data["is_running"]:
            print("人脸识别后台服务已在运行，无需重复启动")
            return True
        shared_data["is_running"] = True
    # 启动后台服务线程（守护线程，主进程退出则停止）
    service_thread = threading.Thread(target=_face_recog_main_loop, daemon=True)
    service_thread.start()
    print("✅ 人脸识别后台服务启动成功！")
    return True

def stop_face_recog_service():
    """停止人脸识别后台服务"""
    with data_lock:
        if not shared_data["is_running"]:
            print("人脸识别后台服务未运行，无需停止")
            return True
        shared_data["is_running"] = False
    # 等待服务线程结束（最多5秒）
    if service_thread is not None:
        service_thread.join(timeout=5.0)
    print("✅ 人脸识别后台服务已停止！")
    return True

# -------------------------- 对外暴露的核心接口（原封不动，仅加锁） --------------------------
def get_latest_face_info_binary():
    with data_lock:
        if not shared_data["is_running"]:
            raise Exception("人脸识别后台服务未启动，请先调用start_face_recog_service()")
        face_info = shared_data["latest_face_info"].copy()
    print("\n" + "-"*60)
    print("【调用接口 - get_latest_face_info_binary】")
    if not face_info:
        print("  暂无识别到的人脸目标")
    else:
        print(f"  共识别到 {len(face_info)} 个目标")
        for idx, info in enumerate(face_info, 1):
            track_id = info["track_id"] if info["track_id"] else "无"
            name = info["name"] if info["name"] is not None else "未识别"
            # 可选：接口打印也显示相似度
            score = shared_data["latest_face_scores"][idx-1] if idx-1 < len(shared_data["latest_face_scores"]) else 0.0
            print(f"  目标{idx}：跟踪ID={track_id} | 姓名={name} | 相似度={score:.2f}")
    print("-"*60 + "\n")
    return pickle.dumps(face_info)

def save_current_frame(save_dir="./saved_frames"):
    """
    对外接口2：保存当前摄像头帧
    Args:
        save_dir: 帧保存目录
    Returns:
        str: 保存成功返回路径，失败返回None
    """
    with data_lock:
        if not shared_data["is_running"]:
            raise Exception("人脸识别后台服务未启动，请先调用start_face_recog_service()")
        frame = shared_data["latest_frame"]
    if frame is None:
        print("❌ 保存失败：无可用摄像头帧")
        return None
    os.makedirs(save_dir, exist_ok=True)
    # 固定文件名（frame_latest.jpg），不再用时间戳，每次保存自动覆盖
    save_file_name = "frame_latest.jpg"
    save_path = os.path.join(save_dir, save_file_name)
    cv2.imwrite(save_path, frame)
    print(f"✅ 最新帧保存成功（覆盖旧文件）：{save_path}")
    return save_path

# 测试用（单独运行该文件时启动服务）
if __name__ == "__main__":
    try:
        start_face_recog_service()
        # 让主进程持续运行，同时监听Ctrl+C
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        # 捕获Ctrl+C，优雅停止服务
        print("\n⚠️  捕获到退出信号，正在停止服务...")
        stop_face_recog_service()
    print("✅ 主进程已正常退出！")
