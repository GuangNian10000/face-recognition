import os
import time
import threading
import socket
import json
import shutil
import cv2
import numpy as np
import torch
import torch.nn as nn
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import Response, JSONResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
from torchvision import transforms
import urllib.parse
from pydantic import BaseModel
import traceback  # 🌟 确保崩溃栈能正确打印

from face_alignment.alignment import norm_crop
from face_detection.yolov5_face.detector import Yolov5Face
from face_recognition.arcface.model import iresnet_inference
from face_recognition.arcface.utils import compare_encodings, read_features

# ================= 🚀 PyTorch 2.x 兼容性补丁 =================
nn.Upsample.recompute_scale_factor = None
# ==========================================================

# 性能优化配置
cv2.setNumThreads(1)
torch.set_num_threads(1)
os.environ["OMP_NUM_THREADS"] = "1"


# --- 🌟 新增：人脸追踪识别辅助函数与状态 🌟 ---
active_face_tracks = []

def get_centroid(box):
    return (box[0] + box[2]) / 2, (box[1] + box[3]) / 2

def get_distance(p1, p2):
    return ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)**0.5
# --------------------------------------------


# ================= 1. 全局配置与状态变量 =================
class ServerConfig(BaseModel):
    threshold: float
    camera_flip: bool
    only_known: bool  # 🌟 新增字段


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 模型与特征库
detector = None
recognizer = None
images_names = []
images_embs = []
images_descs = []  # 🌟 新增描述库
FEATURE_PATH = "./datasets/face_features/feature"
DATASETS_DIR = "./datasets/data"
CONFIG_PATH = "./datasets/server_config.json"

# 线程锁
model_lock = threading.Lock()
feature_lock = threading.Lock()
frame_lock = threading.Lock()

# 视频流实时数据与配置
latest_frame = None
global_threshold = 0.80
server_camera_flip = True
server_only_known = False  # 🌟 新增全局变量
active_tcp_clients = []

# 🌟 摄像头硬件错误状态
camera_error_state = {
    "has_error": False,
    "message": ""
}


def load_server_config():
    global global_threshold, server_camera_flip, server_only_known  # 🌟 声明全局变量
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                cfg = json.load(f)
                global_threshold = cfg.get("threshold", 0.80)
                server_camera_flip = cfg.get("camera_flip", True)
                server_only_known = cfg.get("only_known", False)  # 🌟 加载配置
                print(
                    f"✅ [配置] 成功加载本地设置 (阈值: {global_threshold}, 反转: {server_camera_flip}, 仅熟人: {server_only_known})")
        except:
            pass
    save_server_config()


def save_server_config():
    try:
        # 🌟 保存新字段
        cfg = {"threshold": global_threshold, "camera_flip": server_camera_flip, "only_known": server_only_known}
        os.makedirs(os.path.dirname(CONFIG_PATH), exist_ok=True)
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=False)
    except:
        pass


# ================= 2. 核心功能函数 =================
def load_models_and_data():
    load_server_config()

    global detector, recognizer, images_names, images_embs, images_descs
    print(f"\n🔥 [系统启动] 运行设备: {device}")

    print("⏳ [系统启动] 正在加载 YOLOv5 人脸检测模型...")
    detector = Yolov5Face(model_file="face_detection/yolov5_face/weights/yolov5n-face.pt")

    print("⏳ [系统启动] 正在加载 ArcFace 特征识别模型...")
    recognizer = iresnet_inference(model_name="r100", path="face_recognition/arcface/weights/arcface_r100.pth",
                                   device=device)

    print("⏳ [系统启动] 正在加载人脸特征库...")
    os.makedirs(os.path.dirname(FEATURE_PATH), exist_ok=True)
    os.makedirs(DATASETS_DIR, exist_ok=True)
    names, embs, descs = read_features(feature_path=FEATURE_PATH)
    if names is not None and embs is not None:
        images_names = names
        images_embs = embs
        images_descs = descs
        print(f"✅ [系统启动] 成功加载 {len(images_names)} 个人脸特征！")
    else:
        print("⚠️ [系统启动] 特征库为空。")


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


# ================= 3. 后台视频流与识别线程 =================
def video_stream_worker():
    global latest_frame

    camera_ports = [0, 1, '/dev/video0', '/dev/video1']
    cap = None

    for port in camera_ports:
        print(f"⏳ [视频流] 正在尝试打开摄像头口: {port}...")
        cap = cv2.VideoCapture(port)

        if cap.isOpened():
            print(f"✅ [视频流] 成功连接到摄像头口: {port}")
            camera_error_state["has_error"] = False
            break
        else:
            cap.release()

    if not cap or not cap.isOpened():
        error_msg = "无法打开摄像头"
        print(f"❌ [视频流] {error_msg}")
        camera_error_state["has_error"] = True
        camera_error_state["message"] = error_msg
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    print("🎥 [视频流] 摄像头已准备就绪，开始实时画面捕捉与识别...")

    while True:
        ret, frame = cap.read()
        if not ret:
            error_msg = "读取画面失败，摄像头可能已脱落！"
            print(f"❌ [视频流] {error_msg}")
            camera_error_state["has_error"] = True
            camera_error_state["message"] = error_msg
            time.sleep(1)
            continue

        camera_error_state["has_error"] = False

        if server_camera_flip:
            frame = cv2.flip(frame, -1)
        else:
            frame = cv2.flip(frame, 1)

        with frame_lock:
            latest_frame = frame.copy()

        with model_lock:
            bboxes, landmarks = detector.detect(image=frame)

        if len(bboxes) > 0:
            current_frame_results = []

            with feature_lock:
                current_names = images_names
                current_embs = images_embs
                current_descs = images_descs

            for i in range(len(bboxes)):
                face_name = ""
                face_desc = ""
                face_score = 0.0

                if len(current_embs) > 0 and landmarks is not None:
                    try:
                        face_alignment = norm_crop(img=frame, landmark=landmarks[i])
                        with model_lock:
                            query_emb = get_feature(face_alignment)
                        scores, id_min = compare_encodings(query_emb, current_embs)
                        face_score = float(scores[0])

                        if face_score >= global_threshold:
                            face_name = str(current_names[id_min])
                            face_desc = str(current_descs[id_min])
                    except Exception:
                        pass

                current_frame_results.append({
                    "name": face_name,
                    "desc": face_desc,
                    "score": round(face_score, 4),
                    "box": [int(b) for b in bboxes[i][:4]]
                })

            # --- 🌟 核心改进：引入追踪与平滑逻辑 🌟 ---
            now = time.time()
            # 1. 尝试将当前帧的人脸匹配到现有追踪中 (使用中心点距离)
            for face in current_frame_results:
                matched_track = None
                best_dist = 80  # 像素阈值，根据分辨率 640x480 设定
                face_center = get_centroid(face['box'])

                for t in active_face_tracks:
                    t_center = get_centroid(t['last_box'])
                    dist = get_distance(face_center, t_center)
                    if dist < best_dist:
                        best_dist = dist
                        matched_track = t

                if matched_track:
                    matched_track['last_box'] = face['box']
                    matched_track['last_seen'] = now
                    matched_track['history'].append(face)
                    # 如果当前识别到了熟人，且分数比之前记录的高，则更新最佳结果
                    if face['name']:
                        if not matched_track['best_known'] or face['score'] > matched_track['best_known']['score']:
                            matched_track['best_known'] = face
                else:
                    # 未匹配到，创建新的追踪
                    active_face_tracks.append({
                        'id': int(now * 1000),
                        'history': [face],
                        'last_box': face['box'],
                        'last_seen': now,
                        'start_time': now,
                        'best_known': face if face['name'] else None,
                        'reported_known': False,
                        'reported_unknown': False
                    })

            # 2. 清理过期追踪 (超过 1 秒未出现在画面中)
            active_face_tracks[:] = [t for t in active_face_tracks if now - t['last_seen'] < 1.0]

            # 3. 决定是否推送 TCP 消息 (平滑过滤)
            faces_to_broadcast = []
            for t in active_face_tracks:
                # 情况 A: 发现了熟人
                if t['best_known'] and not t['reported_known']:
                    # 统计历史中熟人出现的次数，防止单帧抖动产生的误报
                    known_count = sum(1 for h in t['history'] if h['name'])
                    if known_count >= 2:  # 至少两帧确认为熟人
                        print(f"🎯 [识别成功] 经过平滑确认: {t['best_known']['name']} (得分: {t['best_known']['score']})")
                        faces_to_broadcast.append(t['best_known'])
                        t['reported_known'] = True
                
                # 情况 B: 判定为陌生人 (等待较长时间确认)
                elif not t['reported_known'] and not t['reported_unknown']:
                    if len(t['history']) >= 10:  # 持续 10 帧 (约 0.5s) 都是未知，才判定为陌生人
                        if not any(h['name'] for h in t['history']):
                            if not server_only_known:
                                print(f"👤 [识别结果] 判定为陌生人 (已持续 10 帧)")
                                faces_to_broadcast.append(t['history'][-1])
                            t['reported_unknown'] = True

            if faces_to_broadcast:
                event_data = {
                    "type": "face_event",
                    "count": len(faces_to_broadcast),
                    "faces": faces_to_broadcast,
                    "global_threshold": global_threshold
                }
                broadcast_tcp_message(event_data)

        time.sleep(0.05)


def status_broadcast_worker():
    while True:
        if camera_error_state["has_error"]:
            broadcast_tcp_message({
                "type": "face_error",
                "message": camera_error_state["message"]
            })
        time.sleep(3)


# ================= 4. TCP 双向通讯核心 =================
def broadcast_tcp_message(msg_dict):
    if not active_tcp_clients:
        return
    msg_bytes = (json.dumps(msg_dict, ensure_ascii=False) + "\n").encode('utf-8')
    disconnected_clients = []
    for client in active_tcp_clients:
        try:
            client.sendall(msg_bytes)
        except Exception:
            disconnected_clients.append(client)
    for client in disconnected_clients:
        if client in active_tcp_clients:
            active_tcp_clients.remove(client)


def handle_tcp_client(conn, addr):
    # 🌟 修复关键点：所有要修改的全局变量必须在函数最开头声明！
    global global_threshold, images_names, images_embs, images_descs

    print(f"🔗 [TCP] 平板客户端已连接: {addr}")
    active_tcp_clients.append(conn)

    if camera_error_state["has_error"]:
        err_msg = json.dumps({"type": "face_error", "message": camera_error_state["message"]},
                             ensure_ascii=False) + "\n"
        try:
            conn.sendall(err_msg.encode('utf-8'))
        except:
            pass

    buffer = ""

    try:
        while True:
            data = conn.recv(4096)
            if not data:
                break

            buffer += data.decode('utf-8')

            while "\n" in buffer:
                line, buffer = buffer.split("\n", 1)
                line = line.strip()
                if not line: continue

                try:
                    req = json.loads(line)

                    # ====== 🌟 核心新增：拦截心跳包并立即回复 pong ======
                    if req.get("type") == "ping":
                        try:
                            conn.sendall('{"type":"pong"}\n'.encode('utf-8'))
                        except Exception as e:
                            print(f"⚠️ [TCP] 发送心跳回执失败: {e}")
                        continue  # 直接跳过，不执行后面的 action 逻辑
                    # ====================================================

                    action = req.get("action")
                    response = {"action": action, "status": "error", "message": "未知指令"}

                    try:
                        if action == "add_person":
                            name = req.get("name")
                            desc = req.get("desc", "")
                            if not name:
                                response["message"] = "必须提供 name 字段"
                            else:
                                with frame_lock:
                                    current_img = latest_frame.copy() if latest_frame is not None else None

                                if current_img is None:
                                    response["message"] = "摄像头尚未准备好，无画面帧"
                                else:
                                    with model_lock:
                                        bboxes, landmarks = detector.detect(image=current_img)

                                    if len(bboxes) == 0:
                                        response["message"] = "画面中没有检测到人脸"
                                    else:
                                        face_alignment = norm_crop(img=current_img, landmark=landmarks[0])
                                        with model_lock:
                                            new_emb = get_feature(face_alignment)

                                        with feature_lock:
                                            if not isinstance(images_names, np.ndarray):
                                                images_names = np.array(images_names)
                                            if not isinstance(images_embs, np.ndarray):
                                                images_embs = np.array(images_embs)
                                            if not isinstance(images_descs, np.ndarray):
                                                images_descs = np.array(images_descs)

                                            is_overwrite = False
                                            if len(images_names) > 0 and name in images_names:
                                                is_overwrite = True
                                                mask = (images_names != name)
                                                images_names = images_names[mask]
                                                images_embs = images_embs[mask]
                                                images_descs = images_descs[mask]

                                            if len(images_names) == 0:
                                                images_names = np.array([name])
                                                images_embs = new_emb
                                                images_descs = np.array([desc])
                                            else:
                                                images_names = np.hstack((images_names, [name]))
                                                images_embs = np.vstack((images_embs, new_emb))
                                                images_descs = np.hstack((images_descs, [desc]))

                                            np.savez_compressed(f"{FEATURE_PATH}.npz", 
                                                                images_name=images_names,
                                                                images_emb=images_embs,
                                                                images_desc=images_descs)

                                        person_dir = os.path.join(DATASETS_DIR, name)
                                        if is_overwrite and os.path.exists(person_dir):
                                            shutil.rmtree(person_dir)
                                        os.makedirs(person_dir, exist_ok=True)
                                        cv2.imwrite(os.path.join(person_dir, f"{int(time.time())}.jpg"), current_img)

                                        response["status"] = "success"
                                        response["message"] = f"成功将 {name} 录入人脸库"

                        elif action == "delete_person":
                            name = req.get("name")
                            with feature_lock:
                                if not isinstance(images_names, np.ndarray):
                                    images_names = np.array(images_names)
                                if not isinstance(images_embs, np.ndarray):
                                    images_embs = np.array(images_embs)
                                if not isinstance(images_descs, np.ndarray):
                                    images_descs = np.array(images_descs)

                                if len(images_names) > 0 and name in images_names:
                                    mask = (images_names != name)
                                    images_names = images_names[mask]
                                    images_embs = images_embs[mask]
                                    images_descs = images_descs[mask]
                                    np.savez_compressed(f"{FEATURE_PATH}.npz", 
                                                        images_name=images_names,
                                                        images_emb=images_embs,
                                                        images_desc=images_descs)

                                    person_dir = os.path.join(DATASETS_DIR, name)
                                    if os.path.exists(person_dir):
                                        shutil.rmtree(person_dir)

                                    response["status"] = "success"
                                    response["message"] = f"已彻底删除 {name}"
                                else:
                                    response["message"] = f"未找到名为 {name} 的人脸"

                        elif action == "get_faces":
                            with feature_lock:
                                result_data = []
                                if len(images_names) > 0:
                                    # 针对 TCP 也返回详细信息
                                    for n, d in zip(images_names, images_descs):
                                        result_data.append({"name": str(n), "desc": str(d)})
                                
                            response["status"] = "success"
                            response["data"] = result_data
                            response["message"] = "查询成功"

                        elif action == "set_threshold":
                            new_threshold = req.get("threshold")
                            if isinstance(new_threshold, (int, float)) and 0.0 <= new_threshold <= 1.0:
                                global_threshold = float(new_threshold)
                                save_server_config()
                                response["status"] = "success"
                                response["message"] = f"阈值已更新为 {global_threshold}"
                            else:
                                response["message"] = "阈值必须是 0 到 1 之间"

                        elif action == "get_threshold":
                            response["status"] = "success"
                            response["data"] = global_threshold
                            response["message"] = "查询成功"

                    except Exception as inner_e:
                        traceback.print_exc()
                        response["status"] = "error"
                        response["message"] = f"服务端处理失败: {str(inner_e)}"

                    conn.sendall((json.dumps(response, ensure_ascii=False) + "\n").encode('utf-8'))

                except json.JSONDecodeError:
                    pass

    except Exception as e:
        print(f"⚠️ [TCP] 客户端异常断开: {e}")
    finally:
        if conn in active_tcp_clients:
            active_tcp_clients.remove(conn)
        conn.close()


def tcp_server_worker():
    host = '0.0.0.0'
    port = 9000
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        server_socket.bind((host, port))
        server_socket.listen(5)
        print(f"🚀 [TCP] 控制服务器已启动，监听端口: {port}")
        while True:
            conn, addr = server_socket.accept()
            threading.Thread(target=handle_tcp_client, args=(conn, addr), daemon=True).start()
    except Exception as e:
        print(f"⚠️ [TCP] 监听失败: {e}")


# ================= 5. HTTP 辅助接口 (FastAPI) =================
app = FastAPI(title="FengAI HTTP 辅助接口")
app.mount("/images", StaticFiles(directory=DATASETS_DIR), name="images")


@app.get("/config")
def get_config():
    # 🌟 返回新增字段
    return {"threshold": global_threshold, "camera_flip": server_camera_flip, "only_known": server_only_known}


@app.post("/config")
def update_config(config: ServerConfig):
    # 🌟 声明修改全局变量
    global global_threshold, server_camera_flip, server_only_known
    global_threshold = config.threshold
    server_camera_flip = config.camera_flip
    server_only_known = config.only_known  # 🌟 更新内存变量
    save_server_config()
    return {"status": "success", "message": "配置已生效"}


@app.get("/get_faces")
def get_faces():
    result = []
    with feature_lock:
        if len(images_names) == 0:
            return {"faces": []}
        
        # 建立 name 到 desc 的映射（假设 name 唯一）
        name_to_desc = {}
        for n, d in zip(images_names, images_descs):
            name_to_desc[str(n)] = str(d)

    for name, desc in name_to_desc.items():
        person_dir = os.path.join(DATASETS_DIR, str(name))
        img_url = None
        if os.path.exists(person_dir):
            imgs = [f for f in os.listdir(person_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            if imgs:
                safe_name = urllib.parse.quote(str(name))
                img_url = f"/images/{safe_name}/{imgs[0]}"
        result.append({"name": name, "desc": desc, "image_url": img_url})
    return {"faces": result}


@app.post("/add_person")
def add_person(name: str = Form(...), desc: str = Form(""), file: UploadFile = File(...)):
    # 🌟 修复
    global images_names, images_embs, images_descs

    image_data = file.file.read()
    frame = read_imagefile(image_data)

    if frame is None:
        return JSONResponse(status_code=400, content={"error": "图片解码失败"})

    with model_lock:
        bboxes, landmarks = detector.detect(image=frame)

    if len(bboxes) == 0:
        return JSONResponse(status_code=400, content={"error": "未能在图片中检测到人脸，请换一张照片"})

    face_alignment = norm_crop(img=frame, landmark=landmarks[0])
    with model_lock:
        new_emb = get_feature(face_alignment)

    with feature_lock:
        if not isinstance(images_names, np.ndarray):
            images_names = np.array(images_names)
        if not isinstance(images_embs, np.ndarray):
            images_embs = np.array(images_embs)
        if not isinstance(images_descs, np.ndarray):
            images_descs = np.array(images_descs)

        is_overwrite = False
        if len(images_names) > 0 and name in images_names:
            is_overwrite = True
            mask = (images_names != name)
            images_names = images_names[mask]
            images_embs = images_embs[mask]
            images_descs = images_descs[mask]

        if len(images_names) == 0:
            images_names = np.array([name])
            images_embs = new_emb
            images_descs = np.array([desc])
        else:
            images_names = np.hstack((images_names, [name]))
            images_embs = np.vstack((images_embs, new_emb))
            images_descs = np.hstack((images_descs, [desc]))

        np.savez_compressed(f"{FEATURE_PATH}.npz", 
                            images_name=images_names, 
                            images_emb=images_embs,
                            images_desc=images_descs)

    person_dir = os.path.join(DATASETS_DIR, name)
    if is_overwrite and os.path.exists(person_dir):
        shutil.rmtree(person_dir)

    os.makedirs(person_dir, exist_ok=True)
    save_path = os.path.join(person_dir, f"{int(time.time())}.jpg")
    cv2.imwrite(save_path, frame)

    action_str = "覆盖更新" if is_overwrite else "新增录入"
    return {"status": "success", "message": f"成功将 {name} {action_str} 到特征库！"}


@app.get("/current_frame")
def get_current_frame():
    with frame_lock:
        if latest_frame is None:
            return JSONResponse(status_code=503, content={"error": "摄像头尚未捕获到画面"})
        frame_copy = latest_frame.copy()

    success, encoded_image = cv2.imencode('.jpg', frame_copy)
    if not success:
        return JSONResponse(status_code=500, content={"error": "图像编码失败"})
    return Response(content=encoded_image.tobytes(), media_type="image/jpeg")


if __name__ == "__main__":
    load_models_and_data()

    threading.Thread(target=video_stream_worker, daemon=True).start()
    threading.Thread(target=tcp_server_worker, daemon=True).start()
    threading.Thread(target=status_broadcast_worker, daemon=True).start()

    print("🚀 [HTTP] 图像资源服务器正在启动，端口: 8000")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning")