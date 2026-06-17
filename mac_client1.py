import cv2
import requests
import threading
import time
import numpy as np

# ====== 配置区：请修改为你 Ubuntu 虚拟机的实际 IP ======
UBUNTU_IP = "192.168.12.107"  # 注意：我看你之前的日志里，Ubuntu 的 IP 好像是 106
RECOGNIZE_URL = f"http://{UBUNTU_IP}:8000/recognize"
ADD_PERSON_URL = f"http://{UBUNTU_IP}:8000/add_person"
# ========================================================

# 全局变量，用于前后台线程数据共享
latest_frame_to_send = None
current_faces = []
is_running = True


def recognition_worker():
    """后台线程：负责每秒向 Ubuntu 发送几次图片，不阻塞 Mac 的摄像头画面"""
    global current_faces, latest_frame_to_send
    while is_running:
        if latest_frame_to_send is not None:
            # 拷贝一份当前帧，防止在主线程刷新时发生数据冲突
            frame_to_send = latest_frame_to_send.copy()

            # 1. 将 numpy 画面编码为 jpg 字节流
            _, img_encoded = cv2.imencode('.jpg', frame_to_send)
            files = {'file': ('frame.jpg', img_encoded.tobytes(), 'image/jpeg')}

            # 2. 发送 POST 请求给 Ubuntu
            try:
                response = requests.post(RECOGNIZE_URL, files=files, timeout=2)
                if response.status_code == 200:
                    data = response.json()

                    # 💡 关键 Debug：把服务端发回来的原话打印出来！
                    print(f"🔍 [Debug] 服务端返回: {data}")

                    # 尝试获取识别结果
                    current_faces = data.get("faces", [])
                else:
                    print(f"⚠️ [Debug] 服务端报错，状态码: {response.status_code}")

            except requests.exceptions.RequestException as e:
                # 忽略网络断开或超时报错
                pass
            except Exception as e:
                # 💡 防止因为 JSON 解析报错导致线程悄悄死亡
                print(f"❌ [Debug] 后台线程发生异常: {e}")

        # 控制发送频率 (0.2 秒一次)
        time.sleep(0.2)


def add_person_action(frame):
    """录入新的人脸的动作"""
    name = input("\n👇 请在终端输入要添加的人脸姓名 (按回车确认): ")
    if not name.strip():
        print("⚠️ 未输入姓名，取消添加。")
        return

    print(f"⏳ 正在上传 {name} 的照片到 Ubuntu 服务器进行特征提取...")
    _, img_encoded = cv2.imencode('.jpg', frame)
    files = {'file': ('frame.jpg', img_encoded.tobytes(), 'image/jpeg')}
    data = {'name': name}

    try:
        response = requests.post(ADD_PERSON_URL, files=files, data=data)
        if response.status_code == 200:
            print(f"✅ 添加成功: {response.json().get('message')}")
        else:
            print(f"❌ 添加失败: {response.text}")
    except Exception as e:
        print(f"❌ 网络请求失败，请检查 Ubuntu 服务是否开启: {e}")


# ================= 主程序 =================
# 启动后台识别请求线程
threading.Thread(target=recognition_worker, daemon=True).start()

# 打开 Mac 的本地摄像头（0 号）
cap = cv2.VideoCapture(0)

print("\n" + "=" * 50)
print("🚀 Mac 视觉前端已启动！")
print("💡 操作指南：")
print("   - 按 'a' 键：截取当前画面，并录入新的人脸")
print("   - 按 'q' 或 'ESC' 键：退出程序")
print("=" * 50 + "\n")

while True:
    ret, frame = cap.read()
    if not ret:
        print("无法读取摄像头画面！")
        break

    # 镜像翻转画面
    frame = cv2.flip(frame, 1)

    # 拷贝一帧极其干净的画面给后台线程发送
    latest_frame_to_send = frame.copy()

    # 在画面上绘制 Ubuntu 返回的识别结果
    for face in current_faces:
        try:
            # 💡 强制把坐标转成整数，专治 OpenCV 各种不服
            x1, y1, x2, y2 = map(int, face["box"])
            name = str(face.get("name", "Unknown"))

            # 兼容 score 可能不存在或者格式不对的情况
            score = face.get("score", 0.0)
            score_val = float(score) if score is not None else 0.0

            # 已知人物画绿框，UNKNOWN 画红框
            color = (0, 255, 0) if name != "UNKNOWN" else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            text = f"{name} ({score_val:.2f})"
            cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        except Exception as e:
            print(f"⚠️ [Debug] 画框失败，数据格式可能有误: {face}, 错误: {e}")

    # 显示画面
    cv2.imshow("Mac Face Recognition Frontend", frame)

    # 键盘事件监听
    key = cv2.waitKey(1) & 0xFF
    if key == 27 or key == ord('q'):
        break
    elif key == ord('a'):
        add_person_action(latest_frame_to_send)

# 退出清理
is_running = False
cap.release()
cv2.destroyAllWindows()