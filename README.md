# Real-Time Face Recognition Stream Server

本项目是一个高性能的人脸识别流媒体服务器，基于 PyTorch 实现。它集成了人脸检测、特征提取与实时流处理，支持通过 **TCP** 发送实时识别结果，并通过 **HTTP** 接口进行人员管理与配置。

<p align="center">
  <img src="./assets/face-recognition.gif" alt="Face Recognition" />
  <br>
  <em>实时人脸识别系统</em>
</p>

## 主要特性

- **双协议支持**：
  - **TCP (Port 9000)**：用于实时推送识别结果（人脸坐标、姓名、描述、置信度）以及接收控制指令（心跳、增删查）。
  - **HTTP (Port 8000)**：用于人员录入、删除、查询、获取实时画面帧以及服务端配置修改。
- **核心算法**：
  - **检测**：YOLOv5-Face (默认使用 `yolov5n-face`)。
  - **识别**：ArcFace (默认使用 `iresnet100`)。
- **动态管理**：无需重启服务即可通过 TCP/HTTP 实时录入或删除人脸特征。
- **配置持久化**：支持动态调整阈值、画面翻转等配置，并自动保存到本地 JSON。

---

## 快速开始

### 1. 环境准备

推荐使用 Python 3.9+ 环境：

```shell
conda create -n face-dev python=3.9
conda activate face-dev

# 安装依赖项
pip install -r requirements.txt
```

### 2. 模型权重

由于模型文件较大，未包含在 Git 仓库中。请确保以下路径存在对应的权重文件：
- `face_detection/yolov5_face/weights/yolov5n-face.pt`
- `face_recognition/arcface/weights/arcface_r100.pth`

### 3. 启动服务

你可以直接运行 Python 脚本或使用提供的启动脚本：

```shell
# 方式一：直接运行
python server_stream.py

# 方式二：使用 shell 脚本启动 (推荐)
chmod +x start_server.sh
./start_server.sh
```

---

## 接口说明

### TCP 控制接口 (Port 9000)
服务端启动后会监听 9000 端口，以 **JSON 换行符（`\n`）** 的形式与客户端通讯。

- **事件推送**：服务端自动推送格式如下：
  ```json
  {"type": "face_event", "count": 1, "faces": [{"name": "张三", "desc": "管理员", "score": 0.92, "box": [x, y, w, h]}]}
  ```
- **控制指令**：支持 `add_person`, `delete_person`, `get_faces`, `set_threshold`, `ping` 等指令。

### HTTP 管理接口 (Port 8000)
- `GET /get_faces`: 获取当前所有已录入的人员列表及头像 URL。
- `POST /add_person`: 表单提交，上传图片并录入人脸特征。
- `GET /current_frame`: 获取当前摄像头的实时抓拍画面（JPEG 格式）。
- `GET /config` / `POST /config`: 查询或修改服务端配置（阈值、镜像反转、仅识别熟人模式）。
- `GET /images/`: 静态资源服务，用于查看录入的人脸底照。

---

## 目录结构

```text
.
├── datasets/
│   ├── data/            # 原始人脸底照存储 (按姓名分文件夹)
│   ├── face_features/   # 特征库文件 (feature.npz)
│   └── server_config.json # 服务端持久化配置文件
├── face_detection/      # YOLOv5-Face 检测模型封装
├── face_recognition/    # ArcFace 识别模型封装
├── server_stream.py     # 🚀 核心流媒体服务程序 (TCP + HTTP)
└── start_server.sh      # 启动脚本
```

## 技术参考

- [InsightFace](https://github.com/deepinsight/insightface)
- [Yolov5-face](https://github.com/deepcam-cn/yolov5-face)
- [FastAPI](https://fastapi.tiangolo.com/)
