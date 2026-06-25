import numpy as np

def read_features(feature_path):
    try:
        data = np.load(feature_path + ".npz", allow_pickle=True)
        images_name = data["images_name"]
        images_emb = data["images_emb"]
        
        # 🌟 兼容性处理：如果不存在描述字段，则生成同等长度的空字符串数组
        if "images_desc" in data:
            images_desc = data["images_desc"]
        else:
            images_desc = np.array([""] * len(images_name))

        return images_name, images_emb, images_desc
    except Exception as e:
        print(f"⚠️ [特征库] 读取失败或文件不存在: {e}")
        return None, None, None

def compare_encodings(encoding, encodings):
    # 计算余弦相似度
    sims = np.dot(encodings, encoding.T)
    # 转换为 1D 数组（如果是多维的话）
    sims = sims.flatten()
    pare_index = np.argmax(sims)
    score = sims[pare_index]
    return [score], pare_index
