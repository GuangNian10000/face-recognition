import numpy as np


def read_features(feature_path):
    try:
        data = np.load(feature_path + ".npz", allow_pickle=True)
        images_name = data["images_name"]
        images_emb = data["images_emb"]
        
        # 兼容性读取描述字段
        if "images_desc" in data:
            images_desc = data["images_desc"]
        else:
            # 如果是旧版数据，生成对应长度的空字符串数组
            images_desc = np.array([""] * len(images_name))

        return images_name, images_emb, images_desc
    except Exception as e:
        print(f"⚠️ [特征库] 读取失败: {e}")
        return None, None, None


def compare_encodings(encoding, encodings):
    sims = np.dot(encodings, encoding.T)
    pare_index = np.argmax(sims)
    score = sims[pare_index]
    return score, pare_index
