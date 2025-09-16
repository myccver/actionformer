import json
import pandas as pd
import numpy as np
import os
from torch.nn import functional as F
import torch

with open('/home/yunchuan/HR-Pro/dataset/ActivityNet1.3/gt_full.json', 'rb') as f:
    fps_gt = json.load(f)['database']

label_dict = {}


# 指定 CSV 文件的路径
file_path = '/home/yunchuan/HR-Pro/dataset/ActivityNet1.3/point_labels/point_gaussian.csv'  # 替换为你的文件路径

# 读取 CSV 文件
data = pd.read_csv(file_path)

# # 打印数据的前几行来查看
# print(data.head())
# 创建一个空字典来存储数据
video_dict = {}

# 遍历 DataFrame 的每一行
for index, row in data.iterrows():
    video_id = row['video_id']

    # 如果字典中没有这个 video_id 的键，创建一个空列表
    if video_id not in video_dict:
        video_dict[video_id] = []

    # 将行信息以字典形式添加到对应的列表中
    video_dict[video_id].append({
        'class': row['class'],
        'start_frame': row['start_frame'],
        'stop_frame': row['stop_frame'],
        'point': row['point'],
        'point_time': row['point'] / fps_gt[video_id]['fps']
    })



for vid, vid_data in video_dict.items():
    if "{}.npy".format(vid) not in os.listdir('/home/yunchuan/actionformer/data/anet_1.3/tsp_features'):
        continue
    feats = np.load('/home/yunchuan/actionformer/data/anet_1.3/tsp_features/{}.npy'.format(vid)).astype(np.float32)

    feat_stride = float(
        (feats.shape[0] - 1) * 16 + 16
    ) / 192
    # center the features
    num_frames = feat_stride

    feat_offset = 0.5 * num_frames / feat_stride

    # T x C -> C x T
    feats = torch.from_numpy(np.ascontiguousarray(feats.transpose()))

    resize_feats = F.interpolate(
        feats.unsqueeze(0),
        size=192,
        mode='linear',
        align_corners=False
    )
    feats = resize_feats.squeeze(0)
    feats= feats.transpose(0,1)

    # deal with downsampling (= increased feat stride)
    feats = feats[::1, :]


    for point_data in vid_data:
        if point_data['class'] not in label_dict:
            label_dict[point_data['class']] = []
        point_feat_index = int(point_data['point_time'] * 15 / feat_stride - feat_offset)
        point_feat_index = min(point_feat_index, feats.shape[0] - 1)
        point_feat = feats[point_feat_index]

        label_dict[point_data['class']].append(point_feat)



import os
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances_argmin



# 聚类参数
k = 4  # 可根据需要修改
k_str = f'k{k}'

# 创建保存目录（如果不存在则创建）
os.makedirs(f"./anet/{k_str}/cluster_centers", exist_ok=True)
os.makedirs(f"./anet/{k_str}/cluster_representatives", exist_ok=True)

for label, feature_list in label_dict.items():

    # features = np.array(feature_list)
    features = np.array([f.detach().cpu().numpy() for f in feature_list])

    if len(features) < k:
        print(f"跳过类 {label}，样本数量少于簇数（{len(features)} < {k}）")
        continue

    # 标准化
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # KMeans 聚类
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(features_scaled)

    # 获取簇中心
    cluster_centers = kmeans.cluster_centers_  # shape: (k, 2048)

    # 找出每个簇中最靠近质心的样本索引
    closest_indices = pairwise_distances_argmin(cluster_centers, features_scaled)

    # 提取这些代表样本的原始特征向量（未标准化）
    representatives = features[closest_indices]  # shape: (k, 2048)

    # 保存两个拼接结果
    np.save(os.path.join(f'./anet/{k_str}', f"cluster_centers/{label}.npy"), cluster_centers)
    np.save(os.path.join(f'./anet/{k_str}', f"cluster_representatives/{label}.npy"), representatives)

    # 打印索引
    print(f"\n【类 {label}】")
    print(f"每个簇中距离质心最近的样本索引： {closest_indices.tolist()}")
    print(f"→ 已保存: ./anet/{k_str}/cluster_centers/{label}.npy 和 ./anet/{k_str}/cluster_representatives/{label}.npy")






pass

