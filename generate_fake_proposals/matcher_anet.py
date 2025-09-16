
import json
import numpy as np
from scipy.optimize import linear_sum_assignment


train_list = [item.strip() for item in list(open('/home/yunchuan/HR-Pro/dataset/ActivityNet1.3/split_train.txt'))]
test_list = [item.strip() for item in list(open('/home/yunchuan/HR-Pro/dataset/ActivityNet1.3/split_test.txt'))]
total_list = train_list +test_list
pass



with open('/home/yunchuan/actionformer/data/anet_1.3/annotations/anet1.3_tsp_filtered.json', 'rb') as f:
    gt = json.load(f)



to_delete = [vid for vid in gt['database']
             if 'v_'+vid not in test_list and gt['database'][vid]['subset'] == 'validation']
for vid in to_delete:
    del gt['database'][vid]

# to_delete = [vid for vid in gt['database']
#              if 'v_'+vid not in train_list and gt['database'][vid]['subset'] == 'training']
# for vid in to_delete:
#     del gt['database'][vid]

pass

with open('/home/yunchuan/actionformer/data/anet_1.3/annotations/anet1.3_fake_gt.json', 'w') as f:
    json.dump(gt, f)



def generalized_bump(t, mu, sigma, k=2):
    """
    Generalized bump function:
    φ(t) = (1 - ((t - mu)/sigma)^2)^k * exp(- (t - mu)^2 / (2 * sigma^2))

    Args:
        t (Tensor): 1D time vector, shape (T,)
        mu (float or Tensor): center position (can be scalar or shape (N,))
        sigma (float or Tensor): width (half duration), same shape as mu
        k (int): polynomial degree controlling boundary sharpness

    Returns:
        Tensor: bump value at each t, shape (T,) or (N, T) if batched
    """
    x = (t - mu) / sigma
    poly = torch.clamp(1 - x**2, min=0.0) ** k  # clip to avoid NaNs outside [-1, 1]
    gauss = torch.exp(-0.5 * x**2)
    return poly * gauss


import torch


def compute_weighted_score_with_adjusted_bounds(pred_entry, pt, alpha=0.5, beta=0.5, k=2):
    """
    先在原始 proposal 区间上计算 bump score，然后再生成调整后的边界。
    """
    seg_start, seg_end = pred_entry["segment"]
    pred_score = pred_entry["score"]

    if not (seg_start <= pt <= seg_end):
        return None

    # ✅ bump score 基于原始 seg_start, seg_end
    mu = (seg_start + seg_end) / 2.0
    sigma = (seg_end - seg_start) / 2.0

    pt_tensor = torch.tensor(pt, dtype=torch.float32)
    mu_tensor = torch.tensor(mu, dtype=torch.float32)
    sigma_tensor = torch.tensor(sigma, dtype=torch.float32)

    bump_value = generalized_bump(pt_tensor, mu_tensor, sigma_tensor, k=k)
    weighted_score = pred_score * bump_value.item()

    # ✅ 之后再调整边界（输出用于后续处理）
    left_offset = pt - seg_start
    right_offset = seg_end - pt


    adjusted_start = min(seg_start + alpha * left_offset, pt)
    adjusted_end = max(seg_end - beta * right_offset, pt)

    return weighted_score, (adjusted_start, adjusted_end)


# 替换train

import json
import pandas as pd



with open('/home/yunchuan/HR-Pro/ckpt/ActivityNet1.3/HR-Pro/stage1/outputs/snippet_result_train.json', 'rb') as f:
    pred = json.load(f)['results']

pass

with open('/home/yunchuan/actionformer/data/anet_1.3/annotations/anet1.3_fake_gt.json', 'rb') as f:
    gt = json.load(f)['database']

with open('/home/yunchuan/HR-Pro/dataset/ActivityNet1.3/gt_full.json', 'rb') as f:
    fps_gt = json.load(f)['database']

label_dict = {}
for key, value in gt.items():
    for act in value['annotations']:
        label_dict[act['label']] = act['label_id']

# 指定 CSV 文件的路径
file_path = '/home/yunchuan/HR-Pro/dataset/ActivityNet1.3/point_labels/point_gaussian.csv'  # 替换为你的文件路径

# 读取 CSV 文件
data = pd.read_csv(file_path)

# 打印数据的前几行来查看
# print(data.head())
# 创建一个空字典来存储数据
video_dict = {}

# 遍历 DataFrame 的每一行
for index, row in data.iterrows():
    video_id = row['video_id'][2:]

    # 如果字典中没有这个 video_id 的键，创建一个空列表
    if video_id not in video_dict:
        video_dict[video_id] = []

    # 将行信息以字典形式添加到对应的列表中
    video_dict[video_id].append({
        'class': row['class'],
        'start_frame': row['start_frame'],
        'stop_frame': row['stop_frame'],
        'point': row['point'],
        'point_time': row['point']/fps_gt[row['video_id']]['fps']
    })

# 打印整理后的字典
#print(video_dict)


def match_points_with_proposals(points, proposals, compute_score_fn, invalid_score=-1e6):
    num_points = len(points)
    num_proposals = len(proposals)
    score_matrix = np.full((num_points, num_proposals), fill_value=invalid_score, dtype=np.float32)

    for i, pt in enumerate(points):
        for j, pred in enumerate(proposals):
            res = compute_score_fn(pred, pt)
            if res is not None:
                score, _ = res
                score_matrix[i, j] = max(score, 0.0)

    max_score = np.max(score_matrix)
    cost_matrix = max_score - score_matrix

    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    matched_pairs = []
    for i, j in zip(row_ind, col_ind):
        if score_matrix[i, j] > invalid_score:
            matched_pairs.append({
                "point_time": points[i],
                "proposal_index": j,
                "score": score_matrix[i, j],
                "proposal": proposals[j]
            })

    return matched_pairs

# --------- 主流程代码开始 ---------

alpha = 0.0
beta = 0.0
k = 2
INVALID_SCORE = -1e6

results = {}

for vid in video_dict:
    if 'v_' + vid not in pred:
        continue  # 跳过pred中没有的视频

    true_entries = video_dict[vid]
    true_point_times = [entry["point_time"] for entry in true_entries]
    pred_entries = pred['v_'+vid]

    # 构建匹配关系
    num_points = len(true_point_times)
    num_proposals = len(pred_entries)
    score_matrix = np.full((num_points, num_proposals), fill_value=INVALID_SCORE, dtype=np.float32)

    for i, pt in enumerate(true_point_times):
        for j, pred_entry in enumerate(pred_entries):
            res = compute_weighted_score_with_adjusted_bounds(pred_entry, pt, alpha=alpha, beta=beta, k=k)
            if res is not None:
                score, _ = res
                score_matrix[i, j] = max(score, 0.0)

    max_score = np.max(score_matrix)
    cost_matrix = max_score - score_matrix
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # 收集有效匹配
    filtered_preds = []
    for i, j in zip(row_ind, col_ind):
        if score_matrix[i, j] > INVALID_SCORE:
            pt = true_point_times[i]
            pred_copy = dict(pred_entries[j])  # 深拷贝
            res = compute_weighted_score_with_adjusted_bounds(pred_copy, pt, alpha=alpha, beta=beta, k=k)
            if res is not None:
                _, (adj_start, adj_end) = res
                pred_copy["segment"] = [adj_start, adj_end]
                filtered_preds.append(pred_copy)

    results[vid] = filtered_preds

# --------- 结果保存在 results 字典中 ---------





for vid ,data in results.items():
    for j in data:
        j['label_id'] = label_dict[j['label']]
#print(results)



with open('/home/yunchuan/actionformer/data/anet_1.3/annotations/anet1.3_fake_gt.json', 'rb') as f:
    fake_gt = json.load(f)

for vid, data in results.items():
    if len(data)==0:
        print(vid)
        continue
    if vid in fake_gt['database']:
        fake_gt['database'][vid]['annotations'] = data
    else:
        print('{} not in fake_gt'.format(vid))
        pass

del_vid_list = []
# 准备删除缺少点标注的gt框
for vid,data in fake_gt['database'].items():
    if vid not in results.keys() and data['subset'] == 'training':
        del_vid_list.append(vid)

for vid in del_vid_list:
    del fake_gt['database'][vid]


with open('/home/yunchuan/actionformer/data/anet_1.3/annotations/anet1.3_fake_gt.json', 'w') as f:
    json.dump(fake_gt, f)
pass