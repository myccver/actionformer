
import json


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


results = {}

# 遍历所有视频ID（假设video_dict和pred中的视频ID一致）
for vid in video_dict:
    if 'v_'+vid not in pred:
        continue  # 跳过pred中没有的视频

    # 提取当前视频的真实时间点
    true_entries = video_dict[vid]
    true_point_times = [entry["point_time"] for entry in true_entries]

    # 提取当前视频的预测结果
    pred_entries = pred['v_'+vid]

    # 记录每个point_time对应的最高分预测
    highest_scores = {}
    # for pred_entry in pred_entries:
    #     seg_start, seg_end = pred_entry["segment"]
    #     # 检查预测时间段是否包含任何真实时间点
    #     for pt in true_point_times:
    #         if seg_start <= pt <= seg_end:
    #             # 若当前预测得分更高，则更新记录
    #             current_score = pred_entry["score"]
    #             if pt not in highest_scores or current_score > highest_scores[pt]["score"]:
    #                 highest_scores[pt] = {"score": current_score, "pred": pred_entry}

    for pred_entry in pred_entries:
        for pt in true_point_times:
            res = compute_weighted_score_with_adjusted_bounds(pred_entry, pt, alpha=0.0, beta=0.0, k=2)
            if res is None:
                continue

            weighted_score, (adj_start, adj_end) = res

            if weighted_score is not None:
                if pt not in highest_scores or weighted_score > highest_scores[pt]["score"]:
                    # 若你还想记录调整后的边界，可加一个字段
                    pred_copy = dict(pred_entry)
                    # pred_copy["adjusted_segment"] = [adj_start, adj_end]
                    pred_copy["segment"] = [adj_start, adj_end]

                    highest_scores[pt] = {
                        "score": weighted_score,
                        "pred": pred_copy
                    }

    # 按原始point_time顺序生成结果（仅保留存在的条目）
    filtered_preds = []
    for pt in true_point_times:
        if pt in highest_scores:
            filtered_preds.append(highest_scores[pt]["pred"])

    results[vid] = filtered_preds
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