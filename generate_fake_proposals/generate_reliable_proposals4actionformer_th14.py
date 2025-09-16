





# import json
# import pandas as pd
#
# with open('/home/yunchuan/HR-Pro/ckpt/THUMOS14/HR-Pro_gaussian/stage1/outputs/snippet_result_train.json',
#           'rb') as f:
#     pred = json.load(f)['results']
#
# pass
#
# with open('/home/yunchuan/actionformer/data/thumos/annotations/thumos14.json', 'rb') as f:
#     gt = json.load(f)['database']
#
# label_dict = {}
# for key, value in gt.items():
#     for act in value['annotations']:
#         label_dict[act['label']] = act['label_id']
#
# # 指定 CSV 文件的路径
# file_path = '/home/yunchuan/HR-Pro/dataset/THUMOS14/point_labels/point_gaussian.csv'  # 替换为你的文件路径
#
# # 读取 CSV 文件
# data = pd.read_csv(file_path)
#
# # 打印数据的前几行来查看
# print(data.head())
# # 创建一个空字典来存储数据
# video_dict = {}
#
# # 遍历 DataFrame 的每一行
# for index, row in data.iterrows():
#     video_id = row['video_id']
#
#     # 如果字典中没有这个 video_id 的键，创建一个空列表
#     if video_id not in video_dict:
#         video_dict[video_id] = []
#
#     # 将行信息以字典形式添加到对应的列表中
#     video_dict[video_id].append({
#         'class': row['class'],
#         'start_frame': row['start_frame'],
#         'stop_frame': row['stop_frame'],
#         'point': row['point'],
#         'point_time': row['point'] / gt[video_id]['fps']
#     })
#
# # 打印整理后的字典
# print(video_dict)
#
# results = {}
#
# # 遍历所有视频ID（假设video_dict和pred中的视频ID一致）
# for vid in video_dict:
#     if vid not in pred:
#         continue  # 跳过pred中没有的视频
#
#     # 提取当前视频的真实时间点
#     true_entries = video_dict[vid]
#     true_point_times = [entry["point_time"] for entry in true_entries]
#
#     # 提取当前视频的预测结果
#     pred_entries = pred[vid]
#
#     # 记录每个point_time对应的最高分预测
#     highest_scores = {}
#     for pred_entry in pred_entries:
#         seg_start, seg_end = pred_entry["segment"]
#         # 检查预测时间段是否包含任何真实时间点
#         for pt in true_point_times:
#             if seg_start <= pt <= seg_end:
#                 # 若当前预测得分更高，则更新记录
#                 current_score = pred_entry["score"]
#                 if pt not in highest_scores or current_score > highest_scores[pt]["score"]:
#                     highest_scores[pt] = {"score": current_score, "pred": pred_entry}
#
#     # 按原始point_time顺序生成结果（仅保留存在的条目）
#     filtered_preds = []
#     for pt in true_point_times:
#         if pt in highest_scores:
#             filtered_preds.append(highest_scores[pt]["pred"])
#
#     results[vid] = filtered_preds
# for vid, data in results.items():
#     for j in data:
#         j['label_id'] = label_dict[j['label']]
# # print(results)
#
# with open('/home/yunchuan/actionformer/data/thumos/annotations/thumos14.json', 'rb') as f:
#     fake_gt = json.load(f)
#
# for vid, data in results.items():
#     fake_gt['database'][vid]['annotations'] = data
#
# with open('/home/yunchuan/actionformer/data/thumos/annotations/fake_thumos14.json', 'w') as f:
#     json.dump(fake_gt, f)
#
# pass
#
#
#
# #--------------------------------多项式高斯分布-------------------------------



import torch

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
# T = 100  # total number of frames
# t = torch.arange(T, dtype=torch.float32)  # time axis
#
# # Proposal: s=20, e=40
# s_i, e_i = 20, 40
# mu_i = (s_i + e_i) / 2
# sigma_i = (e_i - s_i) / 2
#
# # Apply bump
# phi = generalized_bump(t, mu=mu_i, sigma=sigma_i, k=2)
#
# # 可视化（可选）
# import matplotlib.pyplot as plt
# plt.plot(t.numpy(), phi.numpy())
# plt.title("Generalized Bump Function")
# plt.xlabel("Time")
# plt.ylabel("φ(t)")
# plt.grid(True)
# plt.show()


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



import json
import pandas as pd

with open('/home/yunchuan/HR-Pro/ckpt/THUMOS14/HR-Pro_gaussian/stage1/outputs/snippet_result_train.json',
          'rb') as f:
    pred = json.load(f)['results']

pass

with open('/home/yunchuan/actionformer/data/thumos/annotations/thumos14.json', 'rb') as f:
    gt = json.load(f)['database']

label_dict = {}
for key, value in gt.items():
    for act in value['annotations']:
        label_dict[act['label']] = act['label_id']

# 指定 CSV 文件的路径
file_path = '/home/yunchuan/HR-Pro/dataset/THUMOS14/point_labels/point_gaussian.csv'  # 替换为你的文件路径

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
        'point_time': row['point'] / gt[video_id]['fps']
    })

# # 打印整理后的字典
# print(video_dict)

results = {}

for vid in video_dict:
    if vid not in pred:
        continue  # 跳过没有预测的样本

    # 提取该视频的 point-level ground truth
    true_entries = video_dict[vid]
    true_point_times = [entry["point_time"] for entry in true_entries]

    # 该视频对应的预测 proposals（有置信度）
    pred_entries = pred[vid]

    # 存储每个 point_time 对应的最佳 proposal（score × bump 权重最大）
    highest_scores = {}

    # 遍历所有预测 proposal
    # for pred_entry in pred_entries:
    #     seg_start, seg_end = pred_entry["segment"]
    #     pred_score = pred_entry["score"]
    #
    #     # 构造 generalized bump 所需参数
    #     mu = (seg_start + seg_end) / 2.0
    #     sigma = (seg_end - seg_start) / 2.0
    #
    #     # 遍历该视频中所有真实 point_time
    #     for pt in true_point_times:
    #         # 判断 point_time 是否落在 proposal 时间段内
    #         if seg_start <= pt <= seg_end:
    #             # 转成 tensor 参与 bump 计算
    #             pt_tensor = torch.tensor(pt, dtype=torch.float32)
    #             mu_tensor = torch.tensor(mu, dtype=torch.float32)
    #             sigma_tensor = torch.tensor(sigma, dtype=torch.float32)
    #
    #             # 计算 bump 权重（中心最大，边缘为0）
    #             bump_weight = generalized_bump(pt_tensor, mu_tensor, sigma_tensor, k=2)
    #
    #             # 将 bump 值与置信度相乘，作为最终打分
    #             weighted_score = pred_score * bump_weight.item()
    #
    #             # 更新该 point_time 当前最高分 proposal
    #             if pt not in highest_scores or weighted_score > highest_scores[pt]["score"]:
    #                 highest_scores[pt] = {
    #                     "score": weighted_score,
    #                     "pred": pred_entry
    #                 }

    # for pred_entry in pred_entries:
    #     for pt in true_point_times:
    #         res = compute_weighted_score_with_adjusted_bounds(pred_entry, pt, alpha=0.0, beta=0.0, k=2)
    #         if res is None:
    #             continue
    #
    #         weighted_score, (adj_start, adj_end) = res
    #
    #         if weighted_score is not None:
    #             if pt not in highest_scores or weighted_score > highest_scores[pt]["score"]:
    #                 # 若你还想记录调整后的边界，可加一个字段
    #                 pred_copy = dict(pred_entry)
    #                 # pred_copy["adjusted_segment"] = [adj_start, adj_end]
    #                 pred_copy["segment"] = [adj_start, adj_end]
    #
    #                 highest_scores[pt] = {
    #                     "score": weighted_score,
    #                     "pred": pred_copy
    #                 }
    # 使用原始置信度，不做调整
    for pred_entry in pred_entries:
        seg_start, seg_end = pred_entry["segment"]
        # 检查预测时间段是否包含任何真实时间点
        for pt in true_point_times:
            if seg_start <= pt <= seg_end:
                # 若当前预测得分更高，则更新记录
                current_score = pred_entry["score"]
                if pt not in highest_scores or current_score > highest_scores[pt]["score"]:
                    highest_scores[pt] = {"score": current_score, "pred": pred_entry}

    # 构造过滤后的 proposal 列表（按原始 point_time 顺序保留）
    filtered_preds = []
    for pt in true_point_times:
        if pt in highest_scores:
            filtered_preds.append(highest_scores[pt]["pred"])

    # 保存结果
    results[vid] = filtered_preds
for vid, data in results.items():
    for j in data:
        j['label_id'] = label_dict[j['label']]
# print(results)

with open('/home/yunchuan/actionformer/data/thumos/annotations/thumos14.json', 'rb') as f:
    fake_gt = json.load(f)

for vid, data in results.items():
    fake_gt['database'][vid]['annotations'] = data

with open('/home/yunchuan/actionformer/data/thumos/annotations/fake_thumos14.json', 'w') as f:
    json.dump(fake_gt, f)

pass





























