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

pass

vid_cls_dict = {}

for vid, data in video_dict.items():
    tmp = []
    for i in data:
        if i['class'] not in tmp:
            tmp.append(i['class'])
    vid_cls_dict[vid] = tmp


pass



label_list = [
        'BaseballPitch', 'BasketballDunk', 'Billiards', 'CleanAndJerk',
        'CliffDiving', 'CricketBowling', 'CricketShot', 'Diving',
        'FrisbeeCatch', 'GolfSwing', 'HammerThrow', 'HighJump', 'JavelinThrow',
        'LongJump', 'PoleVault', 'Shotput', 'SoccerPenalty', 'TennisSwing',
        'ThrowDiscus', 'VolleyballSpiking'
    ]



with open('/home/yunchuan/HR-Pro/ckpt/THUMOS14/HR-Pro_gaussian/stage1/outputs/snippet_result_test.json',
          'rb') as f:
    test_pred = json.load(f)['vid_score']

vid_thresh = 0.5
correct_count = 0
total_count = len(test_pred)
for vid, data in test_pred.items():
    tmp_label_list = []
    ref_gt_list = []

    assert len(data) == len(label_list)

    # 先找所有超过阈值的类别
    for i, score in enumerate(data):
        if score >= vid_thresh:
            tmp_label_list.append(label_list[i])
    for j in gt[vid]['annotations']:
        ref_gt_list.append(j['label'])
    if set(tmp_label_list) == set(ref_gt_list):
        correct_count+=1

    # 如果没有任何类通过阈值，就选最大分数对应的类
    if len(tmp_label_list) == 0:
        max_idx = data.index(max(data))
        tmp_label_list = [label_list[max_idx]]

    vid_cls_dict[vid] = tmp_label_list


import pickle

with open('/home/yunchuan/actionformer/point_vid_cls/thumos_vid_cls.pkl','wb') as f:
    pickle.dump(vid_cls_dict, f)
