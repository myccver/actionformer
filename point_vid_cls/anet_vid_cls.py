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


vid_cls_dict = {}

for vid, data in video_dict.items():
    tmp = []
    for i in data:
        if i['class'] not in tmp:
            tmp.append(i['class'])
    vid_cls_dict[vid] = tmp


label_list = ['Drinking beer', 'Dodgeball', 'Doing fencing', 'Playing congas',
       'River tubing', 'Changing car wheel', 'Rock-paper-scissors',
       'Knitting', 'Removing ice from car', 'Shoveling snow',
       'Tug of war', 'Shot put', 'Baking cookies', 'Doing crunches',
       'Baton twirling', 'Slacklining', 'Painting furniture', 'Archery',
       'Snow tubing', 'Wakeboarding', 'Ballet', 'Cleaning sink',
       'Disc dog', 'Curling', 'Playing badminton', 'Making an omelette',
       'Hanging wallpaper', 'Playing accordion', 'Rafting', 'Spinning',
       'Throwing darts', 'Playing pool', 'Getting a tattoo', 'Sailing',
       'Playing bagpipes', 'Fun sliding down', 'Smoking hookah',
       'Canoeing', 'Getting a haircut', 'Calf roping', 'Kayaking',
       'Horseback riding', 'Using the pommel horse', 'Bathing dog',
       'Rope skipping', 'Smoking a cigarette', 'Windsurfing',
       'Using the balance beam', 'Chopping wood', 'Arm wrestling',
       'Powerbocking', 'Putting on makeup', 'Starting a campfire',
       'Welding', 'Futsal', 'Shaving', 'Playing flauta',
       'Playing rubik cube', 'Painting', 'Playing lacrosse',
       'Playing piano', 'Longboarding', 'Drinking coffee',
       'Using the rowing machine', 'Making a lemonade',
       'Using parallel bars', 'Fixing the roof', 'Javelin throw',
       'Rollerblading', 'Elliptical trainer', 'Bullfighting',
       'Doing a powerbomb', 'Beer pong', 'Walking the dog',
       'Clean and jerk', 'Grooming horse', 'Hitting a pinata',
       'Braiding hair', 'Grooming dog', 'Peeling potatoes',
       'Vacuuming floor', 'Playing squash', 'Having an ice cream',
       'Tai chi', 'Playing harmonica', 'Swinging at the playground',
       'Camel ride', 'Triple jump', 'Doing kickboxing', 'Laying tile',
       'Springboard diving', 'Skiing', 'Decorating the Christmas tree',
       'Applying sunscreen', 'High jump', 'Preparing pasta',
       'Gargling mouthwash', 'Playing ten pins', 'Spread mulch',
       'Plastering', 'Drum corps', 'Doing step aerobics', 'Surfing',
       'Blowing leaves', 'Snowboarding', 'Playing drums', 'Skateboarding',
       'BMX', 'Raking leaves', 'Cleaning shoes', 'Beach soccer',
       'Ice fishing', 'Playing blackjack', 'Waterskiing', 'Waxing skis',
       'Belly dance', 'Getting a piercing', 'Doing nails',
       'Tennis serve with ball bouncing', 'Discus throw',
       'Mowing the lawn', 'Hand washing clothes', 'Wrapping presents',
       'Playing guitarra', 'Playing water polo', 'Hammer throw',
       'Roof shingle removal', 'Blow-drying hair',
       'Playing beach volleyball', 'Sumo', 'Cheerleading',
       'Bungee jumping', 'Making a cake', 'Rock climbing', 'Hopscotch',
       'Cutting the grass', 'Layup drill in basketball', 'Washing face',
       'Playing violin', 'Sharpening knives', 'Polishing forniture',
       'Ping-pong', 'Mixing drinks', 'Table soccer', 'Playing kickball',
       'Kite flying', 'Playing ice hockey', 'Building sandcastles',
       'Playing polo', 'Doing karate', 'Installing carpet',
       'Running a marathon', 'Painting fence', 'Cleaning windows',
       'Riding bumper cars', 'Ironing clothes', 'Croquet', 'Cumbia',
       'Making a sandwich', 'Capoeira', 'Putting in contact lenses',
       'Brushing teeth', 'Preparing salad', 'Tumbling',
       'Playing field hockey', 'Trimming branches or hedges', 'Long jump',
       'Brushing hair', 'Washing dishes', 'Kneeling', 'Hurling',
       'Hula hoop', 'Washing hands', 'Using the monkey bar',
       'Using uneven bars', 'Hand car wash', 'Mooping floor',
       'Scuba diving', 'Zumba', 'Putting on shoes', 'Polishing shoes',
       'Assembling bicycle', 'Shaving legs', 'Swimming',
       'Clipping cat claws', 'Shuffleboard', 'Volleyball', 'Breakdancing',
       'Paintball', 'Carving jack-o-lanterns', 'Snatch', 'Tango',
       'Cricket', 'Doing motocross', 'Pole vault', 'Playing racquetball',
       'Plataform diving', 'Fixing bicycle', 'Playing saxophone',
       'Removing curlers']


with open('/home/yunchuan/HR-Pro/ckpt/ActivityNet1.3/HR-Pro/stage1/outputs/snippet_result_test.json',
          'rb') as f:
    test_pred = json.load(f)['vid_score']

vid_thresh = 0.3

for vid, data in test_pred.items():
    vid = vid[2:]
    tmp_label_list = []

    assert len(data) == len(label_list)

    # 先找所有超过阈值的类别
    for i, score in enumerate(data):
        if score >= vid_thresh:
            tmp_label_list.append(label_list[i])

    # 如果没有任何类通过阈值，就选最大分数对应的类
    if len(tmp_label_list) == 0:
        max_idx = data.index(max(data))
        tmp_label_list = [label_list[max_idx]]

    vid_cls_dict[vid] = tmp_label_list


import pickle

with open('/home/yunchuan/actionformer/point_vid_cls/anet_vid_cls.pkl','wb') as f:
    pickle.dump(vid_cls_dict, f)