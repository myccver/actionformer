import os
import json
import numpy as np

import torch
from torch.utils.data import Dataset
from torch.nn import functional as F

from .datasets import register_dataset
from .data_utils import truncate_feats, truncate_feats_with_points

@register_dataset("thumos")
class THUMOS14Dataset(Dataset):
    def __init__(
        self,
        is_training,     # if in training mode
        split,           # split, a tuple/list allowing concat of subsets
        feat_folder,     # folder for features
        json_file,       # json file for annotations
        feat_stride,     # temporal stride of the feats
        num_frames,      # number of frames for each feat
        default_fps,     # default fps
        downsample_rate, # downsample rate for feats
        max_seq_len,     # maximum sequence length during training
        trunc_thresh,    # threshold for truncate an action segment
        crop_ratio,      # a tuple (e.g., (0.9, 1.0)) for random cropping
        input_dim,       # input feat dim
        num_classes,     # number of action categories
        file_prefix,     # feature file prefix if any
        file_ext,        # feature file extension if any
        force_upsampling # force to upsample to max_seq_len
    ):
        # file path
        assert os.path.exists(feat_folder) and os.path.exists(json_file)
        assert isinstance(split, tuple) or isinstance(split, list)
        assert crop_ratio == None or len(crop_ratio) == 2
        self.feat_folder = feat_folder
        if file_prefix is not None:
            self.file_prefix = file_prefix
        else:
            self.file_prefix = ''
        self.file_ext = file_ext
        self.json_file = json_file

        # split / training mode
        self.split = split
        self.is_training = is_training

        # features meta info
        self.feat_stride = feat_stride
        self.num_frames = num_frames
        self.input_dim = input_dim
        self.default_fps = default_fps
        self.downsample_rate = downsample_rate
        self.max_seq_len = max_seq_len
        self.trunc_thresh = trunc_thresh
        self.num_classes = num_classes
        self.label_dict = None
        self.crop_ratio = crop_ratio

        # load database and select the subset
        dict_db, label_dict = self._load_json_db(self.json_file)
        assert len(label_dict) == num_classes
        self.data_list = dict_db
        self.label_dict = label_dict

        # dataset specific attributes
        self.db_attributes = {
            'dataset_name': 'thumos-14',
            'tiou_thresholds': np.linspace(0.1, 0.7, 7),
            # we will mask out cliff diving
            'empty_label_ids': [],
        }

        # load point_label
        self.point_dict = self._load_point_label()

    def get_attributes(self):
        return self.db_attributes

    def _load_json_db(self, json_file):
        # load database and select the subset
        with open(json_file, 'r') as fid:
            json_data = json.load(fid)
        json_db = json_data['database']

        # if label_dict is not available
        if self.label_dict is None:
            label_dict = {}
            for key, value in json_db.items():
                for act in value['annotations']:
                    label_dict[act['label']] = act['label_id']

        # fill in the db (immutable afterwards)
        dict_db = tuple()
        for key, value in json_db.items():
            # skip the video if not in the split
            if value['subset'].lower() not in self.split:
                continue
            # or does not have the feature file
            feat_file = os.path.join(self.feat_folder,
                                     self.file_prefix + key + self.file_ext)
            if not os.path.exists(feat_file):
                continue

            # get fps if available
            if self.default_fps is not None:
                fps = self.default_fps
            elif 'fps' in value:
                fps = value['fps']
            else:
                assert False, "Unknown video FPS."

            # get video duration if available
            if 'duration' in value:
                duration = value['duration']
            else:
                duration = 1e8

            # get annotations if available
            if ('annotations' in value) and (len(value['annotations']) > 0):
                # a fun fact of THUMOS: cliffdiving (4) is a subset of diving (7)
                # our code can now handle this corner case
                segments, labels = [], []
                for act in value['annotations']:
                    segments.append(act['segment'])
                    labels.append([label_dict[act['label']]])

                segments = np.asarray(segments, dtype=np.float32)
                labels = np.squeeze(np.asarray(labels, dtype=np.int64), axis=1)
            else:
                segments = None
                labels = None
            dict_db += ({'id': key,
                         'fps' : fps,
                         'duration' : duration,
                         'segments' : segments,
                         'labels' : labels
            }, )

        return dict_db, label_dict

    def _load_point_label(self, csv_file='/home/yunchuan/HR-Pro/dataset/THUMOS14/point_labels/point_gaussian.csv'):
        with open('/home/yunchuan/actionformer/data/thumos/annotations/thumos14.json', 'rb') as f:
            gt = json.load(f)['database']

        import pandas as pd
        # 读取 CSV 文件
        data = pd.read_csv(csv_file)

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
        return video_dict

    def match_and_order_points_by_segments(self, segments, point_list):
        """
        将 point_list 中的点按 segments 的顺序排列，确保每个 point_time 落在对应 segment 内。

        参数:
            segments (np.ndarray): shape [N, 2] 的 segment 数组
            point_list (List[Dict]): 每个元素包含 'point_time' 键

        返回:
            ordered_points (List[Dict]): 匹配并排序后的 point_list
        """
        unused_points = point_list.copy()
        ordered_points = []

        for seg_idx, (seg_start, seg_end) in enumerate(segments):
            match_found = False
            for i, pt_entry in enumerate(unused_points):
                pt = pt_entry['point_time']
                if seg_start <= pt <= seg_end:
                    # ordered_points.append(pt_entry)
                    ordered_points.append(pt)
                    del unused_points[i]
                    match_found = True
                    break
            if not match_found:
                print(f"⚠️ Segment {seg_idx}: No point_time found in segment ({seg_start}, {seg_end})")

        assert len(ordered_points) == len(segments)
        # if len(ordered_points) != len(segments):
        #     print(f"❌ Mismatch: {len(ordered_points)} points matched for {len(segments)} segments.")
        # else:
        #     print("✅ All segments matched with corresponding point_time.")

        return ordered_points

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        # directly return a (truncated) data point (so it is very fast!)
        # auto batching will be disabled in the subsequent dataloader
        # instead the model will need to decide how to batch / preporcess the data
        video_item = self.data_list[idx]

        # 匹配点和segment
        if self.is_training:
            segments = video_item['segments']
            points = self.point_dict[video_item['id']]
            ordered_points = self.match_and_order_points_by_segments(segments, points)


        # load features
        filename = os.path.join(self.feat_folder,
                                self.file_prefix + video_item['id'] + self.file_ext)
        feats = np.load(filename).astype(np.float32)

        # deal with downsampling (= increased feat stride)
        feats = feats[::self.downsample_rate, :]
        feat_stride = self.feat_stride * self.downsample_rate
        feat_offset = 0.5 * self.num_frames / feat_stride
        # T x C -> C x T
        feats = torch.from_numpy(np.ascontiguousarray(feats.transpose()))


        # convert time stamp (in second) into temporal feature grids
        # ok to have small negative values here
        if video_item['segments'] is not None:
            segments = torch.from_numpy(
                video_item['segments'] * video_item['fps'] / feat_stride - feat_offset
            )
            labels = torch.from_numpy(video_item['labels'])
            points = None
            if self.is_training:
                # 转化point从时间到特征索引
                points = torch.from_numpy(np.array(ordered_points) * video_item['fps'] / feat_stride - feat_offset)

        else:
            segments, labels = None, None

        # return a data dict
        data_dict = {'video_id'        : video_item['id'],
                     'feats'           : feats,      # C x T
                     'segments'        : segments,   # N x 2
                     'labels'          : labels,     # N
                     'fps'             : video_item['fps'],
                     'duration'        : video_item['duration'],
                     'feat_stride'     : feat_stride,
                     'feat_num_frames' : self.num_frames,
                     'points'          : points  # N *2
                     }

        # truncate the features during training
        if self.is_training and (segments is not None):
            # data_dict = truncate_feats(
            #     data_dict, self.max_seq_len, self.trunc_thresh, feat_offset, self.crop_ratio
            # )
            data_dict = truncate_feats_with_points(
                data_dict, self.max_seq_len, self.trunc_thresh, feat_offset, self.crop_ratio
            )


        return data_dict


