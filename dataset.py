import math
import pdb
import random
import re
import numpy as np
import torch
import torch.utils.data as data

from utils import load_json, load_hdf5, split_dataset, pad_sequence


class ActivityNet(data.Dataset):
    def __init__(self, opt, mode):
        self.temporal_scale = opt.ActivityNet1_3_temporal_scale
        self.multi_instance = opt.multi_instance
        self.shot = opt.shot
        self.video_features_path = opt.ActivityNet1_3_video_features_path
        self.text_features_path = opt.ActivityNet1_3_text_features_path
        self.long_text_features_path = opt.ActivityNet1_3_long_text_features_path
        self.video_annotations_path = opt.ActivityNet1_3_annotations_path
        self.mode = mode
        self.classes_list = split_dataset(dataset='ActivityNet1.3', subset=self.mode)
        self.video_annotations = load_json(self.video_annotations_path)
        self.video_features = load_hdf5(self.video_features_path)
        self.long_text_features = load_hdf5(self.long_text_features_path)
        self.text_features = load_hdf5(self.text_features_path)
        self.class_wise_database = self._get_class_wise_database()


    def __len__(self):
        # if a video has two classes' instance, we regard it as two video
        length = 0
        for label in self.class_wise_database.keys():
            length += len(self.class_wise_database[label].keys())
        return length

    def __getitem__(self, index):
        return self._load_episodic_data(index)

    def _load_info_by_name(self, video_name):
        video_info = self.video_annotations[video_name]
        return video_info

    def _load_feature_by_name(self, video_name):
        if isinstance(video_name, list):
            video_feature = []
            for item in video_name:
                temp_feature = torch.from_numpy(self.video_features[item][:])
                video_feature.append(temp_feature.to(torch.float32))
            video_feature = torch.stack(video_feature, dim=0)
        else:
            video_feature = torch.from_numpy(self.video_features[video_name][:])
            video_feature = video_feature.unsqueeze(0).to(torch.float32)
        # video_feature [b, t, c] tensor

        if isinstance(video_name, list):
            text_feature = []
            for item in video_name:
                temp_feature = torch.from_numpy(self.text_features[item][:])
                text_feature.append(temp_feature.to(torch.float32))
            text_feature = torch.stack(text_feature, dim=0)
        else:
            text_feature = torch.from_numpy(self.text_features[video_name][:])
            text_feature = text_feature.unsqueeze(0).to(torch.float32)
        # text_feature [b, t, c] tensor

        if isinstance(video_name, list):
            long_text_feature = []
            for item in video_name:
                temp_feature = torch.from_numpy(self.long_text_features[item][:])
                long_text_feature.append(temp_feature.to(torch.float32))
            long_text_feature = pad_sequence(long_text_feature, 13)
        else:
            if re.search(r'_segment_\d+$', video_name):
                video_name = re.sub(r'_segment_\d+$', '', video_name)
            long_text_feature = torch.from_numpy(self.long_text_features[video_name][:])
            long_text_feature = long_text_feature.unsqueeze(0).to(torch.float32)
        return video_feature, text_feature, long_text_feature

    def _load_label_by_name(self, video_name, chosen_class):
        if isinstance(video_name, list):
            masks = []
            for item in video_name:
                mask = torch.zeros((self.temporal_scale, 1))
                video_info = self._load_info_by_name(item)
                video_annotations = video_info['annotations']
                video_frame = video_info['duration_frame']
                video_second = video_info['duration_second']
                feature_frame = video_info['feature_frame']
                corrected_second = float(feature_frame) / video_frame * video_second
                gt_box = []
                for annotation in video_annotations:
                    if annotation['label'] == chosen_class:
                        segment = annotation['segment']
                        start = max(min(1, float(segment[0]) / corrected_second), 0)
                        end = max(min(1, float(segment[1]) / corrected_second), 0)
                        gt_box.append([start, end])
                gt_box = np.array(gt_box)
                gt_x_min = gt_box[:, 0]
                gt_x_max = gt_box[:, 1]
                start_indexes = []
                end_indexes = []
                for idx in range(len(gt_x_min)):
                    start_indexes.append(math.floor(self.temporal_scale * gt_x_min[idx]))
                    end_indexes.append(math.floor(self.temporal_scale * gt_x_max[idx]))
                for i in range(len(start_indexes)):
                    mask[start_indexes[i]:end_indexes[i]] = 1
                masks.append(mask)
            masks = torch.stack(masks, dim=0)
            return masks

        else:
            mask = torch.zeros((self.temporal_scale, 1))
            video_info = self._load_info_by_name(video_name)
            video_annotations = video_info['annotations']
            video_frame = video_info['duration_frame']
            video_second = video_info['duration_second']
            feature_frame = video_info['feature_frame']
            corrected_second = float(feature_frame) / video_frame * video_second
            gt_box = []
            for annotation in video_annotations:
                if annotation['label'] == chosen_class:
                    segment = annotation['segment']
                    start = max(min(1, float(segment[0]) / corrected_second), 0)
                    end = max(min(1, float(segment[1]) / corrected_second), 0)
                    gt_box.append([start, end])
            gt_box = np.array(gt_box)
            gt_x_min = gt_box[:, 0]
            gt_x_max = gt_box[:, 1]
            start_indexes = []
            end_indexes = []
            for idx in range(len(gt_x_min)):
                start_indexes.append(math.floor(self.temporal_scale * gt_x_min[idx]))
                end_indexes.append(math.floor(self.temporal_scale * gt_x_max[idx]))
            for i in range(len(start_indexes)):
                mask[start_indexes[i]:end_indexes[i]] = 1
            return mask.unsqueeze(0)

    def _get_class_wise_database(self):
        database = {class_label: {} for class_label in self.classes_list}
        video_1 = list(self.long_text_features.keys())
        video_2 = list(self.text_features.keys())
        for video_name in self.video_annotations.keys():
            if video_name not in video_1 or video_name not in video_2:
                continue
            video_info = self._load_info_by_name(video_name)
            try:
                corrected_second = video_info['duration']
            except:
                corrected_second = video_info['duration_second'] * video_info['feature_frame'] / video_info[
                    'duration_frame']

            if not self.multi_instance:
                if len(self.video_annotations[video_name]['annotations']) > 1:
                    continue

            cnt = 0
            for annotation in self.video_annotations[video_name]['annotations']:
                corr_start = max(min(1, float(annotation['segment'][0]) / corrected_second), 0)
                corr_end = max(min(1, float(annotation['segment'][1]) / corrected_second), 0)
                if (corr_end - corr_start) < 0.1:
                    cnt += 1
            if cnt > 0:
                continue
            for annotation in self.video_annotations[video_name]['annotations']:
                label = annotation['label']
                if label in self.classes_list:
                    if video_name not in database[label].keys():
                        database[label][video_name] = [annotation['segment']]
                    else:
                        database[label][video_name].append(annotation['segment'])
        return database

    def _load_episodic_data(self, index):
        chosen_class = random.choice(self.classes_list)
        chosen_video_name = random.sample(self.class_wise_database[chosen_class].keys(), 1 + self.shot)
        query_video_name = chosen_video_name[0]
        support_video_name = chosen_video_name[1: self.shot + 1]
        query_data = self._load_feature_by_name(query_video_name)
        query_label = self._load_label_by_name(query_video_name, chosen_class)
        support_data = self._load_feature_by_name(support_video_name)
        support_label = self._load_label_by_name(support_video_name, chosen_class)

        return query_data, query_label, query_video_name, support_data, support_label, support_video_name, chosen_class


