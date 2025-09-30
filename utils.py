import json
import os
import scipy
import pdb
import torch.nn.functional as F
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.manifold import TSNE
import random


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


# dataset utils
def split_dataset(dataset, subset):
    assert subset in ['train', 'validation', 'test'] and dataset in ['THUMOS14',
                                                                     'ActivityNet1.3']
    classes_list = []
    if dataset == 'THUMOS14_I3D':
        if subset == 'train':
            classes_list = ['BaseballPitch', 'BasketballDunk', 'Billiards', 'CleanAndJerk', 'CliffDiving',
                            'CricketBowling', 'CricketShot', 'Diving', 'FrisbeeCatch', 'GolfSwing', 'HammerThrow',
                            'HighJump', 'JavelinThrow', 'LongJump', 'PoleVault', 'Shotput']
        elif subset == 'validation':
            classes_list = ['SoccerPenalty', 'TennisSwing']
        elif subset == 'test':
            classes_list = ['ThrowDiscus', 'VolleyballSpiking']
    else:
        if subset == 'train':
            classes_list = ['Fun sliding down', 'Beer pong', 'Getting a piercing', 'Shoveling snow', 'Kneeling',
                            'Tumbling', 'Playing water polo', 'Washing dishes', 'Blowing leaves', 'Playing congas',
                            'Making a lemonade', 'Playing kickball', 'Removing ice from car', 'Playing racquetball',
                            'Swimming', 'Playing bagpipes', 'Painting', 'Assembling bicycle', 'Playing violin',
                            'Surfing', 'Making a sandwich', 'Welding', 'Hopscotch', 'Gargling mouthwash',
                            'Baking cookies', 'Braiding hair', 'Capoeira', 'Slacklining', 'Plastering',
                            'Changing car wheel', 'Chopping wood', 'Removing curlers', 'Horseback riding',
                            'Smoking hookah', 'Doing a powerbomb', 'Playing ten pins', 'Getting a haircut',
                            'Playing beach volleyball', 'Making a cake', 'Clean and jerk',
                            'Trimming branches or hedges', 'Drum corps', 'Windsurfing', 'Kite flying',
                            'Using parallel bars', 'Doing kickboxing', 'Cleaning shoes', 'Playing field hockey',
                            'Playing squash', 'Rollerblading', 'Playing drums', 'Playing rubik cube',
                            'Sharpening knives', 'Zumba', 'Raking leaves', 'Bathing dog', 'Tug of war',
                            'Ping-pong', 'Using the balance beam', 'Playing lacrosse', 'Scuba diving',
                            'Preparing pasta', 'Brushing teeth', 'Playing badminton', 'Mixing drinks',
                            'Discus throw', 'Playing ice hockey', 'Doing crunches', 'Wrapping presents',
                            'Hand washing clothes', 'Rock climbing', 'Cutting the grass', 'Wakeboarding', 'Futsal',
                            'Playing piano', 'Baton twirling', 'Mooping floor', 'Triple jump', 'Longboarding',
                            'Polishing shoes', 'Doing motocross', 'Arm wrestling', 'Doing fencing', 'Hammer throw',
                            'Shot put', 'Playing pool', 'Blow-drying hair', 'Cricket', 'Spinning',
                            'Running a marathon', 'Table soccer', 'Playing flauta', 'Ice fishing', 'Tai chi',
                            'Archery', 'Shaving', 'Using the monkey bar', 'Layup drill in basketball',
                            'Spread mulch', 'Skateboarding', 'Canoeing', 'Mowing the lawn', 'Beach soccer',
                            'Hanging wallpaper', 'Tango', 'Disc dog', 'Powerbocking', 'Getting a tattoo',
                            'Doing nails', 'Snowboarding', 'Putting on shoes', 'Clipping cat claws', 'Snow tubing',
                            'River tubing', 'Putting on makeup', 'Decorating the Christmas tree', 'Fixing bicycle',
                            'Hitting a pinata', 'High jump', 'Doing karate', 'Kayaking', 'Grooming dog',
                            'Bungee jumping', 'Washing hands', 'Painting fence', 'Doing step aerobics',
                            'Installing carpet', 'Playing saxophone', 'Long jump', 'Javelin throw',
                            'Playing accordion', 'Smoking a cigarette', 'Belly dance', 'Playing polo',
                            'Throwing darts', 'Roof shingle removal', 'Tennis serve with ball bouncing', 'Skiing',
                            'Peeling potatoes', 'Elliptical trainer', 'Building sandcastles', 'Drinking beer',
                            'Rock-paper-scissors', 'Using the pommel horse', 'Croquet', 'Laying tile',
                            'Cleaning windows', 'Fixing the roof', 'Springboard diving', 'Waterskiing',
                            'Using uneven bars', 'Having an ice cream', 'Sailing', 'Washing face', 'Knitting',
                            'Bullfighting', 'Applying sunscreen', 'Painting furniture', 'Grooming horse',
                            'Carving jack-o-lanterns']
        elif subset == 'validation':
            classes_list = ['Swinging at the playground', 'Dodgeball', 'Ballet', 'Playing harmonica', 'Paintball',
                            'Cumbia', 'Rafting', 'Hula hoop', 'Cheerleading', 'Vacuuming floor',
                            'Playing blackjack', 'Waxing skis', 'Curling', 'Using the rowing machine',
                            'Ironing clothes', 'Playing guitarra', 'Sumo', 'Putting in contact lenses',
                            'Brushing hair', 'Volleyball']
        elif subset == 'test':
            classes_list = ['Hurling', 'Polishing forniture', 'BMX', 'Riding bumper cars', 'Starting a campfire',
                            'Walking the dog', 'Preparing salad', 'Plataform diving', 'Breakdancing', 'Camel ride',
                            'Hand car wash', 'Making an omelette', 'Shuffleboard', 'Calf roping', 'Shaving legs',
                            'Snatch', 'Cleaning sink', 'Rope skipping', 'Drinking coffee', 'Pole vault']

    return classes_list


# general utilss
def load_json(file_path):
    with open(file_path, 'r') as json_file:
        json_data = json.load(json_file)
        return json_data


def load_hdf5(file_path):
    dataset = h5py.File(file_path, "r")
    return dataset


def load_txt(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    return [line for line in lines]


def load_txt_by_name(file_names, folder_name):
    if isinstance(file_names, list):
        text = []
        for name in file_names:
            file_path = os.path.join(folder_name, name[0] + '.txt')
            temp = load_txt(file_path)
            text.append(temp)
    else:
        file_path = os.path.join(folder_name, file_names[0] + '.txt')
        text = load_txt(file_path)
    return text

def create_dir(path):
    if os.path.exists(path):
        return
    else:
        os.makedirs(path)
        return

def get_file_names(directory, split=True):
    files = os.listdir(directory)
    if split:
        file_names = [os.path.splitext(f)[0] for f in files if os.path.isfile(os.path.join(directory, f))]
    else:
        file_names = [f for f in files if os.path.isfile(os.path.join(directory, f))]
    return file_names



# evaluation utils

def count_parameters(model):
    s = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Total number of parameters: {s}')

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def GetBoundary(flags):
    """
        input:  flags  bool
        output: boundary [(,), ...]
    """
    flags_1 = torch.cat((flags[1:], torch.tensor([False], device=flags.device)))
    flags_2 = torch.clone(flags)
    flags_2[0] = 0.0
    points = torch.nonzero(flags_1 ^ flags_2)
    points = points.view(-1, 2)
    points[:, 0] += 1
    boundary = points.tolist()

    return boundary


def BoundaryCheck(scores, flags, boundary, error_rate):
    """
        input: scores [100],tensor.float
                flags [100],tensor.bool
                boundary list
                error_rate float
    """
    updated_flags = torch.clone(flags)
    for i in range(len(boundary)):
        start = boundary[i][0]
        end = boundary[i][1]
        left_flag = True
        right_flag = True
        while left_flag:
            if start == 0:
                threshold = 0
            else:
                score = scores[start - 1]
                threshold = score / torch.mean(scores[start: end + 1])
            if threshold > error_rate:
                start = start - 1
            else:
                left_flag = False
        while right_flag:
            if end == len(flags) - 1:
                threshold = 0
            else:
                score = scores[end + 1]
                threshold = score / torch.mean(scores[start: end + 1])
            if threshold > error_rate:
                end = end + 1
            else:
                right_flag = False

        # update flags
        updated_flags[start:end + 1] = True
    return updated_flags


def findTAL(logits, video_name, class_name, opt):
    if opt.dataset == 'THUMOS14_I3D':
        if opt.multi_instance:
            temporal_scale = 256
        else:
            temporal_scale = 256
        tal_thres = opt.THUMOS_TAL_threshold
    elif opt.dataset == 'Anomaly_dataset_I3D':
        temporal_scale = 512
        tal_thres = opt.Anomaly_dataset_TAL_threshold
    else:
        temporal_scale = 100
        tal_thres = opt.ActivityNet_TAL_threshold

    n_tasks, shot, num_classes, H, W = logits.size()
    temp_list = []
    for task in range(n_tasks):
        for shot1 in range(shot):
            scores = logits[task][shot1][0].detach().cpu().view(-1)
            thres = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
            for i in thres:
                fg_flag = scores > i
                segments = GetBoundary(fg_flag)
                for segment in segments:
                    start_pt = segment[0]
                    end_pt = segment[1]
                    if (end_pt - start_pt) / temporal_scale > tal_thres:
                        conf_vals = torch.mean(scores[start_pt:end_pt + 1]).item()
                        temp_list.append(
                            [start_pt / temporal_scale, end_pt / temporal_scale, conf_vals, conf_vals,
                             class_name[0]])

    if len(temp_list) == 0:
        temp_list.append([0.0, 0.0, 0.0, 0.0, 'None'])
    new_props = np.stack(temp_list)
    col_name = ["xmin", "xmax", "clr_score", "reg_socre", 'label']
    new_df = pd.DataFrame(new_props, columns=col_name)
    output_directory = os.path.join(opt.output, opt.dataset, "results_shot_{}".format(opt.shot))
    create_dir(output_directory)
    new_df.to_csv(os.path.join(output_directory, "{}.csv".format(video_name[0])), index=False)


def clear_files(data_path):
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    for fls in os.listdir(data_path):
        os.remove(os.path.join(data_path, fls))


def interpolated_prec_rec(prec, rec):
    """Interpolated AP - VOCdevkit from VOC 2011.
    """
    mprec = np.hstack([[0], prec, [0]])
    mrec = np.hstack([[0], rec, [1]])
    for i in range(len(mprec) - 1)[::-1]:
        mprec[i] = max(mprec[i], mprec[i + 1])
    idx = np.where(mrec[1::] != mrec[0:-1])[0] + 1
    ap = np.sum((mrec[idx] - mrec[idx - 1]) * mprec[idx])
    return ap


def get_similarity(query, support, way='cosine'):
    """
        input: query size(i, m, k)
               support size(j, n, k)
               way (cosine, euclidean, manhattan)
        output: similarity matrix (i*j, m, n)
    """

    if way == 'cosine':
        # get size
        i = query.size(0)
        j = support.size(0)
        #  normalize
        query_norm = F.normalize(query, p=2, dim=-1)
        support_norm = F.normalize(support, p=2, dim=-1)
        #  padding
        query_norm = query_norm.unsqueeze(1).expand(-1, support.size(0), -1, -1)  # size(i, j, m, k)
        support_norm = support_norm.unsqueeze(0).expand(query.size(0), -1, -1, -1)  # size(i, j, n, k)
        # calculate similarity
        similarity_matrix = torch.einsum('ijmk,ijnk->ijmn', query_norm, support_norm)
        similarity_matrix = similarity_matrix.view(i, j, query.size(1), support.size(1))  # shot, t, t
        return similarity_matrix
    elif way == 'euclidean':
        # get size
        i = query.size(0)
        j = support.size(0)
        #  broadcast
        query_expanded = query.unsqueeze(2).unsqueeze(3)  # size(i, m, 1, 1, k)
        support_expanded = support.unsqueeze(0).unsqueeze(0)  # size(1, 1, j, n, k)

        # calculate similarity
        similarity_matrix = torch.sum((query_expanded - support_expanded) ** 2, dim=-1)
        similarity_matrix = similarity_matrix.permute(0, 2, 1, 3)
        return similarity_matrix
    elif way == 'manhattan':
        # get size
        i = query.size(0)
        j = support.size(0)
        #  broadcast
        query_expanded = query.unsqueeze(2).unsqueeze(3)  # size(i, m, 1, 1, k)
        support_expanded = support.unsqueeze(0).unsqueeze(0)  # size(1, 1, j, n, k)

        # calculate similarity
        similarity_matrix = torch.sum(torch.abs((query_expanded - support_expanded)), dim=-1)
        similarity_matrix = similarity_matrix.permute(0, 2, 1, 3)

        return similarity_matrix


def IOU(s1, e1, s2, e2):
    if (s2 > e1) or (s1 > e2):
        return 0
    Aor = max(e1, e2) - min(s1, s2)
    Aand = min(e1, e2) - max(s1, s2)
    return float(Aand) / Aor


def segment_iou(target_segment, candidate_segments):
    """Compute the temporal intersection over union between a
    target segment and all the test segments.

    Parameters
    ----------
    target_segment : 1d array
        Temporal target segment containing [starting, ending] times.
    candidate_segments : 2d array
        Temporal candidate segments containing N x [starting, ending] times.

    Outputs
    -------
    tiou : 1d array
        Temporal intersection over union score of the N's candidate segments.
    """
    tt1 = np.maximum(target_segment[0], candidate_segments[:, 0])
    tt2 = np.minimum(target_segment[1], candidate_segments[:, 1])
    # Intersection including Non-negative overlap score.
    segments_intersection = (tt2 - tt1).clip(0)
    # Segment union.
    segments_union = (candidate_segments[:, 1] - candidate_segments[:, 0]) \
                     + (target_segment[1] - target_segment[0]) - segments_intersection
    # Compute overlap as the ratio of the intersection
    # over union of two segments.
    tIoU = segments_intersection.astype(float) / segments_union
    return tIoU


def pad_sequence(data, max_t=None):
    # data is a tensor list
    if max_t is None:
        max_t = max(tensor.shape[-2] for tensor in data)
    if all(tensor.shape[-2] == max_t for tensor in data):
        return torch.stack(data)
    padded_data = []
    for tensor in data:
        padded_tensor = F.pad(tensor, (0, 0, 0, max_t - tensor.shape[-2]),
                              value=0.0)  # (pad_left, pad_right, pad_top, pad_bottom)
        padded_data.append(padded_tensor)
    combined_tensor = torch.stack(padded_data)
    return combined_tensor


def reshape_batch(batch, mode):
    categories = list(batch.keys())
    reshaped_batch = []
    if mode == 'query':
        category = random.sample(categories, 1)
        reshaped_batch = batch[category[0]][0].unsqueeze(0).unsqueeze(0)
        mask = (reshaped_batch != 0.0).to(torch.float32)
        return reshaped_batch, category, mask
    for category, data in batch.items():
        reshaped_batch.append(pad_sequence(data))
    reshaped_batch = pad_sequence(reshaped_batch)
    c, k, b, t, d = reshaped_batch.size()
    reshaped_batch = reshaped_batch.view(b, c, k, t, d)
    mask = (reshaped_batch != 0.0).to(torch.float32)
    return reshaped_batch, categories, mask


def get_file_paths(directory):
    file_dict = {}
    for root, dirs, files in os.walk(directory):
        for filename in files:
            absolute_path = os.path.abspath(os.path.join(root, filename))
            file_dict[filename] = absolute_path
    return file_dict
