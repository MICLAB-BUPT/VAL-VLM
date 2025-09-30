import json
import os
import pdb
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from utils import load_json, split_dataset, get_file_names, interpolated_prec_rec, IOU, segment_iou


class Detection(object):
    def __init__(self, dataset=None,
                 ground_truth_filename=None,
                 prediction_filename=None,
                 output_path=None,
                 tiou_thresholds=np.linspace(0.5, 0.95, 10),
                 subset="validation",
                 verbose=False):
        self.dataset = dataset
        self.ground_truth_filename = ground_truth_filename
        self.prediction_filename = prediction_filename
        self.output_path = output_path
        self.tiou_thresholds = tiou_thresholds
        self.subset = subset
        self.classes_list = split_dataset(self.dataset, self.subset)
        self.ap = None
        self.verbose = verbose
        self.ground_truth, self.activity_dict = self._import_ground_truth(self.ground_truth_filename)
        self.prediction = self._import_prediction(self.prediction_filename)

        if self.verbose:
            print('[INIT] Loaded annotations from {} subset.'.format(subset))
            print('\tNumber of ground truth instances: {}'.format(len(self.ground_truth)))
            print('\tNumber of predictions: {}'.format(len(self.prediction)))
            print('\tFixed threshold for tiou score: {}'.format(self.tiou_thresholds))

    def _import_ground_truth(self, ground_truth_filename):
        """Reads ground truth file, checks if it is well formatted, and returns
                   the ground truth instances and the activity classes.

        Parameters
        ----------
        ground_truth_filename : str
            Full path to the ground truth json file.

        Outputs
        -------
        ground_truth : df
            Data frame containing the ground truth instances.
        activity_index : dict
            Dictionary containing class index.
        """
        # Read ground truth json file.
        data = load_json(ground_truth_filename)
        activity_dict, class_idx = {}, 0
        video_lst, t_start_lst, t_end_lst, label_lst = [], [], [], []
        for video_name in get_file_names(os.path.join(self.output_path)):
            video_info = data[video_name]
            for annotation in video_info['annotations']:
                if annotation['label'] in self.classes_list:
                    video_lst.append(video_name)
                    t_start_lst.append(float(annotation['segment'][0]))
                    t_end_lst.append(float(annotation['segment'][1]))
                    label_lst.append(annotation['label'])
                    if annotation['label'] not in activity_dict:
                        activity_dict[annotation['label']] = class_idx
                        class_idx += 1
        ground_truth = pd.DataFrame({'video-id': video_lst,
                                     't-start': t_start_lst,
                                     't-end': t_end_lst,
                                     'label': label_lst})
        if self.verbose:
            print(activity_dict)
        return ground_truth, activity_dict

    def _import_prediction(self, prediction_filename):
        """Reads prediction file, checks if it is well formatted, and returns
           the prediction instances.

        Parameters
        ----------
        prediction_filename : str
            Full path to the prediction json file.

        Outputs
        -------
        prediction : df
            Data frame containing the prediction instances.
        """
        # load prediction json file
        data = load_json(prediction_filename)

        # Read predictions.
        video_lst, t_start_lst, t_end_lst = [], [], []
        label_lst, score_lst = [], []
        for video_name, video_predictions in data['results'].items():
            for prediction in video_predictions:
                label = prediction['label']
                video_lst.append(video_name)
                t_start_lst.append(float(prediction['segment'][0]))
                t_end_lst.append(float(prediction['segment'][1]))
                label_lst.append(label)
                score_lst.append(prediction['score'])
        pred = pd.DataFrame({'video-id': video_lst,
                             't-start': t_start_lst,
                             't-end': t_end_lst,
                             'label': label_lst,
                             'score': score_lst})
        return pred

    def _get_predictions_with_label(self, prediction_by_label, label_name):
        """Get all predictions of the given label. Return empty DataFrame if there
        is no predictions with the given label.
        """
        try:
            return prediction_by_label.get_group(label_name).reset_index(drop=True)
        except:
            if self.verbose:
                print('Warning: No predictions of label \'%s\' were provided.' % label_name)
            return pd.DataFrame()

    def _get_ground_truth_with_label(self, ground_truth_by_label, label_name):
        """Get all ground truth of the given label. Return empty DataFrame if there
         is no ground truth with the given label.
        """
        try:
            return ground_truth_by_label.get_group(label_name).reset_index(drop=True)
        except:
            if self.verbose:
                print('Warning: No ground truth of label \'%s\' were provided.' % label_name)
            return pd.DataFrame()

    def wrapper_compute_average_precision(self):
        """Computes average precision for each class in the subset.
        """
        ap = np.zeros((len(self.tiou_thresholds), len(self.activity_dict)))

        # Adaptation to query faster
        ground_truth_by_label = self.ground_truth.groupby('label')
        prediction_by_label = self.prediction.groupby('label')

        results = Parallel(n_jobs=len(self.activity_dict))(
            delayed(self.compute_average_precision_detection)(
                ground_truth=self._get_ground_truth_with_label(ground_truth_by_label, label_name),
                prediction=self._get_predictions_with_label(prediction_by_label, label_name),
                tiou_thresholds=self.tiou_thresholds,
            ) for label_name in self.activity_dict.keys())

        for i, cidx in enumerate(self.activity_dict.values()):
            ap[:, cidx] = results[i]

        return ap

    def evaluate(self):
        """Evaluates a prediction file. For the detection task we measure the
        interpolated mean average precision to measure the performance of a
        method.
        """
        self.ap = self.wrapper_compute_average_precision()
        self.mAP = self.ap.mean(axis=1)
        self.average_mAP = self.mAP.mean()

        if self.verbose:
            print('[RESULTS] Performance on ' + str(self.dataset) + 'detection task.')
            print('Average-mAP: {}'.format(self.average_mAP))

        return self.mAP, self.average_mAP

    def compute_average_precision_detection(self, ground_truth, prediction, tiou_thresholds=np.linspace(0.5, 0.95, 10)):
        """Compute average precision (detection task) between ground truth and
        predictions data frames. If multiple predictions occurs for the same
        predicted segment, only the one with the highest score is matches as
        true positive. This code is greatly inspired by Pascal VOC devkit.

        Parameters
        ----------
        ground_truth : df
            Data frame containing the ground truth instances.
            Required fields: ['video-id', 't-start', 't-end']
        prediction : df
            Data frame containing the prediction instances.
            Required fields: ['video-id, 't-start', 't-end', 'score']
        tiou_thresholds : 1darray, optional
            Temporal intersection over union threshold.

        Outputs
        -------
        ap : float
            Average precision score.
        """
        ap = np.zeros(len(tiou_thresholds))
        if prediction.empty:
            return ap

        npos = float(len(ground_truth))
        lock_gt = np.ones((len(tiou_thresholds), len(ground_truth))) * -1
        # Sort predictions by decreasing score order.
        sort_idx = prediction['score'].values.argsort()[::-1]
        prediction = prediction.loc[sort_idx].reset_index(drop=True)

        # Initialize true positive and false positive vectors.
        tp = np.zeros((len(tiou_thresholds), len(prediction)))
        fp = np.zeros((len(tiou_thresholds), len(prediction)))

        # Adaptation to query faster
        ground_truth_gbvn = ground_truth.groupby('video-id')

        # Assigning true positive to truly grount truth instances.
        for idx, this_pred in prediction.iterrows():

            try:
                # Check if there is at least one ground truth in the video associated.
                ground_truth_videoid = ground_truth_gbvn.get_group(this_pred['video-id'])
            except Exception as e:
                fp[:, idx] = 1
                continue

            this_gt = ground_truth_videoid.reset_index()
            tiou_arr = segment_iou(this_pred[['t-start', 't-end']].values,
                                   this_gt[['t-start', 't-end']].values)
            # We would like to retrieve the predictions with highest tiou score.
            tiou_sorted_idx = tiou_arr.argsort()[::-1]
            for tidx, tiou_thr in enumerate(tiou_thresholds):
                for jdx in tiou_sorted_idx:
                    if tiou_arr[jdx] < tiou_thr:
                        fp[tidx, idx] = 1
                        break
                    if lock_gt[tidx, this_gt.loc[jdx]['index']] >= 0:
                        continue
                    # Assign as true positive after the filters above.
                    tp[tidx, idx] = 1
                    lock_gt[tidx, this_gt.loc[jdx]['index']] = idx
                    break

                if fp[tidx, idx] == 0 and tp[tidx, idx] == 0:
                    fp[tidx, idx] = 1

        tp_cumsum = np.cumsum(tp, axis=1).astype(float)
        fp_cumsum = np.cumsum(fp, axis=1).astype(float)
        recall_cumsum = tp_cumsum / npos

        precision_cumsum = tp_cumsum / (tp_cumsum + fp_cumsum)

        for tidx in range(len(tiou_thresholds)):
            ap[tidx] = interpolated_prec_rec(precision_cumsum[tidx, :], recall_cumsum[tidx, :])

        return ap


def get_infer_dict(opt):
    dataset = opt.dataset
    ground_truth_filename = None
    if dataset == 'THUMOS14_I3D':
        if opt.multi_instance:
            ground_truth_filename = opt.THUMOS14_I3D_annotations_path
        else:
            ground_truth_filename = opt.THUMOS14_I3D_annotations_split_path
    elif dataset == 'ActivityNet1.3':
        ground_truth_filename = opt.ActivityNet1_3_annotations_path


    database = load_json(ground_truth_filename)
    video_dict = {}
    file_names = get_file_names(opt.output + '/' + dataset + '/results_shot_{}'.format(opt.shot), split=True)
    for video_name in file_names:
        video_dict[video_name] = database[video_name]
    return video_dict


def Soft_NMS(df, nms_threshold=0.8, num_prop=200):
    '''
    From BSN code
    :param num_prop:
    :param df:
    :param nms_threshold:
    :return:
    '''
    df = df.sort_values(by="score", ascending=False)

    tstart = list(df.xmin.values[:])
    tend = list(df.xmax.values[:])
    tscore = list(df.score.values[:])
    tlabel = list(df.label.values[:])

    rstart = []
    rend = []
    rscore = []
    rlabel = []

    while len(tscore) > 1 and len(rscore) < num_prop and max(tscore) > 0:
        max_index = tscore.index(max(tscore))
        for idx in range(0, len(tscore)):
            if idx != max_index:
                tmp_iou = IOU(tstart[max_index], tend[max_index], tstart[idx], tend[idx])
                if tmp_iou > 0:
                    tscore[idx] = tscore[idx] * np.exp(-np.square(tmp_iou) / nms_threshold)

        rstart.append(tstart[max_index])
        rend.append(tend[max_index])
        rscore.append(tscore[max_index])
        rlabel.append(tlabel[max_index])
        tstart.pop(max_index)
        tend.pop(max_index)
        tscore.pop(max_index)
        tlabel.pop(max_index)

    newDf = pd.DataFrame()
    newDf['score'] = rscore
    newDf['xmin'] = rstart
    newDf['xmax'] = rend
    newDf['label'] = rlabel
    return newDf


def gen_detection_video(video_name, video_info, opt):
    proposal_list = {}
    if os.path.exists(os.path.join(opt.output, opt.dataset + '/results_shot_{}/'.format(opt.shot) + video_name + ".csv")):
        df = pd.read_csv(os.path.join(opt.output, opt.dataset + '/results_shot_{}/'.format(opt.shot) + video_name + ".csv"))
        df['score'] = df.clr_score.values[:]
        label = df.label.values[0]
        # soft NMS
        if len(df) > 1:
            df = Soft_NMS(df, opt.soft_NMS_threshold)
        df = df.sort_values(by="score", ascending=False)
        try:
            video_duration = float(video_info["duration_frame"]) / video_info["duration_frame"] * video_info[
                "duration_second"]
        except:
            try:
                video_duration = video_info["duration"]
            except:
                video_duration = video_info["total_frame"]
        proposal_list = []
        for i in range(min(opt.top_k, len(df))):
            tmp_proposal = {}
            tmp_proposal["label"] = label
            tmp_proposal["score"] = float(df.score.values[i] * 0.95)
            tmp_proposal["segment"] = [max(0, df.xmin.values[i]) * video_duration,
                                       min(1, df.xmax.values[i]) * video_duration]
            proposal_list.append(tmp_proposal)


    else:
        print(f'{video_name}.csv not exist')
    return {video_name: proposal_list}


def gen_detection_multicore(opt):
    # get video duration
    infer_dict = get_infer_dict(opt)
    parallel = Parallel(n_jobs=15, prefer="processes")
    detection = parallel(delayed(gen_detection_video)(video_name, video_info, opt)
                         for video_name, video_info in infer_dict.items())
    detection_dict = {}
    [detection_dict.update(d) for d in detection]
    output_dict = {"version": opt.dataset, "results": detection_dict, "external_data": {}}

    with open(opt.output + '/' + opt.dataset + '/detection_result_shot_{}.json'.format(opt.shot),
              "w") as out:
        json.dump(output_dict, out, indent=4)


def evaluate(opt, mode='validation', verbose=True):
    if verbose:
        print("Detection post processing start")
    gen_detection_multicore(opt)
    if verbose:
        print("Detection Post processing finished")
    dataset = opt.dataset
    gt_file = None
    if dataset == 'THUMOS14_I3D':
        if opt.multi_instance:
            gt_file = opt.THUMOS14_I3D_annotations_path
        else:
            gt_file = opt.THUMOS14_I3D_annotations_split_path
    elif dataset == 'ActivityNet1.3':
        gt_file = opt.ActivityNet1_3_annotations_path
    detection = Detection(
        dataset=opt.dataset,
        ground_truth_filename=gt_file,
        prediction_filename=(
                opt.output + '/' + opt.dataset + '/detection_result_shot_{}.json'.format(opt.shot)),
        output_path=(opt.output + '/' + opt.dataset + "/results_shot_{}".format(opt.shot)),
        subset=mode,
        verbose=True
    )
    detection.evaluate()

    mAP_at_tIoU = [f'mAP@{t:.2f} {mAP * 100:.3f}' for t, mAP in zip(detection.tiou_thresholds, detection.mAP)]
    results = f'Detection: average-mAP {detection.average_mAP * 100:.3f} {" ".join(mAP_at_tIoU)}'
    if verbose:
        print(results)
    with open(os.path.join(opt.output, opt.dataset + '/results.txt'), 'a') as fobj:
        fobj.write(f'{results}\n')

    return detection.average_mAP * 100, detection.mAP[0] * 100
