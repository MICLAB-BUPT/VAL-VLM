import argparse
from sre_parse import parse

from traitlets import default


def str2bool(str):
    return True if str.lower() == 'true' else False


def parse_opt():
    parser = argparse.ArgumentParser()

    # overall setting
    parser.add_argument('--gpu', type=str, default=0)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--use_seed', type=str2bool, default=False)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--multi_instance', type=str2bool, default=True)
    parser.add_argument('--shot', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--train_episodes', type=int, default=100)
    parser.add_argument('--validation_episodes', type=int, default=100)
    parser.add_argument('--test_episodes', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.00001)
    parser.add_argument('--iteration', type=int, default=1)
    parser.add_argument('--step_size', type=int, default=20)
    parser.add_argument('--weight_decay', type=float, default=0.1)

    # evaluation setting
    parser.add_argument('--filter_thres', type=float, default=0.003)
    parser.add_argument('--smooth', type=str2bool, default=True)
    parser.add_argument('--gaussian_kernel', type=int, default=5)
    parser.add_argument('--soft_NMS_threshold', type=float, default=0.8)
    parser.add_argument('--top_k', type=int, default=100)
    parser.add_argument('--output', type=str, default='output')
    parser.add_argument('--tensorboard_dir', type=str, default='logs')


    # dataset setting
    parser.add_argument('--dataset', type=str, default='ActivityNet1.3')


    # ActivityNet1.3 setting
    parser.add_argument('--ActivityNet1_3_temporal_scale', type=int, default=100)
    parser.add_argument('--ActivityNet_TAL_threshold', type=float, default=0.15)
    parser.add_argument('--ActivityNet1_3_video_features_path', type=str, default='data/anet_1.3/csv_mean_100.hdf5')
    parser.add_argument('--ActivityNet1_3_text_features_path', type=str,
                        default='data/anet_1.3/captions.hdf5')
    parser.add_argument("--ActivityNet1_3_long_text_features_path", type=str,
                        default='data/anet_1.3/long_text.hdf5')
    parser.add_argument('--ActivityNet1_3_annotations_path', type=str,
                        default='data/anet_1.3/annotations/anet_anno_action.json')


    # CLIP
    parser.add_argument('--CLIP_dim', type=int, default=512)


    # stpe setting
    parser.add_argument('--d_model', type=int, default=400)
    parser.add_argument('--d_k', type=int, default=128)
    parser.add_argument('--d_v', type=int, default=128)
    parser.add_argument('--n_head', type=int, default=4)
    parser.add_argument('--n_layer', type=int, default=1)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--truncate', type=str2bool, default=False)
    parser.add_argument('--window_size', type=str, default=[3, 3, 3])  # # The number of children of a parent node.
    parser.add_argument('--inner_size', type=int, default=3)  # The number of adjacent nodes.
    parser.add_argument('--d_bottleneck', type=int, default=100)  # actually->seq_len
    parser.add_argument('--input_size', type=int, default=100)
    parser.add_argument('--d_inner', type=int, default=512)
    parser.add_argument('--MLP_hidden', type=int, default=800)
    parser.add_argument('--semantic_node', type=int, default=6)
    parser.add_argument('--level', type=int, default=1)
    parser.add_argument('--output_size', type=int, default=256)

    return parser.parse_args()
