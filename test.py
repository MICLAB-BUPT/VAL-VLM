from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm

from dataset import *
from evaluation import evaluate
from model.Classsifier.classifierV2 import Classifier
from utils import *


def test_activity(opt):

    writer = SummaryWriter(log_dir=opt.tensorboard_dir)
    classifier_check_point = torch.load('ckpt/ActivityNet/' + f'best_classifier_shot{opt.shot}.pth.tar', weights_only=True)
    classifier = Classifier(opt)
    classifier = classifier.cuda()
    classifier.load_state_dict(classifier_check_point)
    dataset = ActivityNet(opt, mode='test')

    clear_files(os.path.join(opt.output, opt.dataset, "results_shot_{}".format(opt.shot)))

    test_dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)
    iter_test_loader = iter(test_dataloader)

    for episode in tqdm(range(opt.test_episodes)):
        try:
            query_data, query_label, query_video_name, support_data, support_label,support_video_name, chosen_class = next(
                iter_test_loader)
        except:
            iter_test_loder = iter(test_dataloader)
            query_data, query_label, query_video_name, support_data, support_label,support_video_name, chosen_class = next(
                iter_test_loder)
        query_video = query_data[0].cuda()
        query_text = query_data[1].cuda()
        query_long_text = query_data[2].cuda()
        query_label = query_label.cuda()
        support_video = support_data[0].cuda()
        support_text = support_data[1].cuda()
        support_long_text = support_data[2].cuda()
        support_label = support_label.cuda()

        support_video = support_video.detach()  # batch_size, shot, temporal_length, feature_dimension
        support_text = support_text.detach()
        support_long_text = support_long_text.detach()
        query_video = query_video.detach()
        query_text = query_text.detach()
        query_long_text = query_long_text.detach()

        # #
        classifier.eval()
        with torch.no_grad():
            proba_q = classifier(support_video, support_text, support_long_text, support_label, query_video, query_text,
                                 query_long_text, query_label, smooth=True)
        findTAL(proba_q, query_video_name, chosen_class, opt)
    mAP, mAP_0_5 = evaluate(opt, mode='test', verbose=True)




