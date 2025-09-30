from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm
from dataset import ActivityNet
from model.Classsifier.classifierV2 import Classifier
from evaluation import evaluate
from utils import *


def train_anet(opt):

    print(f"shot:{opt.shot} batch_size: {opt.batch_size} gpu:{opt.gpu} multi_instance:{opt.multi_instance}")

    writer = SummaryWriter(log_dir='logs/')
    device = torch.device(opt.device)
    classifier = Classifier(opt)
    classifier.to(device)
    # dataloader
    train_dataloader = DataLoader(dataset=ActivityNet(opt, mode="train"), batch_size=opt.batch_size, shuffle=True,
                                  num_workers=8)
    validation_dataloader = DataLoader(dataset=ActivityNet(opt, mode="validation"), batch_size=1, shuffle=True,
                                       num_workers=0)

    iter_train_loader = iter(train_dataloader)
    iter_validation_loader = iter(validation_dataloader)

    best_mAP = 0.0
    lr = opt.lr
    iteration = opt.iteration

    for epoch in tqdm(range(opt.epochs)):
        classifier.train()
        for episode in tqdm(range(opt.train_episodes)):
            try:
                query_data, query_label, query_video_name, support_data, support_label, support_video_name, chosen_class = next(
                    iter_train_loader)
            except:
                iter_train_loader = iter(train_dataloader)
                query_data, query_label, query_video_name, support_data, support_label, support_video_name, chosen_class = next(
                    iter_train_loader)

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

            train_loss = classifier.TrainEAT(support_video, support_text, support_long_text, support_label, query_video,
                                             query_text, query_long_text,
                                             query_label, lr=lr, iteration=iteration, writer=writer, episode=episode)
        torch.save(classifier.state_dict(), f'./ckpt/ActivityNet/checkpoint_classifier_shot{opt.shot}.pth.tar')

        if epoch < 10:  # warmup
            continue

        # validation
        clear_files(os.path.join(opt.output, opt.dataset, "results_shot_{}".format(opt.shot)))

        classifier.load_state_dict(
            torch.load(f'./ckpt/ActivityNet/checkpoint_classifier_shot{opt.shot}.pth.tar', weights_only=True))
        classifier.eval()
        set_seed(opt.seed)
        for episode in tqdm(range(opt.validation_episodes)):
            loss_avg = AverageMeter()
            try:
                query_data, query_label, query_video_name, support_data, support_label, support_video_name, chosen_class = next(
                    iter_validation_loader)
            except:
                iter_validation_loader = iter(validation_dataloader)
                query_data, query_label, query_video_name, support_data, support_label, support_video_name, chosen_class = next(
                    iter_validation_loader)
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


            with torch.no_grad():
                proba_q = classifier(support_video, support_text, support_long_text, support_label, query_video,
                                     query_text,
                                     query_long_text, query_label, smooth=True)

            findTAL(proba_q, query_video_name, chosen_class, opt)

        mAP, mAP_0_5 = evaluate(opt, mode='validation', verbose=True)
        if mAP_0_5 > best_mAP:
            best_mAP = mAP_0_5
            torch.save(classifier.state_dict(), f'./ckpt/ActivityNet/best_classifier_shot{opt.shot}.pth.tar')
    return







if __name__ == '__main__':
    import option

    opt = option.parse_opt()
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_anet(opt)
