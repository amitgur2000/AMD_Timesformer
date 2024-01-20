import argparse
from itertools import cycle
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import accuracy_score, roc_auc_score, precision_recall_fscore_support, classification_report, confusion_matrix
from sklearn.metrics import roc_curve, auc
from scipy.special import softmax
import pickle as pkl
import torch
from torchvision import transforms
import sys
from fvcore.common.config import CfgNode
# from slowfast.models import build_model
# import slowfast.utils.checkpoint as cu
from timesformer.datasets import decoder as decoder
from timesformer.datasets import video_container as container
from timesformer.datasets import utils as utils
from timesformer.models.vit import TimeSformer
from timesformer.models.build import build_model              # timesformer/models/build.py
from timesformer.utils.parser import load_config, parse_args
import timesformer.utils.metrics as metrics

from timm.data import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
import timesformer.models.losses as losses

from pdb import set_trace as bp
import random

seed=10
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.use_deterministic_algorithms(True)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

SUPPORTED_MODEL_TYPES = ['tf', 'mf']

def config_from_args(args, num_classes=2, temporal_resolution=8, num_gpus=0):
    cfg = CfgNode()
    cfg.NUM_GPUS = num_gpus

    cfg.DATA = CfgNode()
    cfg.DATA.TRAIN_CROP_SIZE = args.spatial_scale
    cfg.DATA.NUM_FRAMES = args.num_frames

    cfg.TRAIN = CfgNode()
    cfg.TRAIN.ENABLE = False
    cfg.TRAIN.DATASET = 'kinetics'

    cfg.TEST = CfgNode()
    cfg.TEST.ENABLE = True
    cfg.TEST.CHECKPOINT_FILE_PATH = args.model_path
    cfg.TEST.CHECKPOINT_TYPE = 'pytorch'

    cfg.MODEL = CfgNode()
    cfg.MODEL.MODEL_NAME = 'VisionTransformer'
    cfg.MODEL.NUM_CLASSES = num_classes

    cfg.VIT = CfgNode()
    cfg.VIT.IM_PRETRAINED = True
    cfg.VIT.PRETRAINED_WEIGHTS = 'vit_1k'
    cfg.VIT.PATCH_SIZE = 16
    cfg.VIT.NUM_HEADS = 12
    cfg.VIT.CHANNELS = 3
    cfg.VIT.EMBED_DIM = 768
    cfg.VIT.DEPTH = 12
    cfg.VIT.MLP_RATIO = 4
    cfg.VIT.QKV_BIAS = True
    cfg.VIT.DROP = 0.0
    cfg.VIT.DROP_PATH = 0.2
    cfg.VIT.HEAD_DROPOUT = 0.0
    cfg.VIT.VIDEO_INPUT = True
    cfg.VIT.TEMPORAL_RESOLUTION = 16 #temporal_resolution 
    cfg.VIT.USE_MLP = True
    cfg.VIT.ATTN_DROPOUT = 0.0
    cfg.VIT.HEAD_ACT = 'tanh'
    cfg.VIT.PATCH_SIZE_TEMP = 2
    cfg.VIT.POS_DROPOUT = 0.0
    cfg.VIT.POS_EMBED = 'separate'
    cfg.VIT.ATTN_LAYER = 'trajectory'
    cfg.VIT.USE_ORIGINAL_TRAJ_ATTN_CODE = True
    cfg.VIT.APPROX_ATTN_TYPE = 'none'
    cfg.VIT.APPROX_ATTN_DIM = 128
    return cfg


def get_frames(video_path, sampling_rate, num_frames, num_ensemble_views, temporal_sampling_index, spatial_scale, same_frame=False, start_pts=None, end_pts=None, is_vid=True):
    decoding_backend = 'pyav'
    target_fps = 30
    MEAN = [0.45, 0.45, 0.45]
    STD = [0.225, 0.225, 0.225]
    RANDOM_FLIP = False # True
    INV_UNIFORM_SAMPLE = False
    crop_size = spatial_scale
    min_spatial_scale = spatial_scale
    max_spatial_scale = spatial_scale
    spatial_sample_index = -1 # center/ middle. TODO: test time augmentation?
    
    if is_vid:
        try:
            video_container = container.get_video_container(
                    video_path,
                    True,
                    decoding_backend,
                )
        except Exception as e:
            print('Failed to load video from {} with error {}'.format(video_path, e))
            return []
        frames = decoder.decode(
                    video_container,
                    sampling_rate,
                    num_frames,
                    temporal_sampling_index,
                    num_ensemble_views,
                    video_meta={},
                    target_fps=target_fps,
                    backend=decoding_backend,
                    max_spatial_scale=min_spatial_scale,
                    start=start_pts,
                    end=end_pts
                )
    else:
        img = Image.open(video_path).resize((spatial_scale, spatial_scale))
        frames = transforms.ToTensor()(img).unsqueeze_(0)
        frames = frames.permute(0, 2, 3, 1)
    # Perform color normalization.
    if frames is None:
        return []
    frames = utils.tensor_normalize(
        frames, MEAN, STD
    )
    # T H W C -> C T H W.
    frames = frames.permute(3, 0, 1, 2)
    if is_vid:
        frames = utils.spatial_sampling(
            frames,
            spatial_idx=spatial_sample_index,
            min_scale=min_spatial_scale,
            max_scale=max_spatial_scale,
            crop_size=crop_size,
            random_horizontal_flip=RANDOM_FLIP,
            inverse_uniform_sampling=INV_UNIFORM_SAMPLE,
        )
    # Perform temporal sampling from the fast pathway
    frames = torch.index_select(
        frames,
        1,
        torch.linspace(
            0, frames.shape[1] - 1, num_frames
        ).long(),
    )
    frames = frames[None, :]
    if same_frame:
        for i in range(8):
            frames[:, :, i, :, :] = frames[:,:,0,:,:]
    return frames

def score(model, video_path, args, cfg, true_label, start_pts=None, end_pts=None, same_frame=False):
    score_by_avg = False # score by average or by majority voting
    if args.head_mask_hl is not None:
        head_mask = torch.ones((model.model.depth, model.model.depth))
        if 'head' in head_mask_hl:
            _, h = head_mask_hl.split('-')
            for l in range(model.model.depth):
                head_mask[l, int(h)] = 0
        elif 'lay' in head_mask_hl:
            _, l = head_mask_hl.split('-')
            for h in range(model.model.depth):
                head_mask[int(l), h] = 0
        elif '-' in head_mask_hl:
            h, l = head_mask_hl.split('-')
            head_mask[int(h), int(l)] = 0
    else:
        head_mask = None
    all_preds = []
    all_temporal = []
    all_spatial = []
    all_labels = []
   
    for ensemble_index in range(args.num_ensemble_views):
        frames = get_frames(video_path,
                            args.sampling_rate,
                            args.num_frames,
                            args.num_ensemble_views,
                            ensemble_index,
                            args.spatial_scale,
                            same_frame=same_frame, start_pts=start_pts, end_pts=end_pts)
        if frames == []: # Failed to decode from some reason
            continue
        
        if args.model_type == 'tf':
            pred, temporal_attentions, spatial_attentions, hidden_states = model(frames, output_attentions=args.output_attentions, head_mask=head_mask)
            print(pred)
        else:
            if torch.cuda.is_available():
                frames.to('cuda')
            pred = model([frames])
            temporal_attentions = None
            spatial_attentions = None
            hidden_states = None

        #assert pred.shape[1] == 2
        preds_np = pred.detach().numpy()
        if args.model_type == 'tf':
            if cfg.MODEL.CLASSIFY_TYPE == "ordinal_regress":
                top1_err, top5_err, preds_2_labels = metrics.calc_top_1_top_5(cfg, pred, torch.tensor(int(true_label)))  # AG: in case of ordinal regression, translate pred to label
                preds_np=preds_2_labels.detach().numpy() 
            else:
                preds_np = softmax(preds_np)
                
        all_preds.append(preds_np)
        all_temporal.append(temporal_attentions)
        all_spatial.append(spatial_attentions)
        #sm = softmax(preds_np)
        if cfg.MODEL.CLASSIFY_TYPE == "ordinal_regress":        
            pred_label = preds_np
        else:
            pred_label = np.argmax(preds_np)
        all_labels.append(pred_label)

    if all_preds == []:
        return None, None, None, None, None
    if score_by_avg:
        pred_label = np.average(all_labels)
    else: # majority voting
        vals, cnt = np.unique(all_labels, return_counts=True)
        pred_label = np.argmax(cnt) if len(cnt) > 1 else vals[0]

    preds_np = np.average(all_preds, axis=0)
    if args.debug:
        print('vals: {}, cnt: {}'.format(vals, cnt)) 
        print('all preds: {}\npred label: {} true label: {}'.format(all_preds, pred_label, true_label))

    #TODO: attentions? currently returning the last.
    return pred_label, preds_np, temporal_attentions, spatial_attentions, hidden_states

def get_loss_func(cfg):
    loss_fun = losses.get_loss_func(cfg.MODEL.LOSS_FUNC)(reduction="mean")
    return loss_fun

def compute_heads_importance(
        model, labels_df, args, cfg, same_frame=False, compute_importance=True, head_mask=None, actually_pruned=False
):
    """This method shows how to compute:
    - head attention entropy
    - head importance scores according to http://arxiv.org/abs/1905.10650
    """
    # Explicitly declare reduction to mean.
    loss_fun = get_loss_func(cfg)
    # Prepare our tensors
    n_layers, n_heads = model.model.depth, model.model.depth
    head_importance = torch.zeros(n_layers, n_heads)#.to(args.device)

    if head_mask is None:
        head_mask = torch.ones(n_layers, n_heads)#.to(args.device)

    head_mask.requires_grad_(requires_grad=True)
    # If actually pruned attention multi-head, set head mask to None to avoid shape mismatch
    # if actually_pruned:
    #     head_mask = None

    # preds = None
    # # labels = None
    # tot_tokens = 0.0
    for line in tqdm(labels_df.iterrows(), total=len(labels_df)):
        vid_path = line[1][0] #   AG fixed #  args.dataset_path + '/' + line[1]['Video'] + args.video_ext
        print(vid_path)
        if not os.path.exists(vid_path):
            print('Can\'t find video {}, skipping'.format(vid_path))
            continue
        frames = get_frames(vid_path, args.sampling_rate, args.num_frames, args.num_ensemble_views, 1, args.spatial_scale, same_frame=same_frame)  # AG TODO support more than one temporal view                                 
        if frames is None:
            print('Skipping {}, cannot extract frames'.format(vid_path))
            continue
        pred, temporal_attentions, spatial_attentions, hidden_states = model(frames,
                                                                             output_attentions=False,
                                                                             head_mask=head_mask)
        
        label = int(line[1][1])
        label = torch.tensor([label])
        # Do a forward pass (not with torch.no_grad() since we need gradients for importance score - see below)
        
        if cfg.MODEL.CLASSIFY_TYPE == "ordinal_regress":
            modified_labels = torch.zeros_like(pred)
            for i, lab in enumerate(label):
                modified_labels[i, 0:int(lab)+1] = 1
            loss = loss_fun(pred, modified_labels)
        else:
            loss = loss_fun(pred, label)
        # Loss and logits are the first, attention the last
        loss.backward()  # Backpropagate to populate the gradients in the head mask

        # if compute_entropy:
        #     for layer, attn in enumerate(spatial_attentions):
        #         masked_entropy = entropy(attn.detach()) * inputs["attention_mask"].float().unsqueeze(1)
        #         attn_entropy[layer] += masked_entropy.sum(-1).sum(0).detach()

        if compute_importance:
            head_importance += head_mask.grad.abs().detach()

        # # Also store our logits/labels if we want to compute metrics afterwards
        # if preds is None:
        #     preds = logits.detach().cpu().numpy()
        #     labels = inputs["labels"].detach().cpu().numpy()
        # else:
        #     preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
        #     labels = np.append(labels, inputs["labels"].detach().cpu().numpy(), axis=0)
        #
        # tot_tokens += inputs["attention_mask"].float().detach().sum().data

    # Normalize
    # attn_entropy /= tot_tokens
    # head_importance /= tot_tokens

    # Print/save matrices
    # np.save(os.path.join(args.output_dir, "attn_entropy.npy"), attn_entropy.detach().cpu().numpy())
    # np.save(os.path.join(args.output_dir, "head_importance.npy"), head_importance.detach().cpu().numpy())

    # logger.info("Attention entropies")
    # print_2d_tensor(attn_entropy)
    # print_2d_tensor(head_importance)
    # logger.info("Head ranked by importance scores")
    head_ranks = torch.zeros(head_importance.numel(), dtype=torch.long)#, device=args.device)
    head_ranks[head_importance.view(-1).sort(descending=True)[1]] = torch.arange(
        head_importance.numel())#, device=args.device
 #   )
    head_ranks = head_ranks.view_as(head_importance)
    # print_2d_tensor(head_ranks)
   # print((head_importance == torch.max(head_importance)).nonzero())
    return head_ranks, head_importance#, preds, labels


def compute_metrics(df, gt_name, pred_name, score_name, split_name='all'):
    y_true = df[gt_name]
    pred_labels = df[pred_name]
    pred_score_full = df[score_name]
    pred_score_array = np.array([p[0] for p in pred_score_full])
    accuracy = accuracy_score(y_true, pred_labels)
    #  scores_positive = [pred_score[0][1] for pred_score in pred_score_full]  # AG TODO bug
    # roc_auc = roc_auc_score(y_true, scores_positive)
    all_classes_roc_auc = roc_auc_score(y_true, pred_score_array, multi_class='ovr')
    print("AUC OvR: ", all_classes_roc_auc)
    # prfs = precision_recall_fscore_support(y_true, pred_labels, average=None, labels=[0,1])
    cls_report = classification_report(y_true, pred_labels, labels=[0, 1])
    print('Results for: {}'.format(split_name))
    print('---------------------------------------')
    print(cls_report)

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    n_classes = 2
    y_test = pd.get_dummies(y_true).to_numpy()
    y_score = np.vstack(pred_score_full)
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Plot all ROC curves
    fig = plt.figure()
    lw = 2
    plt.plot(
        fpr["micro"],
        tpr["micro"],
        label="micro-average ROC curve (area = {0:0.2f})".format(roc_auc["micro"]),
        color="deeppink",
        linestyle=":",
        linewidth=4,
    )

    plt.plot(
        fpr["macro"],
        tpr["macro"],
        label="macro-average ROC curve (area = {0:0.2f})".format(roc_auc["macro"]),
        color="navy",
        linestyle=":",
        linewidth=4,
    )

    colors = cycle(["aqua", "darkorange", "cornflowerblue"])
    for i, color in zip(range(n_classes), colors):
        plt.plot(
            fpr[i],
            tpr[i],
            color=color,
            lw=lw,
            label="ROC curve of class {0} (area = {1:0.2f})".format(i, roc_auc[i]),
        )

    plt.plot([0, 1], [0, 1], "k--", lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve ({})".format(split_name))
    plt.legend(loc="lower right")
    fig.savefig('ROC_{}.png'.format(split_name))


    conf_mat = confusion_matrix(y_true, pred_labels)
    tn, fp, fn, tp = conf_mat.ravel()
    sensitivity = tp / (tp + fn)  # TPR, recall
    specificity = tn / (tn + fp)  # TNR
    fall_out = fp / (fp + tn)  # FPR
    miss_rate = fn / (fn + tp)  # FNR
    print('Accuracy: {}\nROC AUC: {}'.format(accuracy, roc_auc))
    print('Sensitivity (true positive rate, recall): {}\nSpecificity (true negative rate): {}\n'
          'Fall-out (false positive rate): {}\nMiss rate (false negative rate): {}'.
          format(sensitivity, specificity, fall_out, miss_rate))
    if args.compute_conf_mat:
        corr_mat = labels_df.corr(method='pearson')
        print('Correlation mat: {}'.format(corr_mat))
        corr_mat.to_excel(out_dir + 'corr_mat.xlsx', engine='openpyxl')
    print('---------------------------------------')

def main(args):
    if args.model_type not in SUPPORTED_MODEL_TYPES:
        print('Unsupported model type: {}. Supported types: {}'.format(args.model_type, SUPPORTED_MODEL_TYPES))
        exit(1)
    out_dir = args.out_dir
    if out_dir == '':
        out_dir = '.'
    out_dir += '/'
    model_path = args.model_path
    assert os.path.exists(model_path)
    if args.labels_path.endswith('csv'):
        labels_df = pd.read_csv(args.labels_path, header=None) #, delim_whitespace=True)
    else:
        labels_df = pd.read_csv(args.labels_path+'test.csv',  header=None) #, delim_whitespace=True)
    ignores = []
    if os.path.exists(args.ignore_file):
        with open(args.ignore_file, 'r') as f:
            ignores = f.readlines()
        ignores = [ignore.strip() for ignore in ignores]
    if len(ignores) > 0:
        labels_df = labels_df.drop(labels_df[labels_df.Video.isin(ignores)].index)
    if args.model_type == 'tf':
        args_cfg=["--cfg",args.cfg]
        args_2 = parse_args(args_cfg)
        cfg = load_config(args_2)
        model = build_model(cfg)
        model_2 = TimeSformer(cfg=cfg) #, pretrained_model=str(model_path))
    elif args.model_type == 'mf':
        cfg = config_from_args(args)
        model = build_model(cfg)
        cu.load_test_checkpoint(cfg, model)

    model.eval()
    model_2.eval()
    #hi = compute_heads_importance(model, labels_df, args)

    pred_labels = []
    pred_scores = []
    pred_scores_full = []
    attentions_pkl = []
    true_labels=[]
    for line in tqdm(labels_df.iterrows(), total=len(labels_df)):
        # vid_path = args.dataset_path + '/' + line[1]['Video'] + args.video_ext
        vid_path_label = line[1][0].split(' ')
        vid_path = vid_path_label[0]
        true_label = vid_path_label[1]        
        print(vid_path)
        if not os.path.exists(vid_path):
            print('Can\'t find video {}, skipping'.format(vid_path))
            continue
        # label, full_score, temporal_attentions, spatial_attentions, hidden_states = score(model, vid_path,args, cfg, true_label, args.start_pts, args.end_pts)
        label, full_score, temporal_attentions, spatial_attentions, hidden_states = score(model_2, vid_path, args, cfg, true_label, args.start_pts, args.end_pts)
        print("True Label:", true_label)
        print("Pred Label:", label)

        if label == None:
            continue
        true_labels.append(int(true_label))
        pred_labels.append(label)
        pred_scores.append(max(full_score))
        pred_scores_full.append(full_score)
        if args.output_attentions:
            if len(attentions_pkl)<70 or '4viBLg33' in vid_path:
                attentions_pkl.append({'video_path': vid_path, 'pred': label,  'temporal_attn':temporal_attentions,
                                   'spatial_attn': spatial_attentions, 'hidden_states':hidden_states})
    if args.output_attentions:
        model_name = args.model_path.split("/")[-1].split(".")[0]
        path_to_att = out_dir + f'attentions_{model_name}.pkl'
        print(f"Save attentions and hidden states to {path_to_att}")
        with open(path_to_att, 'wb') as f:
            pkl.dump(attentions_pkl, f)
    exp_name = args.exp_name
    if exp_name == '':
        exp_name = os.path.basename(model_path)
    labels_df[exp_name + '_label'] = pred_labels
    labels_df[exp_name + '_score'] = pred_scores_full
    labels_df['ann'] = true_labels
    
    out_path = out_dir + os.path.basename(args.labels_path) + 'score_out.csv'
    labels_df.to_csv(out_path)


    if not args.inference_only:
        if 'LS' in labels_df.columns:
            labels_df['LS'] = pd.to_numeric(labels_df['LS'], errors='coerce').fillna(0).astype(int)

        compute_metrics(labels_df, args.class_col, exp_name + '_label', exp_name + '_score')
        if 'is_train' in labels_df.columns:
            compute_metrics(labels_df[labels_df.is_train == 1], args.class_col, exp_name + '_label', exp_name + '_score', split_name='Train')
            compute_metrics(labels_df[labels_df.is_train == 0], args.class_col, exp_name + '_label', exp_name + '_score', split_name='Test')

            
if __name__ == '__main__':
    parser = argparse.ArgumentParser('Label and score video using trained TimeSFormer model')
    parser.add_argument('--model_type', required=False, type=str, help='Model type (ts for timesformer, ms for motionformer', default='tf')
    parser.add_argument('--dataset_path', required=False, type=str, help='Dataset path', default='/data/home/amitblei/sheba/cd_inception/videos')
    parser.add_argument('--labels_path', required=True, type=str, help='Labels file path', default='/data/home/rkellerm/ibd/sheba/Capsule_inception_CD_with_LS_calpro_updated_003.xlsx')
    parser.add_argument('--model_path', required=True, type=str, help='Trained model file path', default='/data/home/rkellerm/exp/TimeSformer/from03/checkpoints_8_600/checkpoint_epoch_00150.pyth')
    parser.add_argument('--class_col', type=str, help='Classification labels column name', default='ann')
    parser.add_argument('--video_ext', type=str, help='Video files extension', default='.avi')
    parser.add_argument('--ignore_file', required=False, type=str, help='File with entry IDs to ignore', default='/data/home/rkellerm/ibd/sheba/inception_ignore.txt')
    parser.add_argument('--sampling_rate', type=int, default=6)
    parser.add_argument('--num_frames', type=int, default=32)
    parser.add_argument('--num_ensemble_views', type=int, default=5)
    parser.add_argument('--spatial_scale', type=int, default=224)
    parser.add_argument('--exp_name', required=False, default='', help='Experiment name (to appear in the columns to be added of score and label)')
    parser.add_argument('--out_dir', required=False, default='', help='Results directory')
    parser.add_argument('--inference_only', help='Run inference only, without test stats', action='store_true')
    parser.add_argument('--debug', help='Enable debug prints', action='store_true')
    parser.add_argument('--compute_conf_mat', help='Compute confusion matrix (meaningful when therer are several tabular features)', action='store_true')
    parser.add_argument('--output_attentions', action='store_true')
    parser.add_argument('--head_mask_hl', default=None)
    parser.add_argument('--start-pts', type=int, default=None, help='Presentation TimeStamp of the requested frames to start with')
    parser.add_argument('--end-pts', type=int, default=None, help='Presentation TimeStamp of the requested frames to end with')
    parser.add_argument('--cfg', type=str, required=True, default=None, help='Timesformer cfg yaml file')
    
    args = parser.parse_args()
    main(args)
    
    
'''
Command:
python score.py --labels-path test.csv --model-path checkpoints/checkpoint_epoch_00020.pyth --output_attentions --cfg /home/agur/projects/AMD/AMD_Timesformer/configs/AMD_OCT/TimeSformer_AMD_OCT_clasify_sevirity_DB224_FM_C224F38S6D_1gpus.yaml

###python3 score.py --dataset-path /data/home/amitblei/sheba/cd_inception/videos --model-path /data/home/rkellerm/exp/TimeSformer/from03/checkpoints_8_600/checkpoint_epoch_00150.pyth --labels-path /data/home/rkellerm/ibd/sheba/Capsule_inception_CD_with_LS_calpro_updated_003.xlsx --class-col 'received biologic' --video-ext '.mpg'
'''

