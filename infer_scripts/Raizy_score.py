import argparse
import numpy as np
from PIL import Image
import os
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import accuracy_score, roc_auc_score, precision_recall_fscore_support, classification_report, confusion_matrix
from scipy.special import softmax
import pickle as pkl
import torch
from torchvision import transforms
from timesformer.datasets import decoder as decoder
from timesformer.datasets import video_container as container
from timesformer.datasets import utils as utils
from timesformer.models.vit import TimeSformer
from timm.data import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
import timesformer.models.losses as losses

from pdb import set_trace as bp
    
def get_frames(video_path, sampling_rate, num_frames, num_ensemble_views, temporal_sampling_index, spatial_scale, same_frame=False, start_pts=None, end_pts=None, is_vid=True):
    decoding_backend = 'pyav'
    target_fps = 30
    MEAN = [0.45, 0.45, 0.45]
    STD = [0.225, 0.225, 0.225]
    RANDOM_FLIP = True
    INV_UNIFORM_SAMPLE = False
    crop_size = spatial_scale
    min_spatial_scale = spatial_scale
    max_spatial_scale = spatial_scale
    spatial_sample_index = 1 # center/ middle. TODO: test time augmentation?
    
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

def score(model, video_path, sampling_rate, num_frames, num_ensemble_views, spatial_scale, output_attentions, head_mask_hl, start_pts=None, end_pts=None, same_frame=False):
    score_by_avg = False # score by average or by majority voting
    if head_mask_hl is not None:
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
    
    for ensemble_index in range(num_ensemble_views):
        frames = get_frames(video_path, sampling_rate, num_frames, num_ensemble_views, ensemble_index, spatial_scale, same_frame=same_frame, start_pts=start_pts, end_pts=end_pts)
        pred, temporal_attentions, spatial_attentions, hidden_states = model(frames, output_attentions=output_attentions,
                                                                             head_mask=head_mask)
        # pred = model(frames)
        print(f'\n For video-{video_path}, The prediction -{pred}')
        #assert pred.shape[1] == 2
        preds_np = pred.detach().numpy()
        all_preds.append([preds_np[0][0] , preds_np[0][1]])
        all_temporal.append(temporal_attentions)
        all_spatial.append(spatial_attentions)
        #sm = softmax(preds_np)
        pred_label = np.argmax(preds_np)
        all_labels.append(pred_label)


    if score_by_avg:
        pred_label = np.average(all_labels)
    else: # majority voting
        vals, cnt = np.unique(all_labels, return_counts=True)
        pred_label = np.argmax(cnt)

    preds_np = np.average(all_preds, axis=0)
    print(f'All Preds -{all_preds}')


    # all_preds.append()
    #TODO: attentions? currently returning the last.
    return pred_label, preds_np, temporal_attentions, spatial_attentions, hidden_states , all_preds


def get_loss_func():
    loss_fun = losses.get_loss_func('cross_entropy')(reduction="mean")
    return loss_fun

def compute_heads_importance(
        model, labels_df, args, same_frame=False, compute_importance=True, head_mask=None, actually_pruned=False
):
    """This method shows how to compute:
    - head attention entropy
    - head importance scores according to http://arxiv.org/abs/1905.10650
    """
    # Explicitly declare reduction to mean.
    loss_fun = get_loss_func()
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
        vid_path = args.dataset_path + '/' + line[1]['Video'] + args.video_ext
        # print(line[1]['Video'])
        if not os.path.exists(vid_path):
            print('Can\'t find video {}, skipping'.format(vid_path))
            continue
        frames = get_frames(vid_path, args.sampling_rate, args.num_frames, args.spatial_scale, same_frame=same_frame)
        pred, temporal_attentions, spatial_attentions, hidden_states = model(frames,
                                                                             output_attentions=False,
                                                                             head_mask=head_mask)
        label = line[-1][args.class_col]
        label = torch.tensor([label])
        # Do a forward pass (not with torch.no_grad() since we need gradients for importance score - see below)
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

def main(args):
    out_dir = args.out_dir
    if out_dir == '':
        out_dir = '.'
    out_dir += '/'
    out_path = out_dir + os.path.basename(args.labels_path)
    model_path = args.model_path
    # assert os.path.exists(model_path)
    labels_df = pd.read_excel(args.labels_path, engine='openpyxl')

    # ignores = []
    # if os.path.exists(args.ignore_file):
    #     with open(args.ignore_file, 'r') as f:
    #         ignores = f.readlines()
    #     ignores = [ignore.strip() for ignore in ignores]
    #
    # if len(ignores) > 0:
    #     labels_df = labels_df.drop(labels_df[labels_df.Video.isin(ignores)].index)
        
    model = TimeSformer(img_size=args.spatial_scale, num_classes=5, num_frames=args.num_frames, attention_type='divided_space_time',  pretrained_model=str(model_path))
    model.eval()
    #hi = compute_heads_importance(model, labels_df, args)
    final_prediction = pd.DataFrame({'Video':[] , 'full_score': [], 'Predicted Label': []})



    pred_labels = []
    pred_scores = []
    pred_scores_full = []
    attentions_pkl = []
    for line in tqdm(labels_df.iterrows(), total=len(labels_df)):
        # vid_path = args.dataset_path + '/' + line[1]['Video'] + args.video_ext
        vid_path = args.dataset_path + '/' + line[1]['Video'] + '.mpg'

        if not os.path.exists(vid_path):
            print('Can\'t find video {}, skipping'.format(vid_path))
            continue
        label, full_score, temporal_attentions, spatial_attentions, hidden_states, all_preds = score(model, vid_path,
                                                                                     args.sampling_rate,
                                                                                     args.num_frames,
                                                                                     args.num_ensemble_views,
                                                                                     args.spatial_scale,
                                                                                     args.output_attentions,
                                                                                     args.head_mask_hl,
                                                                                     args.start_pts,
                                                                                     args.end_pts)
        pred_labels.append(label)
        pred_scores.append(max(full_score))
        pred_scores_full.append(full_score)
        print("full_score {}".format(full_score)) #, full_score[0][1]))
        final_prediction = final_prediction.append({'Video': line[1]['Video'] ,'full_score': all_preds,'Predicted Label': label}, ignore_index=True)
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
    out_path2 = os.path.join(out_dir, 'final_predictions.xlsx')
    final_prediction.to_excel(out_path2, engine='openpyxl')

    labels_df[exp_name + '_label'] = pred_labels
    labels_df[exp_name + '_score'] = pred_scores
    labels_df.to_excel(out_path, engine='openpyxl')



    if not args.inference_only:
        if 'LS' in labels_df.columns:
            labels_df['LS'] = pd.to_numeric(labels_df['LS'], errors='coerce').fillna(0).astype(int)
        y_true = labels_df[args.class_col]
        accuracy = accuracy_score(y_true, pred_labels)
        print("pred_scores_full {}".format(pred_scores_full))
        scores_positive = [pred_score_full_i[1] for pred_score_full_i in pred_scores_full]
        print("scores_positive {}".format(scores_positive))
        print("y_true {}".format(list(y_true)))
        roc_auc  = roc_auc_score(y_true, scores_positive)
        #prfs = precision_recall_fscore_support(y_true, pred_labels, average=None, labels=[0,1])
        cls_report = classification_report(y_true, pred_labels, labels=[0,1])
        print(cls_report)
        conf_mat = confusion_matrix(y_true, pred_labels)
        tn, fp, fn, tp = conf_mat.ravel()
        sensitivity = tp / (tp + fn) # TPR, recall
        specificity = tn / (tn + fp) # TNR
        fall_out = fp / (fp + tn) # FPR
        miss_rate = fn / (fn + tp) # FNR
        print('Accuracy: {}\nROC AUC: {}'.format(accuracy, roc_auc))
        print('Sensitivity (true positive rate, recall): {}\nSpecificity (true negative rate): {}\n'
              'Fall-out (false positive rate): {}\nMiss rate (false negative rate): {}'.
              format(sensitivity, specificity, fall_out, miss_rate))
        corr_mat = labels_df.corr(method='pearson')
        print('Correlation mat: {}'.format(corr_mat))
        corr_mat.to_excel(out_dir + 'corr_mat.xlsx', engine='openpyxl')

            
if __name__ == '__main__':
    parser = argparse.ArgumentParser('Label and score video using trained TimeSFormer model')
    parser.add_argument('--dataset-path', required=True, type=str, help='Dataset path', default='../data/videos')
    parser.add_argument('--labels-path', required=True, type=str, help='Labels file path', default='../data/labels/Capsule_inception_CD_with_LS_calpro.xlsx')
    parser.add_argument('--model-path', required=True, type=str, help='Trained model file path', default='../data/models/32x32_224_k400/checkpoint.pyth')
    parser.add_argument('--class-col', type=str, help='Classification labels column name', default='received biologic')
    parser.add_argument('--video-ext', type=str, help='Video files extension', default='.avi')
    parser.add_argument('--ignore-file', required=False, type=str, help='File with entry IDs to ignore', default='/data/home/rkellerm/ibd/sheba/inception_ignore.txt')
    parser.add_argument('--sampling-rate', type=int, default=6)
    parser.add_argument('--num-frames', type=int, default=32)
    parser.add_argument('--num-ensemble-views', type=int, default=5)
    parser.add_argument('--spatial-scale', type=int, default=224)
    parser.add_argument('--exp-name', required=False, default='test', help='Experiment name (to appear in the columns to be added of score and label)')
    parser.add_argument('--out-dir', required=False, default='../data/results', help='Results directory')
    parser.add_argument('--inference-only', help='Run inference only, without test stats', action='store_true')
    parser.add_argument('--output_attentions', action='store_true')
    parser.add_argument('--head_mask_hl', default=None)
    parser.add_argument('--start-pts', type=int, default=None, help='Presentation TimeStamp of the requested frames to start with')
    parser.add_argument('--end-pts', type=int, default=None, help='Presentation TimeStamp of the requested frames to end with')

    args = parser.parse_args()
    main(args)
    
    
'''
Command:
python score.py --dataset-path ../data/videos --model-path ../resources/models/32x32_224_k400/checkpoint.pyth --labels-path ../data/patients_info.xlsx --class-col 'amd_stage' --video-ext '.mpg' --exp-name 16x16_448_k400 --out-dir ./results
'''
