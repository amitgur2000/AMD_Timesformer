from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import sys, os
currentdir = os.getcwd()
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
from score import get_frames, compute_heads_importance
import cv2
import seaborn as sns
import glob
import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle as pkl
import torch
from timesformer.models.vit import TimeSformer
from timesformer.utils.parser import load_config, parse_args
import math

NUM_CLASSES = 5
WET_CLASSES = [4]

class VisualizerLoader:
    def __init__(self, args):
        self.args = args
        self.load_metadata()
        self.load_npy_jpg_ulcers()
        self.load_collaterals()
        self.load_model()

    def load_metadata(self):
        print(f"Loading metadata and ground truth from {self.args.labels_path}...")
        self.labels_df = pd.read_csv(self.args.labels_path, header=None, delim_whitespace=True)
        self.labels_df[1] = self.labels_df[1].apply(lambda x: float(x) if isinstance(x, int) else -1)

    def load_npy_jpg_ulcers(self):
        jpg_ulcers_folder = os.path.join(self.args.ulcers_path, '*.jpg')
        print(f"Loading ulcers jpg and pts from {jpg_ulcers_folder}...")
        #  npy_ulcers_paths = glob.glob(os.path.join(self.args.ulcers_path, '*.npy'))
        jpg_ulcers_paths = glob.glob(jpg_ulcers_folder)
        self.jpg_ulcers = {}
        self.pts_ulcers = {}
        for npy in jpg_ulcers_paths:
            vid = npy.split("vid_")[-1].split("_")[0]
            if vid not in self.labels_df["Video"].values:
                continue
            vid = f"{self.args.dataset_path}/{vid}.mpg"
            pts = npy.split("pts_")[-1].split("_")[0]
            if vid not in self.pts_ulcers:
                self.pts_ulcers[vid] = []
                self.jpg_ulcers[vid] = []
            self.pts_ulcers[vid].append(pts)
            self.jpg_ulcers[vid].append(npy)

    def load_collaterals(self):
        print(f"Loading collaterals...")
        with open(self.args.collaterals_path, 'rb') as f:
            self.attentions = pkl.load(f)
        # AG: why to change the paths?    
        # for i, ex in enumerate(self.attentions):
        #     self.attentions[i]['video_path'] = os.path.join(self.args.dataset_path, ex['video_path'].split("/")[-1])
        all_temporal_attentions_per_ex = []
        all_spatial_attentions_per_ex = []
        for ex in self.attentions:
            all_temporal_attentions_per_ex.append(torch.stack(ex['temporal_attn']))
            all_spatial_attentions_per_ex.append(torch.stack(ex['spatial_attn']))
        self.all_temporal_attentions = torch.stack(all_temporal_attentions_per_ex)
        self.all_spatial_attentions = torch.stack(all_spatial_attentions_per_ex)

    def load_model(self):
        print(f"Loading TimeSFormer model from {self.args.model_path} with spatial scale: {self.args.spatial_scale} "
              f"and num frames: {self.args.num_frames}...")
        args_cfg=["--cfg",self.args.cfg]
        args_2 = parse_args(args_cfg)
        cfg = load_config(args_2)
        self.cfg=cfg
        self.model = TimeSformer(cfg)
        self.model.eval()

class Visualizer(VisualizerLoader):
    def __init__(self, args):
        super(Visualizer, self).__init__(args)

    def generate_id_for_videos(self, videos_paths):
        video_indices = []
        for vid_name in videos_paths:
            vid_name = vid_name.split("/")[-1].split(".")[0]
            for i in range(len(self.attentions)):
                if vid_name in self.attentions[i]['video_path']:
                    print(f"In attentions index for video {vid_name} is {i}")
                    video_indices.append(i)
            for vid in self.jpg_ulcers:
                if vid_name in vid:
                    print(f"In ulcers list, jpg for {vid_name} is {vid}")
        return video_indices

    def get_att_and_frame(self, attn_type, layer, head, vid_idx, frame_idx, spatial_idx, same_frame, start_pts, end_pts,
                          recompute):
        frames = get_frames(self.attentions[vid_idx]['video_path'], self.args.sampling_rate, self.args.num_frames, 
                            self.args.num_ensemble_views, 1, self.args.spatial_scale,   
                            same_frame, start_pts, end_pts)               # AG: TODO support more than one temporal view
        if start_pts is not None or end_pts is not None or recompute:
            pred, temporal_attentions, spatial_attentions, hidden_states = self.model(frames, output_attentions=True,
                                                                                      head_mask=None)
            if attn_type == 'spatial':
                attn = spatial_attentions[layer][frame_idx, head]
            elif attn_type == 'temporal':
                attn = temporal_attentions[layer][spatial_idx, head]
        else:
            if attn_type == 'spatial':
                attn = self.all_spatial_attentions[vid_idx, layer, frame_idx, head, :, :]
            elif attn_type == 'temporal':
                attn = self.all_temporal_attentions[vid_idx, layer, spatial_idx, head, :, :]
        frames = frames[0].permute((1, 2, 3, 0))
        frame = frames[frame_idx]
        return attn, frame, frames

    def plot_spatial_attn(self, layer, head, vid_idx, frame_idx, pixel_id, only_cls,
                          start_pts, end_pts, same_frame, recompute, direct):
        attn, frame, frames = self.get_att_and_frame('spatial', layer, head, vid_idx, frame_idx, pixel_id, same_frame,
                                                     start_pts, end_pts, recompute)
        if only_cls:
            attn_from = attn[0, 1:]
            if not direct:
                attn_from = attn[1:, 0]
        else:
            attn_wo_cls = attn[1:, 1:]
            if not direct:
                attn_wo_cls = attn_wo_cls.T
            attn_from = attn_wo_cls[pixel_id]
            mask_frame_to = torch.zeros(196)
            mask_frame_to[pixel_id] = 1
            mask_frame_to = mask_frame_to.view((14, 14))
            mask_frame_to_resized = cv2.resize(mask_frame_to.detach().numpy(), frame.shape[:2],
                                               interpolation=cv2.INTER_LINEAR)

        mask_frame_from = attn_from.view((14, 14))
        print("max Spacial attention:", mask_frame_from.max())  # AG
        mask_frame_from_resized = cv2.resize(mask_frame_from.detach().numpy(), frame.shape[:2],
                                             interpolation=cv2.INTER_LINEAR)
        plt.figure()
        if not only_cls:
            plt.subplot(1, 2, 1)
            plt.title(f"Spatial attention from ")
            plt.imshow((frame - frame.min()) / (frame - frame.min()).max(), 'gray')
            plt.imshow(mask_frame_to_resized, cmap='magma', interpolation='none', alpha=0.5)
            plt.subplot(1, 2, 2)
        else:
            plt.subplot(1, 1, 1)
            pixel_id = '[CLS]'
        plt.title(f"At layer, head : ({layer}, {head}), Vid, frame ({vid_idx}, {frame_idx})")
        plt.imshow((frame - frame.min()) / (frame - frame.min()).max(), 'gray')
        plt.imshow(mask_frame_from_resized * 10, cmap='magma', interpolation='none', alpha=0.6)

    def plot_temporal_attn(self, layer, head, vid_idx, spatial_idx, frame_idx,
                           start_pts, end_pts, same_frame, recompute, direct):
        attn, frame, frames = self.get_att_and_frame('temporal', layer, head, vid_idx, frame_idx, spatial_idx,
                                                     same_frame, start_pts, end_pts, recompute)
        if not direct:
            attn = attn.T
        spatial_pixel = torch.zeros(196) + 0.2
        spatial_pixel[spatial_idx] = 1
        spatial_pixel = spatial_pixel.view((14, 14)).unsqueeze(-1)
        mask_spatial_pixel = cv2.resize(spatial_pixel.detach().numpy(), frame.shape[:2],
                                        interpolation=cv2.INTER_LINEAR)[:, :, np.newaxis]
        attn_from = attn[frame_idx]
        frame_attn = []
        for f, att in zip(frames, attn_from):
            frame_attn.append(f.detach().numpy() * float(att) * mask_spatial_pixel)
        frame *= mask_spatial_pixel
        plt.figure()
        plt.title(f"Temporal attention from at layer, head : ({layer}, {head})")
        plt.imshow((frame - frame.min()) / (frame - frame.min()).max(), 'gray')
        plt.show()
        plt.figure()
        num_cols=4
        num_rows=math.ceil(len(attn_from)/num_cols)     # AG fix the number of rows, in case of more temporal frames
        fig, axs = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(20, 17))

        for i, (att, f) in enumerate(zip(attn_from, frame_attn)):
            axs.flat[i].set_title(f"Att: {round(float(att), 3)}")
            axs.flat[i].imshow((f - f.min()) / (f - f.min()).max(), 'gray')

    def plot_attn(self, attn_type, layer, head, vid_idx, spatial_idx, frame_idx, only_cls_spatial,
                  start_pts=None, end_pts=None, same_frame=False, recompute=False, direct=True):
        if attn_type == 'spatial':
            self.plot_spatial_attn(layer, head, vid_idx, frame_idx, spatial_idx, only_cls_spatial,
                                   start_pts, end_pts, same_frame, recompute, direct)
        elif attn_type == 'temporal':
            self.plot_temporal_attn(layer, head, vid_idx, spatial_idx, frame_idx,
                                    start_pts, end_pts, same_frame, recompute, direct)

    def visualize_stats(self, attn_type='spatial', metric='std'):
        if attn_type == 'spatial':
            all_spatial_or_temporal_attentions = self.all_spatial_attentions
        elif attn_type == 'temporal':
            all_spatial_or_temporal_attentions = self.all_temporal_attentions
        else:
            raise Exception(f"Type {attn_type} not recognized. Choose from [spatial, temporal]")
        attentions_pos = torch.stack(
            [att for i, att in enumerate(all_spatial_or_temporal_attentions) if self.attentions[i]['pred'] in WET_CLASSES])  # AG fixed to mre than 2 classes
        attentions_neg = torch.stack(
            [att for i, att in enumerate(all_spatial_or_temporal_attentions) if self.attentions[i]['pred'] not in WET_CLASSES])

        if metric == 'std':
            metric_temporal_pos = attentions_pos.std((-1, -2))
            metric_temporal_neg = attentions_neg.std((-1, -2))
        elif metric == 'mean':
            metric_temporal_pos = attentions_pos.mean((-1, -2))
            metric_temporal_neg = attentions_neg.mean((-1, -2))
        else:
            raise Exception(f"Metric {metric} not recognized. Choose from [std, mean]")

        metric_temporal_pos = metric_temporal_pos.mean(0).mean(1).detach().numpy()
        metric_temporal_neg = metric_temporal_neg.mean(0).mean(1).detach().numpy()

        self.plot_metrics(metric_temporal_pos, f"Attention {attn_type} pos - {metric}")
        self.plot_metrics(metric_temporal_neg, f"Attention {attn_type} pos - {metric}")

        delta = np.absolute(metric_temporal_pos - metric_temporal_neg)
        self.plot_metrics(delta, f"Attention {attn_type} delta - {metric}")

    def plot_metrics(self, std_attn, title):
        fig, ax = plt.subplots(1, 1)
        att_df = pd.DataFrame(std_attn)
        sns.heatmap(att_df, cmap="Blues", ax=ax)
        ax.set_title(title, fontsize=10)

    def compute_heads_importance(self):
        self.model.train()
        print(self.cfg.MODEL)
        hi = compute_heads_importance(self.model, self.labels_df, self.args, self.cfg)
        figsize = (15, 12)
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        hi_df = pd.DataFrame(hi[1].detach().numpy())
        sns.heatmap(hi_df, cmap="Blues", ax=ax)
        ax.set_title("Heads importance", fontsize=10)
        self.model.eval()

    def reduce_data(self, only_cls, reducer):
        if only_cls:
            hs = self.cls_hidden_states
        else:
            hs = self.patches_hidden_states
        self.tsne_coord = {}
        self.pca_coord = {}
        if reducer in ["both", "tsne"]:
            for layer in tqdm(range(12)):
                results = TSNE(n_components=2, random_state=42, n_jobs=-1).fit_transform(
                    hs[layer].detach().numpy()
                )
                self.tsne_coord[layer] = results
        if reducer in ["both", "pca"]:
            for layer in tqdm(range(12)):
                results = PCA(n_components=2).fit_transform(
                    hs[layer].detach().numpy()
                )
                self.pca_coord[layer] = results
        if reducer not in ["both", "pca", "tsne"]:
            raise Exception(f"Reducer {reducer} is not recognized. Choose in [both, pca, tsne]")

    def load_hidden_states(self, n_data_hiddenstates):
        self.all_hidden_states = torch.stack(
            [torch.concat(self.attentions[i]['hidden_states']) for i in range(n_data_hiddenstates)]).permute(
            (1, 0, 2, 3))[:, :, 1:, :]
        self.patches_hidden_states = self.all_hidden_states.reshape(
            (self.all_hidden_states.shape[0], -1, self.all_hidden_states.shape[-1]))
        all_hidden_states_4cls = torch.stack(
            [torch.concat(self.attentions[i]['hidden_states']) for i in range(n_data_hiddenstates)]).permute(
            (1, 0, 2, 3))
        self.cls_hidden_states = all_hidden_states_4cls[:, :, 0, :]

    def reduce_dim_and_plot(self, n_data_hiddenstates, only_cls, reducer):
        if n_data_hiddenstates < 0:
            n_data_hiddenstates = len(self.attentions)
        self.load_hidden_states(n_data_hiddenstates)
        model_name = self.args.model_path.split("/")[-1].split(".")[0]
        if only_cls:
            path_to_save_to = f"df_{reducer}_{n_data_hiddenstates}_cls_from_{model_name}.csv"
        else:
            path_to_save_to = f"df_{reducer}_{n_data_hiddenstates}_patches_from_{model_name}.csv"
        if os.path.exists(path_to_save_to):
            df = pd.read_csv(path_to_save_to)
            print(f"Loaded from {path_to_save_to}")
        else:
            self.reduce_data(only_cls, reducer)
            df = pd.DataFrame(
                columns=['ex_idx', 'pixel_id', 'tsne_x', 'tsne_y', 'pca_x', 'pca_y', 'pred', 'gt', 'layer'],
                index=range(
                    self.all_hidden_states.shape[0] * self.all_hidden_states.shape[1] *
                    self.all_hidden_states.shape[2]))
            i = 0
            for layer in tqdm(range(12)):
                if reducer in ["both", "pca"]:
                    pca_coord_layer = self.pca_coord[layer].reshape(
                        (n_data_hiddenstates, -1, 2))
                if reducer in ["both", "tsne"]:
                    tsne_coord_layer = self.tsne_coord[layer].reshape(
                        (n_data_hiddenstates, -1, 2))
                for ex_idx in range(n_data_hiddenstates):
                    if only_cls:
                        list_pixel_ids = 1
                    else:
                        list_pixel_ids = self.all_hidden_states.shape[2]
                    for pixel_id in range(list_pixel_ids):
                        df['ex_idx'].iloc[i] = ex_idx
                        df['layer'].iloc[i] = layer
                        df['pred'].iloc[i] = self.attentions[ex_idx]['pred']
                        df['pixel_id'].iloc[i] = pixel_id
                        df['gt'].iloc[i] = int(self.labels_df.iloc[ex_idx][1])
                        if reducer in ["both", "tsne"]:
                            df['tsne_x'].iloc[i] = tsne_coord_layer[ex_idx, pixel_id][0]
                            df['tsne_y'].iloc[i] = tsne_coord_layer[ex_idx, pixel_id][1]
                        if reducer in ["both", "pca"]:
                            df['pca_x'].iloc[i] = pca_coord_layer[ex_idx, pixel_id][0]
                            df['pca_y'].iloc[i] = pca_coord_layer[ex_idx, pixel_id][1]
                        i += 1
            df.to_csv(path_to_save_to)
            print(f"Saved to {path_to_save_to}")
        if only_cls:
            self.df_cls = df
        else:
            self.df_patches = df


def plot_2d(df, colored_by, layer, reducer="tsne", with_legend=False):
    sub_df = df[df['layer'] == layer]
    fig, ax = plt.subplots()
    colors = np.linspace(0, 2, len(sub_df[colored_by]))
    colordict = dict(zip(sub_df[colored_by], colors))
    ax.scatter(sub_df[f"{reducer}_x"], sub_df[f"{reducer}_y"], c=sub_df[colored_by])
    if with_legend:
        ax.legend()
    ax.set_title(f"Layer {layer}, colored by {colored_by}")
    plt.show()


def plot_attn(all_attentions, layer, head, ex_id, spatial_feature, title):
    attn = all_attentions[ex_id, layer, spatial_feature, head, :, :]
    figsize = (5, 3) if attn.shape[0] == 8 else (15, 12)
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    att_df = pd.DataFrame(attn.detach().numpy())
    sns.heatmap(att_df, cmap="Blues", ax=ax)
    ax.set_title(title, fontsize=10)


def pixels_with_max_att(attn_wo_cls):
    pixel_where_max = -1
    maxi = -1
    for pixel_id in range(len(attn_wo_cls)):
        attn_from = attn_wo_cls[pixel_id]
        if attn_from.max() > maxi:
            pixel_where_max = pixel_id
            maxi = attn_from.max()
    return pixel_where_max, maxi


def layers_with_max_att(all_spatial_attentions, vid_idx, frame_idx, pixel_id):
    lh_where_max = -1
    maxi = -1
    for layer in range(12):
        for head in range(12):
            attn = all_spatial_attentions[vid_idx, layer, frame_idx, head, :, :]
            attn = attn[1:, 1:]
            attn = attn[pixel_id]
            if attn.max() > maxi:
                lh_where_max = (layer, head)
                maxi = attn.max()
    return lh_where_max, maxi


import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def plot_img(img_png):
    img = mpimg.imread(img_png)
    plt.imshow(img)
    plt.show()


