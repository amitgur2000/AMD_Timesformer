import tqdm as notebook_tqdm
import visualize as vis




class Arguments:
    cfg = "/home/agur/projects/AMD/AMD_Timesformer/configs/AMD_OCT/TimeSformer_AMD_OCT_clasify_sevirity_DB224_FM_C224F38S6D_4gpus.yaml"
    labels_path = "/home/agur/projects/AMD/AMD_Timesformer/test.csv"
    model_path = "checkpoints/checkpoint_epoch_00021.pyth"
    out_dir = ""
    spatial_scale = 224
    num_ensemble_views=1
    num_frames=32
    dataset_path = "/DCAIOCTO/ModelDataZoo/datasets/AMD_data/WIP_vids_no_Fund/"
    video_ext = ".avi"
    sampling_rate = 6
    class_col = 1 # "AMD_STAGE"  "received biologic"
    collaterals_path =  "/home/agur/projects/AMD/AMD_Timesformer/attentions_checkpoint_epoch_00021.pkl" #'/home/agur/projects/AMD/AMD_Timesformer/checkpoints/attentions_ckpt_val.pkl'
    ulcers_path = '/home/agur/projects/AMD/AMD_Timesformer'
args = Arguments()
print(args.labels_path)



Viz = vis.Visualizer(args)


import csv
video_paths=[]
labels_orig=[]
with open(args.labels_path, 'r') as f:
    file_name_and_ann=csv.reader(f, delimiter=' ')
    for rows in file_name_and_ann:
        video_paths.append(rows[0])
        labels_orig.append(rows[1])
print("Video files: \n", video_paths)
num_videos=len(video_paths)
pts_ = [(0, args.num_frames-1)] * num_videos
print(pts_)


# %%
Viz.generate_id_for_videos(video_paths)

# %% [markdown]
# ## Visualizations

# %% [markdown]
# ### Attentions visualizations

# %%
for layer, head in [(1,9)]:
    video_list_indx = list(range(num_videos))
    for vid_idx, pts in zip(video_list_indx, pts_):
        print(layer, head, pts, vid_idx)

# %%
frame_idx=9
only_cls = True
#video_list_indx=[4]
list_layers_heads = [(10,3), (2,10)] # [(0,2),(0,6),(0,7),(1,6),(2,0),(3,5),(3,8),(8,9),(9,4),(10,1),(11,8)] #   (a,b) for a in range(12) for b in range(12)]
for layer, head in list_layers_heads:
    for vid_idx, pts in zip(video_list_indx, pts_):
        # Viz.plot_attn('spatial', layer, head, vid_idx, -1, frame_idx, only_cls, start_pts=None, end_pts=None, direct=False)  #=pts[0], end_pts=pts[1], direct=False)
        Viz.plot_attn('spatial', layer, head, vid_idx, -1, frame_idx, only_cls, start_pts=pts[0], end_pts=pts[1], direct=False)            

# %%
frame_idx=8
for layer, head in [(0,10)]:
        for vid_idx, pts in zip([0, 1], pts_):
            Viz.plot_attn('spatial', layer, head, vid_idx, -1, frame_idx, only_cls, start_pts=pts[0], end_pts=pts[1], direct=True)


# %%
frame_idx=8
only_cls = False
spatial_idx = 80
for layer, head in [(0,10)]:
    for vid_idx, pts in zip(video_list_indx, pts_):
        Viz.plot_attn('spatial', layer, head, vid_idx, spatial_idx, frame_idx, only_cls, start_pts=pts[0], end_pts=pts[1], direct=True)


# %%
start_pts = 0
end_pts = 29
layer =0
head=0
vid_idx =4
spatial_idx = 100
frame_idx = 9
only_cls=False
Viz.plot_attn('temporal', layer, head, vid_idx, spatial_idx, frame_idx, only_cls, start_pts=pts[0], end_pts=pts[1], direct=True)


# %%
Viz.visualize_stats(attn_type='spatial', metric='std')

# %%
Viz.visualize_stats(attn_type='spatial', metric='mean')

# %%
Viz.compute_heads_importance()

# %% [markdown]
# ### Hidden states visalizations

# %%
Viz.reduce_dim_and_plot(n_data_hiddenstates=-1, only_cls=True, reducer="both")

# %%
Viz.reduce_dim_and_plot(n_data_hiddenstates=2, only_cls=False, reducer="both")

# %%
Viz.all_hidden_states.shape

# %%
colored_by = 'pred'
for layer in range(12):
    plot_2d(Viz.df_patches, colored_by, layer)

# %%
colored_by = 'pixel_id'
for layer in range(12):
    plot_2d(Viz.df_patches, colored_by, layer)

# %%
colored_by = 'pred'
for layer in range(12):
    plot_2d(Viz.df_cls, colored_by, layer)

# %%
colored_by = 'gt'
for layer in range(12):
    plot_2d(Viz.df_cls, colored_by, layer)

# %%
colored_by = 'ex_idx'
for layer in range(12):
    plot_2d(Viz.df_cls, colored_by, layer)

# %%



