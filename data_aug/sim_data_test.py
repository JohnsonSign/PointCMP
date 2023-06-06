import os
import sys
import numpy as np
import copy
import torch
from torch.utils.data import Dataset


class CLRMSRSubject(Dataset):
    def __init__(self, root, meta, frames_per_clip=16, step_between_clips=1, num_points=2048, sub_clips=4, step_between_frames=1, train=True):
        super(CLRMSRSubject, self).__init__()

        self.sub_clips = sub_clips
        self.root = root
        self.videos = []
        self.index_map = []
        index = 0

        with open(meta, 'r') as f:
            for line in f:
                name, nframes = line.split()
                if train:
                    if int(name.split('_')[1].split('s')[1]) <= 5:
                        nframes = int(nframes)
                        for t in range(0, nframes-step_between_frames*(frames_per_clip-1), step_between_clips):
                            self.index_map.append((index, t))
                        index += 1
                        self.videos.append(os.path.join(root, name+'.npz'))

        self.frames_per_clip = frames_per_clip
        self.step_between_clips = step_between_clips
        self.step_between_frames = step_between_frames
        self.num_points = num_points
        self.train = train


    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        index, t = self.index_map[idx]

        video_name = self.videos[index]
        video = np.load(video_name, allow_pickle=True)['point_clouds']

        clip = [video[t+i*self.step_between_frames] for i in range(self.frames_per_clip)]
        for i, p in enumerate(clip):
            if p.shape[0] > self.num_points:
                r = np.random.choice(p.shape[0], size=self.num_points, replace=False)
            else:
                repeat, residue = self.num_points // p.shape[0], self.num_points % p.shape[0]
                r = np.random.choice(p.shape[0], size=residue, replace=False)
                r = np.concatenate([np.arange(p.shape[0]) for _ in range(repeat)] + [r], axis=0)
            clip[i] = p[r, :]
        clip = np.array(clip)

        clipv2 = copy.deepcopy(clip)

        # V1
        scales = np.random.uniform(0.9, 1.1, size=3)
        clip = clip * scales
        clip = clip / 300 
        clips = np.split(clip, indices_or_sections=self.sub_clips, axis=0)
        clips = np.array(clips) # [S, L', N, 3]

        # V2
        scalesv2 = np.random.uniform(0.9, 1.1, size=3)
        clipv2 = clipv2 * scalesv2
        clipv2 = clipv2 / 300 

        jittered_data = np.random.normal(0, 0.01, size=(clipv2.shape[0],clipv2.shape[1],3)).clip(-0.02, 0.02)
        translation = np.random.normal(0, 0.01, size=(3)).clip(-0.05, 0.05)
        clipv2 = clipv2 + jittered_data + translation

        clipsv2 = np.split(clipv2, indices_or_sections=self.sub_clips, axis=0)
        clipsv2 = np.array(clipsv2) # [S, L', N, 3]

        return clips.astype(np.float32), clipsv2.astype(np.float32), index
            

if __name__ == '__main__':
    np.random.seed(0)
    dataset = CLRMSRSubject(root='./MSRAction')
    clips, clipsv2, index = dataset[0]
    print('clips.shape:', clips.shape)
