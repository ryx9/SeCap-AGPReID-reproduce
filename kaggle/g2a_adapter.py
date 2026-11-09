"""Separate SeCap-on-G2A-VReID experiment using official video query indices.

Protocol parser follows FHR-L/VSLA-CLIP datasets/data_manager.py (G2A).
Image SeCap trains on the supplied training frames; video retrieval averages
all frame embeddings per official tracklet. No temporal network is introduced.
"""
from pathlib import Path
import os

import numpy as np
from scipy.io import loadmat
import torch

from fastreid.data.datasets import DATASET_REGISTRY
from fastreid.data.datasets.bases import ImageDataset
from fastreid.evaluation import ReidEvaluator
from fastreid.utils import comm


@DATASET_REGISTRY.register()
class G2AVReID(ImageDataset):
    dataset_name = 'g2avreid'

    def __init__(self, root='datasets', **kwargs):
        folder = Path(root) / 'G2A-VReID'
        info = folder / 'info'
        train_names = (info / 'train_name.txt').read_text().splitlines()
        test_names = (info / 'test_name.txt').read_text().splitlines()
        train_meta = loadmat(info / 'tracks_train_info.mat')['track_train_info']
        test_meta = loadmat(info / 'tracks_test_info.mat')['track_test_info']
        queries = loadmat(info / 'query_IDX.mat')['query_IDX'].reshape(-1).astype(int) - 1
        if len(set(queries)) != len(queries) or queries.min() < 0 or queries.max() >= len(test_meta):
            raise ValueError('Invalid official query tracklet indices')
        query_set = set(queries.tolist())
        galleries = [i for i in range(len(test_meta)) if i not in query_set]
        ground_cameras = set(int(x) for x in test_meta[queries, 3])
        aerial_cameras = set(int(x) for x in test_meta[galleries, 3])
        if ground_cameras & aerial_cameras:
            raise ValueError('G2A query/gallery camera sets overlap; verify ground/aerial mapping')
        self.view_mapping = {c: 'Ground' for c in ground_cameras}
        self.view_mapping.update({c: 'Aerial' for c in aerial_cameras})

        directory_entries = {}

        def tracks(names, meta, subset):
            result = []
            for start, end, pid, camera in meta.astype(int):
                if pid == -1:
                    continue
                if camera not in self.view_mapping:
                    raise ValueError(f'Camera {camera} has no verified ground/aerial mapping')
                frames = names[start - 1:end]
                if not frames:
                    raise ValueError('Empty official tracklet')
                paths = [str(folder / subset / frame[:4] / frame) for frame in frames]
                missing = []
                for path in paths:
                    parent, filename = os.path.split(path)
                    if parent not in directory_entries:
                        directory_entries[parent] = set(os.listdir(parent))
                    if filename not in directory_entries[parent]:
                        missing.append(path)
                if missing:
                    raise FileNotFoundError(f'Official G2A tracklet references missing frames: {missing[:2]}')
                result.append((paths, int(pid), int(camera) - 1, self.view_mapping[camera]))
            return result

        train_tracks = tracks(train_names, train_meta, 'bbox_train')
        self.query_tracks = tracks(test_names, test_meta[queries], 'bbox_test')
        self.gallery_tracks = tracks(test_names, test_meta[galleries], 'bbox_test')
        train_ids = {p for _, p, _, _ in train_tracks}
        test_ids = {p for _, p, _, _ in self.query_tracks + self.gallery_tracks}
        if train_ids & test_ids:
            raise ValueError('G2A train/test identity leakage')
        if not {p for _, p, _, _ in self.query_tracks} <= {p for _, p, _, _ in self.gallery_tracks}:
            raise ValueError('Some official queries have no gallery identity match')

        def flatten(items, training=False):
            return [(path, f'g2avreid_{pid}' if training else pid,
                     f'g2avreid_{cam}' if training else cam, view)
                    for paths, pid, cam, view in items for path in paths]

        self.all_tracks = self.query_tracks + self.gallery_tracks
        self.path_to_track = {p: i for i, (paths, _, _, _) in enumerate(self.all_tracks) for p in paths}
        expected = sum(len(paths) for paths, _, _, _ in self.all_tracks)
        if len(self.path_to_track) != expected:
            raise ValueError('A G2A frame belongs to multiple test tracklets')
        super().__init__(flatten(train_tracks, True), flatten(self.query_tracks), flatten(self.gallery_tracks), **kwargs)


class TrackletEvaluator(ReidEvaluator):
    def __init__(self, cfg, dataset, output_dir=None):
        self.dataset = dataset
        super().__init__(cfg, len(dataset.query_tracks), output_dir)

    def reset(self):
        self.sums = None
        self.counts = torch.zeros(len(self.dataset.all_tracks), dtype=torch.long)
        self._predictions = []

    def process(self, inputs, outputs):
        feats = outputs.detach().cpu().float()
        if self.sums is None:
            self.sums = torch.zeros(len(self.dataset.all_tracks), feats.shape[1])
        indices = torch.tensor([self.dataset.path_to_track[p] for p in inputs['img_paths']], dtype=torch.long)
        self.sums.index_add_(0, indices, feats)
        self.counts.index_add_(0, indices, torch.ones(len(indices), dtype=torch.long))

    def evaluate(self):
        # Pool all frames per official video tracklet before the author's ReID
        # metrics. Gathering uses all ranks; only rank zero owns pooled features.
        gathered = comm.gather((self.sums, self.counts), dst=0)
        self._predictions = []
        if comm.is_main_process():
            sums = sum(s for s, _ in gathered)
            counts = sum(c for _, c in gathered)
            expected = torch.tensor([len(paths) for paths, _, _, _ in self.dataset.all_tracks])
            if not torch.equal(counts, expected):
                raise RuntimeError('Distributed video evaluation lost or duplicated frames')
            feats = sums / counts[:, None]
            self._predictions = [dict(feats=feats,
                pids=torch.tensor([pid for _, pid, _, _ in self.dataset.all_tracks]),
                camids=torch.tensor([cam for _, _, cam, _ in self.dataset.all_tracks]))]
        return super().evaluate()
