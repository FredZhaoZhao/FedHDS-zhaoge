import gc
import math

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from m_utils import clustering, get_flatten_features


class Client(object):
    def __init__(self, idx, args, train_loader):
        self.idx = idx
        self.args = args
        self.device = torch.device(f'cuda:{args.device}')

        self.model = None

        self.full_train_dataset = train_loader.dataset
        self.full_collate_fn = train_loader.collate_fn
        self.batch_size = train_loader.batch_size or args.batch_size

        self.train_loader = None
        self.train_iterator = None
        self.selected_clusters = []
        self.feature_sample_indices = []

        self.use_full_training_set()

    def _build_loader(self, dataset, shuffle):
        return DataLoader(
            dataset,
            shuffle=shuffle,
            batch_size=self.batch_size,
            collate_fn=self.full_collate_fn,
            drop_last=False,
        )

    def _build_subset_loader(self, sample_indices, shuffle):
        subset = Subset(self.full_train_dataset, sample_indices)
        return self._build_loader(subset, shuffle=shuffle)

    def use_full_training_set(self):
        self.train_loader = self._build_loader(self.full_train_dataset, shuffle=True)
        self.train_iterator = iter(self.train_loader)

    def get_full_train_loader(self, shuffle=False, sample_limit=0):
        dataset_size = len(self.full_train_dataset)
        if sample_limit is not None and sample_limit > 0 and sample_limit < dataset_size:
            sample_indices = list(range(sample_limit))
            return self._build_subset_loader(sample_indices, shuffle=shuffle)
        return self._build_loader(self.full_train_dataset, shuffle=shuffle)

    def sample_reference_examples(self, sample_size, shuffle=False):
        dataset_size = len(self.full_train_dataset)
        if dataset_size == 0:
            return []

        if sample_size is None or sample_size <= 0 or sample_size >= dataset_size:
            sample_indices = list(range(dataset_size))
        elif shuffle:
            sample_indices = torch.randperm(dataset_size).tolist()[:sample_size]
        else:
            sample_indices = list(range(sample_size))

        return [self.full_train_dataset[idx] for idx in sample_indices]

    def _build_filtering_candidate_loader(self):
        dataset_size = len(self.full_train_dataset)
        sample_limit = self.args.filtering_sample_limit

        if sample_limit is not None and sample_limit > 0 and sample_limit < dataset_size:
            self.feature_sample_indices = list(range(sample_limit))
            return self._build_subset_loader(self.feature_sample_indices, shuffle=False)

        self.feature_sample_indices = list(range(dataset_size))
        return self._build_loader(self.full_train_dataset, shuffle=False)

    def calculated_cluster_center(self):
        if self.model is None:
            print(f"[Client {self.idx}] Error: No model available for feature extraction.")
            return []

        torch.cuda.empty_cache()
        self.model.eval()

        loader = self._build_filtering_candidate_loader()
        with torch.no_grad():
            flatten_hidden_state_list = get_flatten_features(self.model, loader, args=self.args)

        gc.collect()

        features_np = np.array(flatten_hidden_state_list)
        if len(features_np) == 0:
            return []

        feature_dim = features_np.shape[1] if features_np.ndim > 1 else 1
        target_dim = max(1, min(self.args.compound_dim, len(features_np), feature_dim))

        if self.args.feature_layer == 'tsne':
            from sklearn.manifold import TSNE

            perplexity = min(30, len(features_np) - 1)
            if perplexity >= 1 and target_dim < feature_dim:
                tsne = TSNE(n_components=target_dim, perplexity=perplexity)
                reduced_feature_list = tsne.fit_transform(features_np)
            else:
                reduced_feature_list = features_np
        elif self.args.feature_layer == 'pca':
            from sklearn.decomposition import PCA

            if target_dim < feature_dim:
                pca = PCA(n_components=target_dim)
                reduced_feature_list = pca.fit_transform(features_np)
            else:
                reduced_feature_list = features_np
        else:
            reduced_feature_list = features_np

        cluster_labels, centroids, _ = clustering(reduced_feature_list, args=self.args)
        self.reduced_feature_list = reduced_feature_list
        self.cluster_labels = cluster_labels
        self.centroids = centroids

        return list(centroids.items())

    def build_training_set_with_precalculated_clusters(self):
        selected_full_sample_ids = []

        if not hasattr(self, 'cluster_labels') or not self.selected_clusters:
            self.train_loader = None
            self.train_iterator = None
            return

        for cluster_id in self.selected_clusters:
            centroid = self.centroids.get(cluster_id)
            if centroid is None:
                continue

            sample_id_in_cluster = np.argwhere(self.cluster_labels == cluster_id).flatten()
            if len(sample_id_in_cluster) == 0:
                continue

            features_in_cluster = self.reduced_feature_list[sample_id_in_cluster]
            distances = np.linalg.norm(features_in_cluster - centroid, axis=1)
            chosen_local_idx = int(sample_id_in_cluster[np.argmin(distances)])
            selected_full_sample_ids.append(self.feature_sample_indices[chosen_local_idx])

        selected_full_sample_ids = list(dict.fromkeys(selected_full_sample_ids))
        if not selected_full_sample_ids:
            self.train_loader = None
            self.train_iterator = None
            return

        self.train_loader = self._build_subset_loader(selected_full_sample_ids, shuffle=True)
        self.train_iterator = iter(self.train_loader)

    def local_train(self, cur_round, memory_record_dic=None):
        if self.train_loader is None or self.model is None:
            return

        lr = self.args.lr * math.pow(self.args.lr_decay, cur_round - 1)
        if self.args.batch_or_epoch == 'epoch':
            iter_steps = self.args.local_step * len(self.train_loader)
        else:
            iter_steps = self.args.local_step

        self.model.train()
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]

        try:
            from bitsandbytes.optim import PagedAdamW8bit

            optimizer = PagedAdamW8bit(trainable_params, lr=lr)
        except ImportError:
            optimizer = torch.optim.AdamW(trainable_params, lr=lr)

        amp_dtype = getattr(self.args, 'amp_dtype', torch.float16)
        for _ in range(iter_steps):
            try:
                batch = next(self.train_iterator)
            except (StopIteration, TypeError):
                self.train_iterator = iter(self.train_loader)
                batch = next(self.train_iterator)

            batch = {k: v.to(self.device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
            with torch.amp.autocast('cuda', enabled=True, dtype=amp_dtype):
                outputs = self.model(**batch)
                loss = outputs.loss

            if torch.isfinite(loss):
                loss.backward()
                if self.args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(trainable_params, self.args.grad_clip)
                optimizer.step()

            optimizer.zero_grad()

        if memory_record_dic is not None:
            memory_record_dic[self.device.index] = {'max_mem': torch.cuda.max_memory_allocated(self.device)}

    def clear_model(self):
        self.model = None

        if not getattr(self.args, 'use_fedhds_unlearn', False):
            for attr in ('reduced_feature_list', 'cluster_labels', 'centroids', 'feature_sample_indices'):
                if hasattr(self, attr):
                    delattr(self, attr)

        gc.collect()
        torch.cuda.empty_cache()

    def pull(self, model):
        self.model = model
        if next(self.model.parameters()).device != self.device:
            self.model.to(self.device)
