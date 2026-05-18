import gc

import numpy as np
import torch


def resolve_amp_dtype(requested="auto"):
    requested = str(requested).lower()

    if requested == "fp16":
        return torch.float16

    if requested == "bf16":
        checker = getattr(torch.cuda, "is_bf16_supported", None)
        if torch.cuda.is_available() and callable(checker) and checker():
            return torch.bfloat16
        return torch.float16

    if requested != "auto":
        raise ValueError(f"Unsupported amp dtype '{requested}'. Use auto, bf16, or fp16.")

    checker = getattr(torch.cuda, "is_bf16_supported", None)
    if torch.cuda.is_available() and callable(checker) and checker():
        return torch.bfloat16
    return torch.float16


def get_flatten_features(model, data_loader, args):
    model.eval()
    features = []
    device = next(model.parameters()).device

    print(f"[DEBUG] Extracting features on {device}")

    with torch.no_grad():
        for batch in data_loader:
            if "labels" in batch:
                batch.pop("labels")

            mask = batch.get("attention_mask", None)
            batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
            outputs = model(**batch, output_hidden_states=True, return_dict=True)
            hidden_states = outputs.hidden_states[-2]

            if mask is not None:
                mask_float = mask.to(device).unsqueeze(-1).expand(hidden_states.size()).float()
                sum_embeddings = torch.sum(hidden_states * mask_float, dim=1)
                sum_mask = torch.clamp(mask_float.sum(dim=1), min=1e-9)
                pooled = (sum_embeddings / sum_mask).detach().cpu()
            else:
                pooled = hidden_states.mean(dim=1).detach().cpu()

            features.append(pooled)

            del outputs, hidden_states
            torch.cuda.empty_cache()

    if not features:
        return np.array([])

    final_features = torch.cat(features, dim=0).float().numpy()
    gc.collect()
    return final_features


def clustering(features, args):
    if torch.is_tensor(features):
        features = features.numpy()

    cluster_labels, centroids = _cluster(features, args)
    return cluster_labels, centroids, features


def _cluster(features, args):
    n_samples = features.shape[0]
    if n_samples == 0:
        return np.array([], dtype=int), {}
    if n_samples == 1:
        return np.array([0], dtype=int), {0: features[0]}

    clustering_method = getattr(args, 'clustering', 'kmeans').lower()

    if clustering_method == 'kmeans':
        from sklearn.cluster import KMeans

        n_clusters = getattr(args, 'n_cluster', 5)
        actual_clusters = min(n_clusters, n_samples)
        clusterer = KMeans(
            n_clusters=actual_clusters,
            max_iter=1000,
            n_init='auto',
            init='k-means++',
            random_state=42,
        )
        cluster_labels = clusterer.fit_predict(features)
        centroids = {cluster_id: center for cluster_id, center in enumerate(clusterer.cluster_centers_)}
        return cluster_labels, centroids

    if clustering_method == 'hdbscan':
        try:
            from hdbscan import HDBSCAN

            min_cluster = getattr(args, 'min_cluster', 2)
            actual_min_size = max(2, min(min_cluster, max(2, n_samples // 2)))
            clusterer = HDBSCAN(min_cluster_size=actual_min_size, allow_single_cluster=True)
            cluster_labels = clusterer.fit_predict(features)

            centroids = {}
            for label in np.unique(cluster_labels):
                if label == -1:
                    continue
                mask = cluster_labels == label
                centroids[int(label)] = features[mask].mean(axis=0)

            return cluster_labels, centroids
        except ImportError:
            print("[Error] Please install hdbscan or set --clustering=kmeans.")
            fallback_center = features.mean(axis=0)
            return np.zeros(n_samples, dtype=int), {0: fallback_center}

    raise ValueError(f"Unsupported clustering method: {clustering_method}")
