import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cli_config import build_arg_parser


def _str_to_bool(value):
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n", ""}:
        return False
    raise ValueError(f"Cannot parse boolean value from {value!r}")


def parse_saved_config(config, parser=None):
    parser = parser or build_arg_parser()
    defaults = parser.parse_args([])
    parsed = vars(defaults).copy()

    for action in parser._actions:
        dest = getattr(action, "dest", None)
        if not dest or dest == "help" or dest not in config:
            continue

        raw_value = config[dest]
        if raw_value in (None, "None"):
            parsed[dest] = None
            continue

        if action.__class__.__name__ == "_StoreTrueAction":
            parsed[dest] = _str_to_bool(raw_value)
            continue

        if action.__class__.__name__ == "_StoreFalseAction":
            parsed[dest] = not _str_to_bool(raw_value)
            continue

        if dest == "amp_dtype" and str(raw_value).startswith("torch."):
            parsed[dest] = "bf16" if "bfloat16" in str(raw_value) else "fp16"
            continue

        if action.type is not None:
            parsed[dest] = action.type(raw_value)
        else:
            parsed[dest] = raw_value

    return SimpleNamespace(**parsed)


def roc_auc_from_scores(labels, scores):
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    pos_mask = labels == 1
    neg_mask = labels == 0
    pos_scores = scores[pos_mask]
    neg_scores = scores[neg_mask]
    if len(pos_scores) == 0 or len(neg_scores) == 0:
        return float("nan")

    greater = 0.0
    ties = 0.0
    for pos_score in pos_scores:
        greater += np.sum(pos_score > neg_scores)
        ties += np.sum(pos_score == neg_scores)
    return float((greater + 0.5 * ties) / (len(pos_scores) * len(neg_scores)))


def tpr_at_fpr(labels, scores, target_fpr):
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    order = np.argsort(-scores)
    labels = labels[order]
    scores = scores[order]

    total_pos = np.sum(labels == 1)
    total_neg = np.sum(labels == 0)
    if total_pos == 0 or total_neg == 0:
        return float("nan")

    tp = 0
    fp = 0
    best_tpr = 0.0
    for label in labels:
        if label == 1:
            tp += 1
        else:
            fp += 1
        current_fpr = fp / total_neg
        current_tpr = tp / total_pos
        if current_fpr <= target_fpr:
            best_tpr = max(best_tpr, current_tpr)
        else:
            break
    return float(best_tpr)


def sample_category_matched_nonmembers(member_categories, candidate_categories, sample_size, rng=None):
    rng = rng or random.Random(0)
    member_categories = list(member_categories)
    candidate_categories = list(candidate_categories)
    sample_size = min(sample_size, len(candidate_categories))
    if sample_size <= 0:
        return []

    candidate_by_category = defaultdict(list)
    for idx, category in enumerate(candidate_categories):
        candidate_by_category[category].append(idx)

    member_counts = Counter(member_categories)
    selected = []

    for category, count in member_counts.items():
        pool = candidate_by_category.get(category, [])
        if not pool:
            continue
        rng.shuffle(pool)
        take = min(count, len(pool))
        selected.extend(pool[:take])
        candidate_by_category[category] = pool[take:]

    if len(selected) < sample_size:
        remaining = []
        for pool in candidate_by_category.values():
            remaining.extend(pool)
        rng.shuffle(remaining)
        selected.extend(remaining[: sample_size - len(selected)])

    return selected[:sample_size]


def _load_result_json(result_path):
    with open(result_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _save_result_json(result_path, payload):
    with open(result_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def _select_checkpoint_path(run_dir, checkpoint_label="auto"):
    candidates = []
    if checkpoint_label == "auto":
        candidates = ["lora_state_post_unlearn.pt", "lora_state_post_train.pt"]
    else:
        candidates = [f"lora_state_{checkpoint_label}.pt"]

    for candidate in candidates:
        candidate_path = run_dir / candidate
        if candidate_path.exists():
            return candidate_path
    return None


def _setup_seed(seed):
    import torch

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def _rebuild_clients(args):
    from client import Client
    from m_utils import resolve_amp_dtype
    from utils_data.load_data import get_loaders

    args.amp_dtype = resolve_amp_dtype(args.amp_dtype)
    _setup_seed(args.seed)
    list_train_loader, eval_loader, _ = get_loaders(args)
    client_list = [Client(idx, args, train_loader) for idx, train_loader in enumerate(list_train_loader)]
    return client_list, eval_loader


def _build_member_nonmember_examples(client_list, forget_client_idx, sample_size=None, rng=None):
    rng = rng or random.Random(0)
    target_client = client_list[forget_client_idx]
    member_examples = list(target_client.full_train_dataset)

    candidate_examples = []
    for client in client_list:
        if client.idx == forget_client_idx:
            continue
        candidate_examples.extend(list(client.full_train_dataset))

    member_categories = [example["categories"] for example in member_examples]
    candidate_categories = [example["categories"] for example in candidate_examples]
    target_size = len(member_examples) if sample_size is None else min(sample_size, len(member_examples))
    selected_indices = sample_category_matched_nonmembers(
        member_categories=member_categories,
        candidate_categories=candidate_categories,
        sample_size=target_size,
        rng=rng,
    )
    nonmember_examples = [candidate_examples[idx] for idx in selected_indices]
    return member_examples[:target_size], nonmember_examples


def _build_loader_from_examples(target_client, examples):
    from torch.utils.data import DataLoader

    return DataLoader(
        examples,
        shuffle=False,
        batch_size=target_client.batch_size,
        collate_fn=target_client.full_collate_fn,
        drop_last=False,
    )


def _compute_per_example_losses(server, data_loader):
    import torch
    import torch.nn.functional as F

    model = server.model
    device = server.device
    losses = []
    was_training = model.training
    model.eval()

    with torch.no_grad():
        for batch in data_loader:
            batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
            labels = batch["labels"]
            with server._autocast_context():
                outputs = model(**batch)
                logits = outputs.logits.float()

            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            token_losses = F.cross_entropy(
                shift_logits.transpose(1, 2),
                shift_labels,
                reduction="none",
                ignore_index=-100,
            )
            valid_mask = shift_labels.ne(-100)
            denom = valid_mask.sum(dim=1).clamp(min=1)
            batch_losses = (token_losses * valid_mask).sum(dim=1) / denom
            losses.extend(batch_losses.detach().cpu().tolist())

    if was_training:
        model.train()
    return losses


def _coerce_nonnegative_int(candidate):
    try:
        idx = int(candidate)
    except (TypeError, ValueError):
        return None
    return idx if idx >= 0 else None


def _infer_forget_client_idx_from_sibling_runs(result_path):
    if result_path is None:
        return None

    seed_root = Path(result_path).resolve().parent.parent
    if not seed_root.exists():
        return None

    inferred = set()
    for sibling_path in sorted(seed_root.rglob("final_results.json")):
        if sibling_path.resolve() == Path(result_path).resolve():
            continue

        sibling_result = _load_result_json(sibling_path)
        sibling_metrics = sibling_result.get("experiment_metrics") or {}
        sibling_candidates = [
            sibling_metrics.get("forget_client_idx"),
            sibling_metrics.get("retrain_exclude_client_idx"),
            sibling_metrics.get("retrain_exclude_client"),
        ]
        sibling_args = parse_saved_config(sibling_result.get("config", {}), build_arg_parser())
        sibling_candidates.extend(
            [
                getattr(sibling_args, "forget_client_idx", None),
                getattr(sibling_args, "retrain_exclude_client", None),
            ]
        )

        for candidate in sibling_candidates:
            idx = _coerce_nonnegative_int(candidate)
            if idx is not None:
                inferred.add(idx)

    if len(inferred) == 1:
        return inferred.pop()
    return None


def resolve_forget_client_idx(args, result, result_path=None, fallback_forget_client_idx=None):
    metrics = result.get("experiment_metrics") or {}
    candidates = [
        metrics.get("forget_client_idx"),
        metrics.get("retrain_exclude_client_idx"),
        fallback_forget_client_idx,
        getattr(args, "forget_client_idx", None),
        getattr(args, "retrain_exclude_client", None),
    ]
    for candidate in candidates:
        idx = _coerce_nonnegative_int(candidate)
        if idx is not None:
            return idx

    sibling_idx = _infer_forget_client_idx_from_sibling_runs(result_path)
    if sibling_idx is not None:
        return sibling_idx

    raise ValueError("Could not resolve a valid forget client index for MIA evaluation.")


def evaluate_run(result_path, checkpoint_label="auto", sample_limit=0, fallback_forget_client_idx=None):
    import torch
    from server import Server

    result_path = Path(result_path).resolve()
    result = _load_result_json(result_path)
    args = parse_saved_config(result.get("config", {}), build_arg_parser())

    client_list, eval_loader = _rebuild_clients(args)
    checkpoint_path = _select_checkpoint_path(result_path.parent, checkpoint_label=checkpoint_label)
    if checkpoint_path is None:
        raise FileNotFoundError(
            f"Could not find a LoRA checkpoint under {result_path.parent} "
            f"for checkpoint_label={checkpoint_label!r}."
        )

    forget_client_idx = resolve_forget_client_idx(
        args,
        result,
        result_path=result_path,
        fallback_forget_client_idx=fallback_forget_client_idx,
    )
    target_client = client_list[forget_client_idx]
    member_examples, nonmember_examples = _build_member_nonmember_examples(
        client_list=client_list,
        forget_client_idx=forget_client_idx,
        sample_size=sample_limit if sample_limit > 0 else None,
        rng=random.Random(args.seed),
    )

    member_loader = _build_loader_from_examples(target_client, member_examples)
    nonmember_loader = _build_loader_from_examples(target_client, nonmember_examples)

    server = Server(args, eval_loader=eval_loader, log_dir=str(result_path.parent))
    lora_state = torch.load(checkpoint_path, map_location="cpu")
    server.load_lora_state_dict(lora_state)

    member_losses = _compute_per_example_losses(server, member_loader)
    nonmember_losses = _compute_per_example_losses(server, nonmember_loader)
    membership_scores = [-loss for loss in member_losses] + [-loss for loss in nonmember_losses]
    labels = [1] * len(member_losses) + [0] * len(nonmember_losses)

    metrics = {
        "mia_checkpoint_path": str(checkpoint_path),
        "mia_member_count": len(member_losses),
        "mia_nonmember_count": len(nonmember_losses),
        "mia_score_type": "negative_loss",
        "mia_loss_auc": roc_auc_from_scores(labels, membership_scores),
        "mia_loss_tpr_at_fpr001": tpr_at_fpr(labels, membership_scores, 0.01),
        "mia_member_mean_loss": float(np.mean(member_losses)) if member_losses else float("nan"),
        "mia_nonmember_mean_loss": float(np.mean(nonmember_losses)) if nonmember_losses else float("nan"),
    }

    result.setdefault("experiment_metrics", {}).update(metrics)
    _save_result_json(result_path, result)
    return metrics


def _iter_result_paths(path_like):
    target = Path(path_like).resolve()
    if target.is_file():
        return [target]
    return sorted(target.rglob("final_results.json"))


def main():
    parser = argparse.ArgumentParser(description="Evaluate loss-based MIA and write metrics back to final_results.json.")
    parser.add_argument("target", help="A final_results.json file or a result root directory.")
    parser.add_argument(
        "--checkpoint-label",
        default="auto",
        help="Checkpoint label to load: auto, post_train, or post_unlearn.",
    )
    parser.add_argument(
        "--sample-limit",
        type=int,
        default=0,
        help="Optional cap for member/non-member set size. 0 means full forget-client size.",
    )
    parser.add_argument(
        "--forget-client-idx",
        type=int,
        default=-1,
        help="Optional fallback forget-client index for runs that do not record it explicitly.",
    )
    args = parser.parse_args()

    result_paths = _iter_result_paths(args.target)
    if not result_paths:
        raise FileNotFoundError(f"No final_results.json files found under {args.target}")

    for result_path in result_paths:
        metrics = evaluate_run(
            result_path=result_path,
            checkpoint_label=args.checkpoint_label,
            sample_limit=args.sample_limit,
            fallback_forget_client_idx=args.forget_client_idx,
        )
        print(
            f"[MIA] {result_path}: "
            f"AUC={metrics['mia_loss_auc']:.4f}, "
            f"TPR@FPR=1%={metrics['mia_loss_tpr_at_fpr001']:.4f}, "
            f"members={metrics['mia_member_count']}, nonmembers={metrics['mia_nonmember_count']}"
        )


if __name__ == "__main__":
    main()
