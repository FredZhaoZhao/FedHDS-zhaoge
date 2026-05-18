import torch
import torch.utils
import os
import numpy as np
from torch.utils.data import DataLoader, Subset
from transformers import AutoTokenizer
from utils_data.default_tokens import DefaultToken
from utils_data.partition_data import *
from collections import Counter


def get_loaders(args, only_eval=False):
    """
    Return: list of train_loaders, eval_loader
    """
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    tokenizer.model_max_length = args.max_length
    special_tokens = dict()
    if tokenizer.pad_token is None:
        special_tokens["pad_token"] = DefaultToken.PAD_TOKEN.value
    if tokenizer.eos_token is None:
        special_tokens["eos_token"] = DefaultToken.EOS_TOKEN.value
    if tokenizer.bos_token is None:
        special_tokens["bos_token"] = DefaultToken.BOS_TOKEN.value
    if tokenizer.unk_token is None:
        special_tokens["unk_token"] = DefaultToken.UNK_TOKEN.value

    tokenizer.add_special_tokens(special_tokens)

    # Generation task
    if args.dataset in ['dolly']:
        from utils_data.llm_dataset import LLMDataset, LLMDataCollator
        import json
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        candidate_paths = []
        if getattr(args, 'data_path', ''):
            candidate_paths.append(os.path.expanduser(args.data_path))
        candidate_paths.append(os.path.join(project_root, 'data', 'databricks-dolly-15k.jsonl'))
        candidate_paths.append(os.path.join('data', 'databricks-dolly-15k.jsonl'))

        dolly_path = next((p for p in candidate_paths if os.path.exists(p)), None)
        if dolly_path is None:
            searched = '\n'.join(candidate_paths)
            raise FileNotFoundError(
                f"Cannot find Dolly dataset file. Searched:\n{searched}\n"
                "Use --data_path to provide the file path explicitly."
            )

        if args.eval_metrics == 'none':
            raw_datasets = LLMDataset(dolly_path, tokenizer=tokenizer, generation=False)
        else:
            raw_datasets = LLMDataset(dolly_path, tokenizer=tokenizer, generation=True)

        data_collator = LLMDataCollator(tokenizer=tokenizer)

        import json
        train_ratio, val_ratio, _ = json.loads(args.split)

        # Subset sampling
        sample_size = int(len(raw_datasets) * args.data_sample)
        raw_datasets, _ = torch.utils.data.dataset.random_split(
            raw_datasets, [sample_size, len(raw_datasets) - sample_size]
        )

        if args.zeroshot:
            y_all = np.array([item['categories'] for item in raw_datasets])
            if '[' in str(args.zerotask):
                zerotasks = json.loads(args.zerotask)
                index_eval = []
                for t in zerotasks:
                    index_eval.extend(np.where(y_all == int(t))[0])
                index_eval = np.array(index_eval)
            else:
                index_eval = np.where(y_all == int(args.zerotask))[0]

            index_train = np.delete(np.arange(len(y_all)), index_eval)
            raw_datasets_list = list(raw_datasets)
            train_set = [raw_datasets_list[i] for i in index_train]
            eval_set = [raw_datasets_list[i] for i in index_eval]
            y_train = np.array([item['categories'] for item in train_set])
        else:
            train_len = int(len(raw_datasets) * train_ratio)
            val_len = int(len(raw_datasets) * val_ratio)
            test_len = len(raw_datasets) - train_len - val_len
            train_set, val_set, eval_set = torch.utils.data.dataset.random_split(
                raw_datasets, [train_len, val_len, test_len]
            )
            y_train = np.array([item['categories'] for item in train_set])

        counter = Counter(y_train)
        noniid_flag = str(args.iid)  # convert to string for safe parsing

        # --- Data partitioning logic ---
        if 'dir' in noniid_flag:
            # Dirichlet distribution for Non-IID partition, e.g. --iid dir0.5
            alpha_val = float(noniid_flag.replace('dir', ''))
            print(f">>> Using Dirichlet partition, alpha = {alpha_val}")
            split_dic = partition_idx_labeldir(y_train, n_parties=args.num_clients, alpha=alpha_val,
                                               num_classes=len(counter))
            split_trainsets = []
            for client_id, sample_indices in split_dic.items():
                split_trainsets.append(Subset(train_set, indices=sample_indices))
        else:
            # Traditional IID / Non-IID switch
            try:
                # float string to int: 1.0 → 1, avoids ValueError on "1.0"
                noniid_val = int(float(noniid_flag))
            except ValueError:
                noniid_val = 0

            if noniid_val == 0:
                print(">>> Using uniform (IID) partition")
                n_parts = [int(len(train_set) / args.num_clients) for _ in range(args.num_clients - 1)]
                n_parts.append(len(train_set) - sum(n_parts))
                split_trainsets = torch.utils.data.dataset.random_split(train_set, n_parts)
            else:
                print(f">>> Using label-based Non-IID partition, {noniid_val} labels per client")
                split_dic = partition_idx_labelnoniid(y_train, n_parties=args.num_clients, label_num=noniid_val,
                                                      num_classes=len(counter))
                split_trainsets = []
                for client_id, sample_indices in split_dic.items():
                    split_trainsets.append(Subset(train_set, indices=sample_indices))
        # --- End partitioning logic ---

        list_train_loader = [
            DataLoader(
                subset, shuffle=True, batch_size=args.batch_size, collate_fn=data_collator
            ) for subset in split_trainsets
        ]
        eval_loader = DataLoader(
            eval_set, batch_size=args.batch_size, collate_fn=data_collator
        )
    elif args.dataset in ['instruct']:
        from utils_data.natural_instruction_loader import get_instruction_dataset
        list_train_loader, eval_loader = get_instruction_dataset(args, tokenizer, only_eval=only_eval)
    else:
        raise AttributeError(f'dataset {args.dataset} not implemented')
    return list_train_loader, eval_loader, tokenizer
