import argparse
import os
import random
import time

import numpy as np
import torch

from m_utils import resolve_amp_dtype

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # federation
    parser.add_argument('--num_clients', type=int, default=20)
    parser.add_argument('-k', '--k', dest='k', type=float, default=0.05)
    parser.add_argument('--rounds', type=int, default=40)
    parser.add_argument('--batch_or_epoch', type=str, default='batch', choices=['epoch', 'batch'])
    parser.add_argument('--local_step', type=int, default=1)
    parser.add_argument('--equal_weight', default=False, action='store_true')

    # data
    parser.add_argument('--dataset', type=str, default='dolly')
    parser.add_argument('--data_path', type=str, default='')
    parser.add_argument('--data_sample', type=float, default=1.0)
    parser.add_argument('--iid', type=str, default='0')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--max_length', type=int, default=128)
    parser.add_argument('--zeroshot', default=True, action='store_true')
    parser.add_argument('--zerotask', default='7', type=str)
    parser.add_argument('--split', type=str, default='[0.98, 0.01, 0.01]')
    parser.add_argument('--train_eval_ratio', default='[0.99, 0.01]', type=str)
    parser.add_argument('--use_prompts', default=False, action='store_true')

    # filtering (FedHDS)
    parser.add_argument('--filtering', action='store_true', default=False)
    parser.add_argument('--feature_layer', default='-1', type=str)
    parser.add_argument('--compound_dim', default=2, type=int)
    parser.add_argument('--feature_token', default='avg', type=str, choices=['avg', 'last'])
    parser.add_argument('--clustering_score', default='ch', type=str, choices=['ch', 'sc', 'db'])
    parser.add_argument('--clustering', type=str, default='kmeans', choices=['kmeans', 'hdbscan'])
    parser.add_argument('--n_cluster', type=int, default=7)
    parser.add_argument('--kernel_ratio', type=float, default=1.0)
    parser.add_argument('--filtering_model', type=str, default='same')
    parser.add_argument('--filtering_sample_limit', type=int, default=50)
    parser.add_argument('--dp_noise', type=float, default=0.0)
    parser.add_argument('--min_cluster', type=int, default=2)

    # model
    parser.add_argument('--model', type=str, default='Qwen/Qwen2-0.5B')
    parser.add_argument('--peft', action='store_true', default=False)
    parser.add_argument('--peft_method', default='lora', type=str, choices=['lora', 'prefix', 'p-tuning', 'prompt'])

    # training
    parser.add_argument('--optimizer', default='adam', choices=['adam', 'sgd'])
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--lr_decay', type=float, default=1.0)
    parser.add_argument('--grad_clip', type=float, default=-100.0)
    parser.add_argument('--amp_dtype', type=str, default='auto', choices=['auto', 'bf16', 'fp16'])

    # environment
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--log', default=False, action='store_true')
    parser.add_argument('--log_root', default='')
    parser.add_argument('--seed', default=42, type=int)

    # evaluation
    parser.add_argument('--eval_metrics', default='none', type=str)
    parser.add_argument('--generate_eval', default='rouge', type=str, choices=['rouge', 'bleu'])
    parser.add_argument('--eval_subsampling', default=False, action='store_true')
    parser.add_argument('--full_evaluation', default=False, action='store_true')
    parser.add_argument('--start_eval_epoch', default=30, type=int)
    parser.add_argument('--eval_interval', default=1, type=int)
    parser.add_argument('--loss', default=False, action='store_true')

    # ckpt
    parser.add_argument('--save', default=False, action='store_true')

    # FedHDS unlearning
    parser.add_argument('--use_fedhds_unlearn', default=False, action='store_true')
    parser.add_argument('--use_retained_hessian', default=False, action='store_true')
    parser.add_argument('--forget_client_idx', type=int, default=-1)
    parser.add_argument('--hessian_ref_per_client', type=int, default=2)
    parser.add_argument('--unlearn_eta', type=float, default=1.0)
    parser.add_argument('--unlearn_max_update_norm', type=float, default=0.0)
    parser.add_argument('--unlearn_num_steps', type=int, default=1)
    parser.add_argument('--unlearn_ref_loss_guard_ratio', type=float, default=0.0)
    parser.add_argument('--unlearn_global_loss_guard_max', type=float, default=0.0)
    parser.add_argument('--unlearn_update_sign', type=str, default='positive', choices=['positive', 'negative', 'auto'])
    parser.add_argument('--unlearn_direction_check', default=False, action='store_true')
    parser.add_argument('--unlearn_forget_loss_guard', default=False, action='store_true')
    parser.add_argument('--unlearn_forget_loss_min_gain', type=float, default=0.0)
    parser.add_argument('--unlearn_forget_loss_tolerance', type=float, default=0.0)
    parser.add_argument('--unlearn_forget_guard_sample_size', type=int, default=0)
    parser.add_argument('--lissa_depth', type=int, default=1)
    parser.add_argument('--lissa_damping', type=float, default=0.01)
    parser.add_argument('--unlearn_grad_sample_size', type=int, default=0)

    # retrain baseline
    parser.add_argument('--retrain_exclude_client', type=int, default=-1)

    args = parser.parse_args()
    time_stamp = str(time.time())

    supported_datasets = {'dolly', 'instruct'}
    if args.dataset not in supported_datasets:
        raise ValueError(f"Unsupported dataset '{args.dataset}'. Supported datasets: {sorted(supported_datasets)}")
    if args.filtering_model != 'same':
        raise ValueError(
            "The legacy --filtering_model!=same path has been removed. "
            "FedHDS filtering now always uses the training model."
        )
    if args.use_retained_hessian and not args.use_fedhds_unlearn:
        raise ValueError("--use_retained_hessian requires --use_fedhds_unlearn.")
    if args.use_fedhds_unlearn and not (0 <= args.forget_client_idx < args.num_clients):
        raise ValueError("--forget_client_idx must be in [0, num_clients) when unlearning is enabled.")
    if args.hessian_ref_per_client < 1:
        raise ValueError("--hessian_ref_per_client must be at least 1.")
    if args.unlearn_grad_sample_size < 0:
        raise ValueError("--unlearn_grad_sample_size must be non-negative.")
    if args.unlearn_num_steps < 1:
        raise ValueError("--unlearn_num_steps must be at least 1.")
    if args.unlearn_ref_loss_guard_ratio < 0:
        raise ValueError("--unlearn_ref_loss_guard_ratio must be non-negative.")
    if args.unlearn_global_loss_guard_max < 0:
        raise ValueError("--unlearn_global_loss_guard_max must be non-negative.")
    if args.unlearn_forget_loss_min_gain < 0:
        raise ValueError("--unlearn_forget_loss_min_gain must be non-negative.")
    if args.unlearn_forget_loss_tolerance < 0:
        raise ValueError("--unlearn_forget_loss_tolerance must be non-negative.")
    if args.unlearn_forget_guard_sample_size < 0:
        raise ValueError("--unlearn_forget_guard_sample_size must be non-negative.")
    if args.retrain_exclude_client >= args.num_clients:
        raise ValueError("--retrain_exclude_client must be in [0, num_clients).")

    args.amp_dtype = resolve_amp_dtype(args.amp_dtype)

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)

    from client import Client
    from server import Server
    from utils_data.load_data import get_loaders

    setup_seed(args.seed)
    memory_record_dic = {}

    if args.batch_or_epoch == 'epoch':
        print(f"[CONFIG] Local training mode: epoch, local_step={args.local_step} "
              f"(each selected client runs {args.local_step} local epoch(s) per round)")
    else:
        print(f"[CONFIG] Local training mode: batch, local_step={args.local_step} "
              f"(each selected client runs {args.local_step} local batch(es) per round)")

    start_time = time.time()
    list_train_loader, eval_loader, _ = get_loaders(args)
    print(f'Data loaded in {time.time() - start_time:.2f}s. {len(eval_loader.dataset)} eval samples.')

    model_name_clean = args.model.replace('/', '_')
    base_dir = 'exp_lrdecay' if args.lr_decay != 1.0 else 'exp'
    log_dir = os.path.join(
        base_dir,
        args.optimizer,
        f"{args.dataset}-iid{args.iid}",
        f"zeroshot{args.zerotask if args.zeroshot else 'fewshot'}",
        model_name_clean,
        str(args.lr),
        f"step{args.local_step}",
        f"seed{args.seed}",
        time_stamp,
    )
    if args.log_root:
        log_dir = os.path.join(args.log_root, log_dir)
    if args.log:
        os.makedirs(log_dir, exist_ok=True)

    server = Server(args, eval_loader=eval_loader, log_dir=log_dir)
    client_list = [Client(idx, args, train_loader) for idx, train_loader in enumerate(list_train_loader)]

    retrain_heldout_client = None
    if args.retrain_exclude_client >= 0:
        retrain_heldout_client = client_list[args.retrain_exclude_client]
        list_train_loader = [
            loader for i, loader in enumerate(list_train_loader)
            if i != args.retrain_exclude_client
        ]
        args.num_clients = len(list_train_loader)
        client_list = [Client(idx, args, train_loader) for idx, train_loader in enumerate(list_train_loader)]
        print(f"[RETRAIN] Client {args.retrain_exclude_client} excluded from training. "
              f"Training with {args.num_clients} clients.")

    client_indices_rounds = []
    for _ in range(args.rounds):
        available = list(range(args.num_clients))
        num_to_select = max(1, int(args.num_clients * args.k))
        client_indices_rounds.append(np.random.choice(available, size=num_to_select, replace=False))

    for r in range(1, args.rounds + 1):
        print(f"\n--- Start Training Round {r} ---")
        selected_indices = client_indices_rounds[r - 1]
        selected_clients = [client_list[i] for i in selected_indices]

        server.prepare_aggregate()

        if args.filtering:
            staged_clusterid_clientid_centroid = []

            for client in selected_clients:
                client.selected_clusters = []
                client.pull(server.model)
                for cluster_id, centroid in client.calculated_cluster_center():
                    staged_clusterid_clientid_centroid.append((cluster_id, client.idx, centroid))

            center_list = np.array([item[2] for item in staged_clusterid_clientid_centroid])
            if len(center_list) > 0:
                if len(center_list) < 2:
                    for cluster_id, client_idx, _ in staged_clusterid_clientid_centroid:
                        client_list[client_idx].selected_clusters.append(cluster_id)
                else:
                    from hdbscan import HDBSCAN

                    clusterer = HDBSCAN(min_cluster_size=2, allow_single_cluster=True)
                    labels = clusterer.fit_predict(center_list)
                    if (labels == -1).all():
                        for cluster_id, client_idx, _ in staged_clusterid_clientid_centroid:
                            client_list[client_idx].selected_clusters.append(cluster_id)
                    else:
                        for label in range(labels.max() + 1):
                            idx_in_cluster = np.where(labels == label)[0]
                            if len(idx_in_cluster) == 0:
                                continue
                            rep_idx = idx_in_cluster[0]
                            cluster_id, client_idx, _ = staged_clusterid_clientid_centroid[rep_idx]
                            client_list[client_idx].selected_clusters.append(cluster_id)

                        noise_indices = np.where(labels == -1)[0]
                        for noise_idx in noise_indices:
                            cluster_id, client_idx, _ = staged_clusterid_clientid_centroid[noise_idx]
                            client_list[client_idx].selected_clusters.append(cluster_id)

            for client in selected_clients:
                client.build_training_set_with_precalculated_clusters()
        else:
            for client in selected_clients:
                client.use_full_training_set()

        active_clients = [client for client in selected_clients if client.train_loader is not None]

        for client in active_clients:
            server.load_lora_state_dict(server.global_lora_state_dict)
            client.pull(server.model)

            start_client = time.time()
            client.local_train(cur_round=r, memory_record_dic=memory_record_dic)
            elapsed = time.time() - start_client
            server.train_time.append([elapsed, len(client.train_loader.dataset), client.idx, r])

            if client.model is not None:
                local_lora_state = server.get_lora_state_dict()
                client_weight = server.get_client_weight(client, active_clients)
                server.online_aggregate(local_lora_state, client_weight)

        server.finish_aggregate()

        for client in selected_clients:
            client.clear_model()

        torch.cuda.empty_cache()
        res = server.eval_loss(cur_round=r)
        print(f"[PROGRESS] Round {r} Loss: {res:.4f}")

    if args.save:
        server.save_checkpoint(label='post_train')

    if retrain_heldout_client is not None:
        print(f"\n>>> [RETRAIN BASELINE] Evaluating on excluded Client {args.retrain_exclude_client} <<<")
        torch.cuda.empty_cache()

        heldout_loader = retrain_heldout_client.get_full_train_loader(shuffle=False)
        heldout_loss = server.eval_loss_on_loader(
            heldout_loader,
            desc=f'Retrain Heldout Client {args.retrain_exclude_client} Eval',
        )
        server.experiment_metrics.update({
            'retrain_exclude_client_idx': args.retrain_exclude_client,
            'forget_client_idx': args.retrain_exclude_client,
            'forget_client_num_samples': len(heldout_loader.dataset),
            'forget_client_loss_before_unlearning': heldout_loss,
            'forget_client_loss_after_unlearning': heldout_loss,
            'global_loss_before_unlearning': server.eval_history[-1] if server.eval_history else None,
            'global_loss_after_unlearning': server.eval_history[-1] if server.eval_history else None,
            'hessian_reference_mode': 'retrain-from-scratch',
            'effective_hessian_reference_mode': 'retrain-from-scratch',
            'unlearning_time_sec': 0,
            'unlearn_steps_accepted': 0,
            'unlearn_steps_attempted': 0,
            'unlearn_guard_stop_reason': 'retrain_baseline',
        })
        print(f"[RETRAIN BASELINE] Heldout client {args.retrain_exclude_client} loss: {heldout_loss:.4f}")

    elif args.use_fedhds_unlearn and args.forget_client_idx >= 0:
        print(f"\n>>> Starting FedHDS Precise Unlearning for Client {args.forget_client_idx} <<<")
        torch.cuda.empty_cache()

        forget_eval_loader = client_list[args.forget_client_idx].get_full_train_loader(shuffle=False)
        forget_loss_before = server.eval_loss_on_loader(
            forget_eval_loader,
            desc=f'Forget Client {args.forget_client_idx} Pre-Unlearn Eval',
        )
        server.experiment_metrics.update({
            'global_loss_before_unlearning': server.eval_history[-1] if server.eval_history else None,
            'forget_client_idx': args.forget_client_idx,
            'forget_client_num_samples': len(forget_eval_loader.dataset),
            'forget_client_loss_before_unlearning': forget_loss_before,
            'hessian_reference_mode': 'retained-set' if args.use_retained_hessian else 'forget-batch',
        })
        print(f"[UNLEARN CHECK] Forget client loss before unlearning: {forget_loss_before:.4f}")

        unlearn_start_time = time.time()
        server.apply_fedhds_unlearning(
            forget_client_idx=args.forget_client_idx,
            client_list=client_list,
        )
        unlearn_elapsed = time.time() - unlearn_start_time
        server.experiment_metrics['unlearning_time_sec'] = unlearn_elapsed
        print(f"[UNLEARN CHECK] Unlearning time: {unlearn_elapsed:.2f}s")

        forget_loss_after = server.eval_loss_on_loader(
            forget_eval_loader,
            desc=f'Forget Client {args.forget_client_idx} Post-Unlearn Eval',
        )
        server.experiment_metrics['forget_client_loss_after_unlearning'] = forget_loss_after
        print(f"[UNLEARN CHECK] Forget client loss after unlearning: {forget_loss_after:.4f}")

        print(">>> Post-Unlearning Performance Check <<<")
        unlearn_eval_res = server.eval_loss(cur_round="Unlearn_Final")
        server.experiment_metrics['global_loss_after_unlearning'] = unlearn_eval_res
        print(f"[UNLEARN CHECK] Global Loss after Newton-step: {unlearn_eval_res:.4f}")

    if args.save:
        server.save_checkpoint(label='post_unlearn')
        torch.cuda.empty_cache()

        gen_global = server.eval_generation(
            server.eval_loader,
            desc='Gen-Eval Global',
            max_samples=200,
        )
        server.experiment_metrics['global_rouge_l'] = gen_global['rouge_l']
        server.experiment_metrics['global_bleu'] = gen_global['bleu']

        forget_client_idx_eval = (
            args.retrain_exclude_client
            if args.retrain_exclude_client >= 0
            else args.forget_client_idx
        )
        if forget_client_idx_eval >= 0:
            # Use the original (pre-exclusion) client reference for retrain path
            if retrain_heldout_client is not None:
                forget_gen_loader = retrain_heldout_client.get_full_train_loader(shuffle=False)
            else:
                forget_gen_loader = client_list[forget_client_idx_eval].get_full_train_loader(shuffle=False)

            gen_forget = server.eval_generation(
                forget_gen_loader,
                desc=f'Gen-Eval Forget Client {forget_client_idx_eval}',
                max_samples=200,
            )
            server.experiment_metrics['forget_client_rouge_l'] = gen_forget['rouge_l']
            server.experiment_metrics['forget_client_bleu'] = gen_forget['bleu']

    server.save_results(
        eval_history=server.eval_history,
        train_time=server.train_time,
        rounds_completed=args.rounds,
        num_clients=args.num_clients,
    )

    print("\n[FINISH] All Rounds and Unlearning Procedure Completed.")
