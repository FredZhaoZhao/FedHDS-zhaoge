import argparse


def build_arg_parser():
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
    parser.add_argument(
        '--unlearn_method',
        type=str,
        default='second_order',
        choices=['second_order', 'ga', 'ga_guarded'],
    )
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

    return parser
