import json
import os
from contextlib import contextmanager

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from evaluations import bleu_score, rouge_score
from utils_data.default_tokens import DefaultToken

os.environ["TRANSFORMERS_OFFLINE"] = "0"


class Server(object):
    def __init__(self, args, eval_loader, log_dir):
        self.args = args
        self.eval_loader = eval_loader
        self.log_dir = log_dir
        self.device = torch.device(f'cuda:{self.args.device}')

        self.tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
        self.tokenizer.model_max_length = self.args.max_length

        special_tokens = {}
        if self.tokenizer.pad_token is None:
            special_tokens["pad_token"] = DefaultToken.PAD_TOKEN.value
        if self.tokenizer.eos_token is None:
            special_tokens["eos_token"] = DefaultToken.EOS_TOKEN.value
        if self.tokenizer.bos_token is None:
            special_tokens["bos_token"] = DefaultToken.BOS_TOKEN.value
        if self.tokenizer.unk_token is None:
            special_tokens["unk_token"] = DefaultToken.UNK_TOKEN.value
        self.tokenizer.add_special_tokens(special_tokens)

        torch.cuda.empty_cache()

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=self.args.amp_dtype,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            args.model,
            device_map={"": self.args.device},
            output_hidden_states=True,
            torch_dtype=self.args.amp_dtype,
            trust_remote_code=True,
            quantization_config=quantization_config,
        )
        self.model = prepare_model_for_kbit_training(self.model)

        config = LoraConfig(
            r=8,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        self.model = get_peft_model(self.model, config)

        for name, param in self.model.named_parameters():
            if "lora" in name:
                param.requires_grad = True
                param.data = param.data.to(torch.float32)

        self.model.print_trainable_parameters()

        self.eval_loss_history = []
        self.eval_history = []
        self.train_time = []
        self.experiment_metrics = {}
        self.delta_storage = {}
        self.aggregate_buffer = {}
        self.aggregate_weight_total = 0.0
        self.global_lora_state_dict = self.get_lora_state_dict()

    def _autocast_context(self):
        return torch.amp.autocast('cuda', dtype=self.args.amp_dtype)

    def get_lora_state_dict(self):
        return {
            key: value.detach().clone().cpu()
            for key, value in self.model.state_dict().items()
            if "lora_" in key
        }

    def load_lora_state_dict(self, lora_state_dict):
        self.model.load_state_dict(lora_state_dict, strict=False)

    def prepare_aggregate(self):
        self.global_lora_state_dict = self.get_lora_state_dict()
        self.aggregate_buffer = {
            key: torch.zeros_like(value)
            for key, value in self.global_lora_state_dict.items()
        }
        self.aggregate_weight_total = 0.0

    def get_client_weight(self, client, active_client_list):
        if not active_client_list or client.train_loader is None:
            return 0.0

        if self.args.equal_weight:
            return 1.0 / len(active_client_list)

        total_samples = sum(len(active_client.train_loader.dataset) for active_client in active_client_list)
        if total_samples <= 0:
            return 0.0
        return len(client.train_loader.dataset) / total_samples

    def online_aggregate(self, client_lora_state, client_weight):
        if client_weight <= 0:
            return

        for key, value in self.aggregate_buffer.items():
            if key in client_lora_state:
                self.aggregate_buffer[key] = value + client_lora_state[key].detach().cpu() * client_weight
        self.aggregate_weight_total += client_weight

    def finish_aggregate(self):
        if not self.aggregate_buffer:
            return

        if self.aggregate_weight_total <= 0:
            self.load_lora_state_dict(self.global_lora_state_dict)
        else:
            if self.aggregate_weight_total != 1.0:
                for key in self.aggregate_buffer:
                    self.aggregate_buffer[key] /= self.aggregate_weight_total
            self.load_lora_state_dict(self.aggregate_buffer)

        self.aggregate_buffer = {}
        self.aggregate_weight_total = 0.0
        torch.cuda.empty_cache()

    def compute_hvp(self, loss, params, vec):
        with self._second_order_attention_context():
            grads = torch.autograd.grad(loss, params, create_graph=True, retain_graph=True, allow_unused=True)

            valid_grads = []
            valid_vec = []
            for grad, vec_item in zip(grads, vec):
                if grad is not None:
                    valid_grads.append(grad)
                    valid_vec.append(vec_item)

            if not valid_grads:
                return [torch.zeros_like(param) for param in params]

            grad_vec_dot = sum(torch.sum(grad * vec_item) for grad, vec_item in zip(valid_grads, valid_vec))
            hvp = torch.autograd.grad(grad_vec_dot, params, retain_graph=True, allow_unused=True)
        return [item if item is not None else torch.zeros_like(param) for item, param in zip(hvp, params)]

    @contextmanager
    def _second_order_attention_context(self):
        if not torch.cuda.is_available():
            yield
            return

        cuda_backend = torch.backends.cuda
        has_sdp_toggles = all(
            hasattr(cuda_backend, name)
            for name in (
                'flash_sdp_enabled',
                'mem_efficient_sdp_enabled',
                'math_sdp_enabled',
                'enable_flash_sdp',
                'enable_mem_efficient_sdp',
                'enable_math_sdp',
            )
        )

        if has_sdp_toggles:
            prev_flash = cuda_backend.flash_sdp_enabled()
            prev_mem_efficient = cuda_backend.mem_efficient_sdp_enabled()
            prev_math = cuda_backend.math_sdp_enabled()
            cuda_backend.enable_flash_sdp(False)
            cuda_backend.enable_mem_efficient_sdp(False)
            cuda_backend.enable_math_sdp(True)
            try:
                yield
            finally:
                cuda_backend.enable_flash_sdp(prev_flash)
                cuda_backend.enable_mem_efficient_sdp(prev_mem_efficient)
                cuda_backend.enable_math_sdp(prev_math)
            return

        try:
            from torch.nn.attention import SDPBackend, sdpa_kernel

            with sdpa_kernel([SDPBackend.MATH]):
                yield
        except (ImportError, AttributeError):
            with torch.backends.cuda.sdp_kernel(
                enable_flash=False,
                enable_mem_efficient=False,
                enable_math=True,
            ):
                yield

    def _compute_average_forget_gradients(self, data_loader, params):
        grad_sums = [torch.zeros_like(param, dtype=torch.float32, device=param.device) for param in params]
        total_samples = 0
        reference_batch = None

        for batch in data_loader:
            batch = {k: v.to(self.device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
            with self._second_order_attention_context(), self._autocast_context():
                outputs = self.model(**batch)
                loss = outputs.loss

            if not torch.isfinite(loss):
                continue

            batch_size = batch['input_ids'].size(0)
            grads = torch.autograd.grad(loss, params, retain_graph=False, allow_unused=True)
            grads = [grad if grad is not None else torch.zeros_like(param) for grad, param in zip(grads, params)]

            for idx, grad in enumerate(grads):
                grad_sums[idx].add_(grad.detach().to(torch.float32), alpha=batch_size)

            total_samples += batch_size
            if reference_batch is None:
                reference_batch = {key: value.detach().clone() for key, value in batch.items()}

        if total_samples == 0 or reference_batch is None:
            return None, None

        avg_grads = [grad_sum / total_samples for grad_sum in grad_sums]
        return avg_grads, reference_batch

    def _tensor_list_stats(self, tensors):
        squared_norm_sum = 0.0
        max_abs = 0.0
        nonzero_tensors = 0

        for tensor in tensors:
            detached = tensor.detach().float()
            tensor_norm = torch.norm(detached).item()
            tensor_max_abs = torch.max(torch.abs(detached)).item() if detached.numel() > 0 else 0.0
            squared_norm_sum += tensor_norm * tensor_norm
            max_abs = max(max_abs, tensor_max_abs)
            if tensor_max_abs > 0:
                nonzero_tensors += 1

        return {
            'l2_norm': squared_norm_sum ** 0.5,
            'max_abs': max_abs,
            'nonzero_tensors': nonzero_tensors,
            'num_tensors': len(tensors),
        }

    def _param_delta_stats(self, params, params_before):
        deltas = [
            param.detach().float() - before.float().to(param.device)
            for param, before in zip(params, params_before)
        ]
        return self._tensor_list_stats(deltas)

    def build_retained_reference_loader(self, forget_client_idx, client_list):
        reference_examples = []
        retained_client = None
        sample_size = max(1, getattr(self.args, 'hessian_ref_per_client', 1))

        for client in client_list:
            if client.idx == forget_client_idx:
                continue

            if retained_client is None:
                retained_client = client

            reference_examples.extend(
                client.sample_reference_examples(sample_size=sample_size, shuffle=False)
            )

        if retained_client is None or not reference_examples:
            return None, 0

        reference_loader = DataLoader(
            reference_examples,
            shuffle=False,
            batch_size=retained_client.batch_size,
            collate_fn=retained_client.full_collate_fn,
            drop_last=False,
        )
        return reference_loader, len(reference_examples)

    def _compute_average_loss(self, data_loader):
        loss_sum = None
        total_samples = 0

        for batch in data_loader:
            batch = {k: v.to(self.device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
            with self._second_order_attention_context(), self._autocast_context():
                outputs = self.model(**batch)
                loss = outputs.loss

            if not torch.isfinite(loss):
                continue

            batch_size = batch['input_ids'].size(0)
            batch_loss = loss * batch_size
            loss_sum = batch_loss if loss_sum is None else loss_sum + batch_loss
            total_samples += batch_size

        if loss_sum is None or total_samples == 0:
            return None

        return loss_sum / total_samples

    def _compute_average_hvp_on_loader(self, data_loader, params, vec):
        hvp_sums = [torch.zeros_like(param, dtype=torch.float32, device=param.device) for param in params]
        total_samples = 0

        for batch in data_loader:
            batch = {k: v.to(self.device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
            with self._second_order_attention_context(), self._autocast_context():
                outputs = self.model(**batch)
                loss = outputs.loss

            if not torch.isfinite(loss):
                continue

            batch_size = batch['input_ids'].size(0)
            hvp_batch = self.compute_hvp(loss, params, vec)
            for idx, hvp_item in enumerate(hvp_batch):
                hvp_sums[idx].add_(hvp_item.detach().to(torch.float32), alpha=batch_size)

            total_samples += batch_size

        if total_samples == 0:
            return None

        return [hvp_sum / total_samples for hvp_sum in hvp_sums]

    def _compute_batch_loss_value(self, batch):
        batch = {k: v.to(self.device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
        if not batch:
            return None

        was_training = self.model.training
        self.model.eval()
        with torch.no_grad(), self._autocast_context():
            outputs = self.model(**batch)
            loss = outputs.loss
        if was_training:
            self.model.train()

        if not torch.isfinite(loss):
            return None
        return loss.item()

    def _compute_average_loss_value(self, data_loader):
        loss_total = 0.0
        total_samples = 0
        was_training = self.model.training
        self.model.eval()

        with torch.no_grad():
            for batch in data_loader:
                batch = {k: v.to(self.device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
                with self._autocast_context():
                    outputs = self.model(**batch)
                    loss = outputs.loss

                if not torch.isfinite(loss):
                    continue

                batch_size = batch['input_ids'].size(0)
                loss_total += loss.item() * batch_size
                total_samples += batch_size

        if was_training:
            self.model.train()

        if total_samples == 0:
            return None
        return loss_total / total_samples

    def _compute_guard_reference_loss_value(self, ref_loader, reference_batch):
        if ref_loader is not None:
            return self._compute_average_loss_value(ref_loader)
        return self._compute_batch_loss_value(reference_batch)

    def _build_forget_guard_loader(self, target_client):
        sample_limit = getattr(self.args, 'unlearn_forget_guard_sample_size', 0)
        if sample_limit is None or sample_limit <= 0:
            sample_limit = getattr(self.args, 'unlearn_grad_sample_size', 0)

        return target_client.get_full_train_loader(
            shuffle=False,
            sample_limit=sample_limit,
        )

    def _restore_params(self, params, params_before):
        with torch.no_grad():
            for param, before in zip(params, params_before):
                param.data.copy_(before.to(param.device))

    def _apply_scaled_update(self, params, updates, alpha):
        with torch.no_grad():
            for param, update in zip(params, updates):
                param.data.add_(update.to(param.device), alpha=alpha)

    def _to_float_or_none(self, value):
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _probe_unlearning_update_direction(
        self,
        params,
        params_before_probe,
        updates,
        scale,
        forget_guard_loader,
        global_guard_max,
        global_guard_enabled,
    ):
        records = []
        global_loss_before = self._to_float_or_none(
            self.experiment_metrics.get('global_loss_before_unlearning')
        )
        forget_loss_before = self.eval_loss_on_loader(
            forget_guard_loader,
            desc='Unlearn Direction Probe Forget Baseline',
        )

        for sign_label, sign_multiplier in (('positive', 1.0), ('negative', -1.0)):
            self._restore_params(params, params_before_probe)
            self._apply_scaled_update(params, updates, alpha=scale * sign_multiplier)

            forget_loss_after = self.eval_loss_on_loader(
                forget_guard_loader,
                desc=f'Unlearn Direction Probe {sign_label} Forget',
            )
            global_loss_after = self.eval_loss_on_loader(
                self.eval_loader,
                desc=f'Unlearn Direction Probe {sign_label} Global',
            )

            record = {
                'sign': sign_label,
                'sign_multiplier': sign_multiplier,
                'forget_loss_before_probe': forget_loss_before,
                'forget_loss_after_probe': forget_loss_after,
                'forget_loss_delta_probe': forget_loss_after - forget_loss_before,
                'global_loss_before_probe': global_loss_before,
                'global_loss_after_probe': global_loss_after,
                'global_loss_delta_probe': (
                    global_loss_after - global_loss_before
                    if global_loss_before is not None
                    else None
                ),
                'global_guard_passed_probe': (
                    (not global_guard_enabled)
                    or global_loss_after <= global_guard_max
                ),
            }
            records.append(record)
            print(
                "[FedHDS] Direction probe "
                f"{sign_label}: forget_delta={record['forget_loss_delta_probe']:+.6f}, "
                f"global_after={global_loss_after:.6f}"
            )

        self._restore_params(params, params_before_probe)
        self.model.train()

        guard_passing_records = [
            record for record in records
            if record.get('global_guard_passed_probe')
        ]
        candidates = guard_passing_records or records
        selected = max(
            candidates,
            key=lambda record: (
                record['forget_loss_delta_probe'],
                -float('inf') if record['global_loss_delta_probe'] is None else -record['global_loss_delta_probe'],
            ),
        )

        return selected, records

    def _apply_precomputed_unlearning_updates(
        self,
        params,
        updates,
        target_client,
        scale,
        reference_batch=None,
        ref_loader=None,
        allow_clipping=True,
        allow_direction_check=True,
        allow_ref_guard=True,
        allow_global_guard=True,
        allow_forget_guard=True,
        fixed_sign=None,
        method_label='FedHDS',
    ):
        update_stats = self._tensor_list_stats(updates)
        max_update_norm = getattr(self.args, 'unlearn_max_update_norm', 0.0)
        clip_enabled = allow_clipping and max_update_norm is not None and max_update_norm > 0
        clip_coefficient = 1.0
        scaled_update_l2_norm_before = update_stats['l2_norm'] * abs(scale)
        scaled_update_max_abs_before = update_stats['max_abs'] * abs(scale)

        if clip_enabled and scaled_update_l2_norm_before > max_update_norm and scaled_update_l2_norm_before > 0:
            clip_coefficient = max_update_norm / scaled_update_l2_norm_before
            scale *= clip_coefficient

        scaled_update_l2_norm_after = scaled_update_l2_norm_before * clip_coefficient
        scaled_update_max_abs_after = scaled_update_max_abs_before * clip_coefficient

        self.experiment_metrics['unlearn_eta'] = getattr(self.args, 'unlearn_eta', 1.0)
        self.experiment_metrics['unlearn_max_update_norm'] = max_update_norm
        self.experiment_metrics['update_clipping_enabled'] = clip_enabled
        self.experiment_metrics['update_clip_applied'] = clip_coefficient < 1.0
        self.experiment_metrics['update_clip_coefficient'] = clip_coefficient
        self.experiment_metrics['scaled_update_l2_norm_before_clipping'] = scaled_update_l2_norm_before
        self.experiment_metrics['scaled_update_l2_norm_after_clipping'] = scaled_update_l2_norm_after
        self.experiment_metrics['scaled_update_max_abs_before_clipping'] = scaled_update_max_abs_before
        self.experiment_metrics['scaled_update_max_abs_after_clipping'] = scaled_update_max_abs_after
        self.experiment_metrics['scaled_update_l2_norm'] = scaled_update_l2_norm_after
        self.experiment_metrics['scaled_update_max_abs'] = scaled_update_max_abs_after
        print(
            f"[{method_label}] Scaled update norm: "
            f"before_clip_L2={scaled_update_l2_norm_before:.6e}, "
            f"after_clip_L2={scaled_update_l2_norm_after:.6e}, "
            f"clip_coef={clip_coefficient:.6e}, "
            f"after_clip_max_abs={scaled_update_max_abs_after:.6e}"
        )

        unlearn_num_steps = max(1, getattr(self.args, 'unlearn_num_steps', 1))
        ref_guard_ratio = (
            getattr(self.args, 'unlearn_ref_loss_guard_ratio', 0.0)
            if allow_ref_guard
            else 0.0
        )
        global_guard_max = (
            getattr(self.args, 'unlearn_global_loss_guard_max', 0.0)
            if allow_global_guard
            else 0.0
        )
        update_sign_mode = fixed_sign or getattr(self.args, 'unlearn_update_sign', 'positive')
        direction_check_enabled = (
            allow_direction_check and getattr(self.args, 'unlearn_direction_check', False)
        )
        forget_loss_guard_enabled = (
            allow_forget_guard and getattr(self.args, 'unlearn_forget_loss_guard', False)
        )
        forget_loss_min_gain = (
            getattr(self.args, 'unlearn_forget_loss_min_gain', 0.0)
            if allow_forget_guard
            else 0.0
        )
        forget_loss_tolerance = (
            getattr(self.args, 'unlearn_forget_loss_tolerance', 0.0)
            if allow_forget_guard
            else 0.0
        )
        ref_guard_enabled = ref_guard_ratio is not None and ref_guard_ratio > 0
        global_guard_enabled = global_guard_max is not None and global_guard_max > 0

        if update_sign_mode == 'auto' and allow_direction_check:
            direction_check_enabled = True
        elif update_sign_mode == 'auto':
            update_sign_mode = 'positive'

        sign_multiplier = -1.0 if update_sign_mode == 'negative' else 1.0
        step_scale = (scale * sign_multiplier) / unlearn_num_steps
        step_update_l2_norm = scaled_update_l2_norm_after / unlearn_num_steps
        step_update_max_abs = scaled_update_max_abs_after / unlearn_num_steps
        ref_loss_before_guard = None
        ref_loss_after_last_accepted = None
        global_loss_after_last_accepted = None
        forget_loss_before_guard = None
        forget_loss_after_last_accepted = None
        forget_guard_loader = None
        guard_step_records = []
        guard_stop_reason = 'completed'

        if ref_guard_enabled:
            if ref_loader is None and reference_batch is None:
                print(f"[{method_label}] Reference loss guard disabled because no reference data is available.")
                ref_guard_enabled = False
            else:
                ref_loss_before_guard = self._compute_guard_reference_loss_value(ref_loader, reference_batch)
                if ref_loss_before_guard is None:
                    print(f"[{method_label}] Reference loss guard disabled because reference loss is unavailable.")
                    ref_guard_enabled = False
                else:
                    print(
                        f"[{method_label}] Reference loss guard: "
                        f"base={ref_loss_before_guard:.6f}, ratio={ref_guard_ratio:.4f}, "
                        f"limit={ref_loss_before_guard * ref_guard_ratio:.6f}"
                    )

        if global_guard_enabled:
            print(f"[{method_label}] Global loss guard: max={global_guard_max:.6f}")

        if direction_check_enabled or forget_loss_guard_enabled:
            forget_guard_loader = self._build_forget_guard_loader(target_client)
            self.experiment_metrics['unlearn_forget_guard_sample_size'] = len(forget_guard_loader.dataset)
        else:
            self.experiment_metrics['unlearn_forget_guard_sample_size'] = 0

        if direction_check_enabled:
            params_before_direction_probe = [param.detach().clone() for param in params]
            direction_record, direction_records = self._probe_unlearning_update_direction(
                params=params,
                params_before_probe=params_before_direction_probe,
                updates=updates,
                scale=scale,
                forget_guard_loader=forget_guard_loader,
                global_guard_max=global_guard_max,
                global_guard_enabled=global_guard_enabled,
            )
            self.experiment_metrics['unlearn_direction_probe_records'] = direction_records
            self.experiment_metrics['unlearn_best_probe_update_sign'] = direction_record['sign']
            self.experiment_metrics['unlearn_best_probe_direction_forget_delta'] = direction_record[
                'forget_loss_delta_probe'
            ]
            self.experiment_metrics['unlearn_best_probe_direction_global_loss_after'] = direction_record[
                'global_loss_after_probe'
            ]
            if getattr(self.args, 'unlearn_update_sign', 'positive') == 'auto':
                sign_multiplier = direction_record['sign_multiplier']
                step_scale = (scale * sign_multiplier) / unlearn_num_steps
                self.experiment_metrics['unlearn_selected_update_sign'] = direction_record['sign']
                self.experiment_metrics['unlearn_selected_direction_forget_delta'] = direction_record[
                    'forget_loss_delta_probe'
                ]
                self.experiment_metrics['unlearn_selected_direction_global_loss_after'] = direction_record[
                    'global_loss_after_probe'
                ]
                print(f"[{method_label}] Auto-selected update sign: {direction_record['sign']}")
            else:
                configured_sign = 'negative' if sign_multiplier < 0 else 'positive'
                self.experiment_metrics['unlearn_selected_update_sign'] = configured_sign
                print(f"[{method_label}] Direction check kept configured update sign: {update_sign_mode}")
        else:
            self.experiment_metrics['unlearn_selected_update_sign'] = (
                'negative' if sign_multiplier < 0 else 'positive'
            )

        if forget_loss_guard_enabled:
            if forget_guard_loader is None:
                forget_guard_loader = self._build_forget_guard_loader(target_client)
                self.experiment_metrics['unlearn_forget_guard_sample_size'] = len(forget_guard_loader.dataset)

            forget_loss_before_guard = self.eval_loss_on_loader(
                forget_guard_loader,
                desc='Unlearn Forget Guard Baseline',
            )
            if forget_loss_min_gain > 0:
                forget_loss_limit = forget_loss_before_guard + forget_loss_min_gain
                print(
                    f"[{method_label}] Forget loss guard: "
                    f"base={forget_loss_before_guard:.6f}, min_gain={forget_loss_min_gain:.6f}, "
                    f"limit={forget_loss_limit:.6f}"
                )
            else:
                forget_loss_limit = forget_loss_before_guard - forget_loss_tolerance
                print(
                    f"[{method_label}] Forget loss guard: "
                    f"base={forget_loss_before_guard:.6f}, tolerance={forget_loss_tolerance:.6f}, "
                    f"limit={forget_loss_limit:.6f}"
                )
        else:
            forget_loss_limit = None

        self.experiment_metrics['unlearn_num_steps'] = unlearn_num_steps
        self.experiment_metrics['unlearn_ref_loss_guard_ratio'] = ref_guard_ratio
        self.experiment_metrics['unlearn_global_loss_guard_max'] = global_guard_max
        self.experiment_metrics['unlearn_update_sign_mode'] = update_sign_mode
        self.experiment_metrics['unlearn_direction_check_enabled'] = direction_check_enabled
        self.experiment_metrics['unlearn_forget_loss_guard_enabled'] = forget_loss_guard_enabled
        self.experiment_metrics['unlearn_forget_loss_min_gain'] = forget_loss_min_gain
        self.experiment_metrics['unlearn_forget_loss_tolerance'] = forget_loss_tolerance
        self.experiment_metrics['unlearn_ref_loss_guard_enabled'] = ref_guard_enabled
        self.experiment_metrics['unlearn_global_loss_guard_enabled'] = global_guard_enabled
        self.experiment_metrics['unlearn_reference_loss_before_guard'] = ref_loss_before_guard
        self.experiment_metrics['unlearn_forget_loss_before_guard'] = forget_loss_before_guard
        self.experiment_metrics['unlearn_forget_loss_limit'] = forget_loss_limit
        self.experiment_metrics['unlearn_requested_scale_after_clipping'] = scale
        self.experiment_metrics['unlearn_step_scale'] = step_scale
        self.experiment_metrics['unlearn_step_update_l2_norm'] = step_update_l2_norm
        self.experiment_metrics['unlearn_step_update_max_abs'] = step_update_max_abs

        params_before = [param.detach().clone() for param in params]
        steps_attempted = 0
        steps_accepted = 0

        for step_idx in range(1, unlearn_num_steps + 1):
            steps_attempted += 1
            params_before_step = [param.detach().clone() for param in params]
            self._apply_scaled_update(params, updates, alpha=step_scale)

            step_record = {
                'step': step_idx,
                'accepted': True,
                'step_scale': step_scale,
                'step_update_l2_norm': step_update_l2_norm,
            }
            reject_reason = None

            if ref_guard_enabled:
                ref_loss_after_step = self._compute_guard_reference_loss_value(ref_loader, reference_batch)
                ref_loss_limit = ref_loss_before_guard * ref_guard_ratio
                step_record['reference_loss_after_step'] = ref_loss_after_step
                step_record['reference_loss_limit'] = ref_loss_limit
                if ref_loss_after_step is None:
                    reject_reason = 'reference_loss_unavailable'
                elif ref_loss_after_step > ref_loss_limit:
                    reject_reason = 'reference_loss_guard'

            if reject_reason is None and global_guard_enabled:
                global_loss_after_step = self.eval_loss_on_loader(
                    self.eval_loader,
                    desc=f'Unlearn Guard Step {step_idx}/{unlearn_num_steps}',
                )
                step_record['global_loss_after_step'] = global_loss_after_step
                step_record['global_loss_limit'] = global_guard_max
                if global_loss_after_step > global_guard_max:
                    reject_reason = 'global_loss_guard'

            if reject_reason is None and forget_loss_guard_enabled:
                forget_loss_after_step = self.eval_loss_on_loader(
                    forget_guard_loader,
                    desc=f'Unlearn Forget Guard Step {step_idx}/{unlearn_num_steps}',
                )
                step_record['forget_loss_after_step'] = forget_loss_after_step
                step_record['forget_loss_limit'] = forget_loss_limit
                step_record['forget_loss_delta_from_guard_base'] = (
                    forget_loss_after_step - forget_loss_before_guard
                    if forget_loss_before_guard is not None
                    else None
                )
                if forget_loss_after_step is None:
                    reject_reason = 'forget_loss_unavailable'
                elif forget_loss_after_step < forget_loss_limit:
                    reject_reason = 'forget_loss_guard'

            if reject_reason is not None:
                self._restore_params(params, params_before_step)
                step_record['accepted'] = False
                step_record['reject_reason'] = reject_reason
                guard_stop_reason = reject_reason
                guard_step_records.append(step_record)
                print(f"[{method_label}] Step {step_idx}/{unlearn_num_steps} rejected: {reject_reason}.")
                break

            steps_accepted += 1
            ref_loss_after_last_accepted = step_record.get(
                'reference_loss_after_step',
                ref_loss_after_last_accepted,
            )
            global_loss_after_last_accepted = step_record.get(
                'global_loss_after_step',
                global_loss_after_last_accepted,
            )
            forget_loss_after_last_accepted = step_record.get(
                'forget_loss_after_step',
                forget_loss_after_last_accepted,
            )
            guard_step_records.append(step_record)
            print(f"[{method_label}] Step {step_idx}/{unlearn_num_steps} accepted.")
            self.model.train()

        delta_stats = self._param_delta_stats(params, params_before)
        self.experiment_metrics['actual_param_delta_l2_norm'] = delta_stats['l2_norm']
        self.experiment_metrics['actual_param_delta_max_abs'] = delta_stats['max_abs']
        self.experiment_metrics['actual_param_delta_nonzero_tensors'] = delta_stats['nonzero_tensors']
        self.experiment_metrics['unlearn_steps_attempted'] = steps_attempted
        self.experiment_metrics['unlearn_steps_accepted'] = steps_accepted
        self.experiment_metrics['unlearn_guard_stop_reason'] = guard_stop_reason
        self.experiment_metrics['unlearn_guard_step_records'] = guard_step_records
        self.experiment_metrics['unlearn_reference_loss_after_last_accepted'] = ref_loss_after_last_accepted
        self.experiment_metrics['unlearn_global_loss_after_last_accepted'] = global_loss_after_last_accepted
        self.experiment_metrics['unlearn_forget_loss_after_last_accepted'] = forget_loss_after_last_accepted
        self.experiment_metrics['unlearn_applied_scale'] = step_scale * steps_accepted
        self.experiment_metrics['unlearn_applied_update_l2_norm_estimate'] = step_update_l2_norm * steps_accepted
        print(
            f"[{method_label}] Actual parameter delta norm: "
            f"L2={delta_stats['l2_norm']:.6e}, max_abs={delta_stats['max_abs']:.6e}, "
            f"nonzero_tensors={delta_stats['nonzero_tensors']}/{delta_stats['num_tensors']}"
        )
        print(
            f"[{method_label}] Step guard summary: "
            f"accepted={steps_accepted}/{steps_attempted}, stop_reason={guard_stop_reason}"
        )

        return step_scale * steps_accepted

    def apply_fedhds_unlearning(self, forget_client_idx, client_list):
        print(f"--- [FedHDS Unlearning] Starting Process for Client {forget_client_idx} ---")
        use_retained_hessian = getattr(self.args, 'use_retained_hessian', False)
        hessian_mode = 'retained-set' if use_retained_hessian else 'forget-batch (baseline)'
        print(f"[FedHDS] Hessian reference mode: {hessian_mode}")
        self.experiment_metrics['requested_hessian_reference_mode'] = hessian_mode
        if use_retained_hessian:
            print(f"[FedHDS] Retained samples per client: {getattr(self.args, 'hessian_ref_per_client', 1)}")

        target_client = next((client for client in client_list if client.idx == forget_client_idx), None)
        if target_client is None:
            print("[Error] Forget client not found.")
            return

        forget_loader = target_client.get_full_train_loader(
            shuffle=False,
            sample_limit=getattr(self.args, 'unlearn_grad_sample_size', 0),
        )
        print(f"[FedHDS] Forget gradient sample size: {len(forget_loader.dataset)}")
        self.experiment_metrics['forget_gradient_sample_size'] = len(forget_loader.dataset)

        if getattr(self.model, 'is_gradient_checkpointing', False):
            print("[FedHDS] Disabling gradient checkpointing for autograd.grad-based unlearning.")
            self.model.gradient_checkpointing_disable()
            self.experiment_metrics['gradient_checkpointing_disabled_for_unlearning'] = True
        else:
            self.experiment_metrics['gradient_checkpointing_disabled_for_unlearning'] = False

        if hasattr(self.model, 'config'):
            self.model.config.use_cache = False

        self.model.train()
        params = [param for param in self.model.parameters() if param.requires_grad]

        grads, reference_batch = self._compute_average_forget_gradients(forget_loader, params)
        if grads is None or reference_batch is None:
            print("[Error] Forget client data loader is empty.")
            return

        grad_stats = self._tensor_list_stats(grads)
        self.experiment_metrics['forget_gradient_l2_norm'] = grad_stats['l2_norm']
        self.experiment_metrics['forget_gradient_max_abs'] = grad_stats['max_abs']
        self.experiment_metrics['forget_gradient_nonzero_tensors'] = grad_stats['nonzero_tensors']
        print(
            "[FedHDS] Forget gradient norm: "
            f"L2={grad_stats['l2_norm']:.6e}, max_abs={grad_stats['max_abs']:.6e}, "
            f"nonzero_tensors={grad_stats['nonzero_tensors']}/{grad_stats['num_tensors']}"
        )

        inverse_hvp = [grad.detach().clone() for grad in grads]

        print("[FedHDS] Estimating Inverse Hessian Vector Product...")
        reference_loss = None
        ref_loader = None
        use_retained_reference_hvp = False

        if use_retained_hessian:
            ref_loader, ref_size = self.build_retained_reference_loader(forget_client_idx, client_list)
            print(f"[FedHDS] Retained reference set size: {ref_size}")
            self.experiment_metrics['retained_reference_set_size'] = ref_size
            use_retained_reference_hvp = ref_loader is not None and ref_size > 0
            if not use_retained_reference_hvp:
                print("[FedHDS] Retained reference set unavailable. Falling back to forget-batch Hessian.")
        else:
            self.experiment_metrics['retained_reference_set_size'] = 0

        if not use_retained_reference_hvp:
            print("[FedHDS] Using forget client batch as Hessian reference.")
            self.experiment_metrics['effective_hessian_reference_mode'] = 'forget-batch'
            with self._second_order_attention_context(), self._autocast_context():
                reference_outputs = self.model(**reference_batch)
                reference_loss = reference_outputs.loss
        else:
            self.experiment_metrics['effective_hessian_reference_mode'] = 'retained-set' if use_retained_hessian else 'forget-batch'

        recursion_depth = max(0, getattr(self.args, 'lissa_depth', 1))
        damping = getattr(self.args, 'lissa_damping', 0.01)
        print(f"[FedHDS] LiSSA depth={recursion_depth}, damping={damping}")
        self.experiment_metrics['lissa_depth'] = recursion_depth
        self.experiment_metrics['lissa_damping'] = damping

        for _ in range(recursion_depth):
            if use_retained_reference_hvp:
                hvp = self._compute_average_hvp_on_loader(ref_loader, params, inverse_hvp)
                if hvp is None:
                    print("[FedHDS] Retained HVP loader produced no valid batches. Falling back to forget-batch Hessian.")
                    self.experiment_metrics['effective_hessian_reference_mode'] = 'forget-batch'
                    use_retained_reference_hvp = False
                    with self._second_order_attention_context(), self._autocast_context():
                        reference_outputs = self.model(**reference_batch)
                        reference_loss = reference_outputs.loss
                    hvp = self.compute_hvp(reference_loss, params, inverse_hvp)
            else:
                hvp = self.compute_hvp(reference_loss, params, inverse_hvp)
            hvp_stats = self._tensor_list_stats(hvp)
            self.experiment_metrics['last_hvp_l2_norm'] = hvp_stats['l2_norm']
            self.experiment_metrics['last_hvp_max_abs'] = hvp_stats['max_abs']
            print(
                "[FedHDS] HVP norm: "
                f"L2={hvp_stats['l2_norm']:.6e}, max_abs={hvp_stats['max_abs']:.6e}"
            )
            inverse_hvp = [
                grad + (1 - damping) * ihvp - hvp_item
                for grad, ihvp, hvp_item in zip(grads, inverse_hvp, hvp)
            ]

        scale = 1.0 / (self.args.num_clients - 1) if self.args.num_clients > 1 else 1.0
        scale *= getattr(self.args, 'unlearn_eta', 1.0)

        ihvp_stats = self._tensor_list_stats(inverse_hvp)
        self.experiment_metrics['inverse_hvp_l2_norm'] = ihvp_stats['l2_norm']
        self.experiment_metrics['inverse_hvp_max_abs'] = ihvp_stats['max_abs']

        applied_scale = self._apply_precomputed_unlearning_updates(
            params=params,
            updates=inverse_hvp,
            target_client=target_client,
            scale=scale,
            reference_batch=reference_batch,
            ref_loader=ref_loader,
            allow_clipping=True,
            allow_direction_check=True,
            allow_ref_guard=True,
            allow_global_guard=True,
            allow_forget_guard=True,
            fixed_sign=None,
            method_label='FedHDS',
        )

        print(f'[FedHDS Unlearn] Newton-step completed. Applied scale = {applied_scale:.6e}')
        torch.cuda.empty_cache()

    def apply_gradient_ascent_unlearning(self, forget_client_idx, client_list, use_guards=False):
        print(f"--- [Gradient Ascent Unlearning] Starting Process for Client {forget_client_idx} ---")

        target_client = next((client for client in client_list if client.idx == forget_client_idx), None)
        if target_client is None:
            print("[Error] Forget client not found.")
            return

        forget_loader = target_client.get_full_train_loader(
            shuffle=False,
            sample_limit=getattr(self.args, 'unlearn_grad_sample_size', 0),
        )
        print(f"[GA] Forget gradient sample size: {len(forget_loader.dataset)}")
        self.experiment_metrics['forget_gradient_sample_size'] = len(forget_loader.dataset)
        self.experiment_metrics['requested_hessian_reference_mode'] = 'gradient-ascent'
        self.experiment_metrics['effective_hessian_reference_mode'] = 'gradient-ascent'
        self.experiment_metrics['retained_reference_set_size'] = 0
        self.experiment_metrics['lissa_depth'] = 0
        self.experiment_metrics['lissa_damping'] = 0.0

        if getattr(self.model, 'is_gradient_checkpointing', False):
            print("[GA] Disabling gradient checkpointing for autograd.grad-based unlearning.")
            self.model.gradient_checkpointing_disable()
            self.experiment_metrics['gradient_checkpointing_disabled_for_unlearning'] = True
        else:
            self.experiment_metrics['gradient_checkpointing_disabled_for_unlearning'] = False

        if hasattr(self.model, 'config'):
            self.model.config.use_cache = False

        self.model.train()
        params = [param for param in self.model.parameters() if param.requires_grad]
        grads, _ = self._compute_average_forget_gradients(forget_loader, params)
        if grads is None:
            print("[Error] Forget client data loader is empty.")
            return

        grad_stats = self._tensor_list_stats(grads)
        self.experiment_metrics['forget_gradient_l2_norm'] = grad_stats['l2_norm']
        self.experiment_metrics['forget_gradient_max_abs'] = grad_stats['max_abs']
        self.experiment_metrics['forget_gradient_nonzero_tensors'] = grad_stats['nonzero_tensors']
        print(
            "[GA] Forget gradient norm: "
            f"L2={grad_stats['l2_norm']:.6e}, max_abs={grad_stats['max_abs']:.6e}, "
            f"nonzero_tensors={grad_stats['nonzero_tensors']}/{grad_stats['num_tensors']}"
        )

        scale = 1.0 / (self.args.num_clients - 1) if self.args.num_clients > 1 else 1.0
        scale *= getattr(self.args, 'unlearn_eta', 1.0)

        if getattr(self.args, 'unlearn_ref_loss_guard_ratio', 0.0) > 0:
            print("[GA] Reference loss guard is disabled for gradient-ascent unlearning.")

        applied_scale = self._apply_precomputed_unlearning_updates(
            params=params,
            updates=grads,
            target_client=target_client,
            scale=scale,
            reference_batch=None,
            ref_loader=None,
            allow_clipping=use_guards,
            allow_direction_check=False,
            allow_ref_guard=False,
            allow_global_guard=use_guards,
            allow_forget_guard=use_guards,
            fixed_sign='positive',
            method_label='GA',
        )

        print(f'[GA Unlearn] Gradient-ascent step completed. Applied scale = {applied_scale:.6e}')
        torch.cuda.empty_cache()

    def eval_loss(self, cur_round):
        return self.eval_loss_on_loader(
            self.eval_loader,
            desc=f'Evaluating Round {cur_round}',
            history_label=cur_round,
        )

    def eval_loss_on_loader(self, data_loader, desc, history_label=None):
        self.model.eval()
        progress_bar_eval = tqdm(data_loader, desc=desc)
        loss_total_eval = 0.0
        num_eval = 0

        with torch.no_grad():
            for batch in progress_bar_eval:
                batch = {k: v.to(self.device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
                with self._autocast_context():
                    outputs = self.model(**batch)
                    loss = outputs.loss

                if torch.isfinite(loss):
                    batch_size = batch['input_ids'].size(0)
                    loss_total_eval += loss.item() * batch_size
                    num_eval += batch_size

                if num_eval > 0:
                    progress_bar_eval.set_description(f'{desc} Loss: {loss_total_eval / num_eval:.4f}')

        avg_loss = loss_total_eval / num_eval if num_eval > 0 else 10.0
        if history_label is not None:
            self.eval_history.append(avg_loss)
        return avg_loss

    def eval_generation(self, data_loader, desc, max_samples=200):
        self.model.eval()
        rouge_scores = []
        bleu_scores = []
        sample_count = 0

        with torch.no_grad():
            for batch in tqdm(data_loader, desc=desc):
                if sample_count >= max_samples:
                    break

                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels']

                with self._autocast_context():
                    generated = self.model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=64,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                        do_sample=False,
                    )

                for i in range(len(generated)):
                    if sample_count >= max_samples:
                        break

                    gen_ids = generated[i][input_ids[i].shape[0]:]
                    gen_text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)

                    label_ids = labels[i]
                    label_ids = label_ids[label_ids != -100]
                    ref_text = self.tokenizer.decode(label_ids, skip_special_tokens=True)

                    if not gen_text.strip() or not ref_text.strip():
                        continue

                    try:
                        r_score = rouge_score(gen_ids.unsqueeze(0), label_ids.unsqueeze(0), self.tokenizer)
                        rouge_scores.append(r_score)
                    except Exception:
                        pass

                    try:
                        b_score = bleu_score(gen_ids.unsqueeze(0), label_ids.unsqueeze(0), self.tokenizer)
                        bleu_scores.append(b_score)
                    except Exception:
                        pass

                    sample_count += 1

        avg_rouge = float(sum(rouge_scores) / len(rouge_scores)) if rouge_scores else 0.0
        avg_bleu = float(sum(bleu_scores) / len(bleu_scores)) if bleu_scores else 0.0

        print(f"[GEN-EVAL] {desc}: ROUGE-L={avg_rouge:.4f}, BLEU={avg_bleu:.4f}, samples={sample_count}")
        return {'rouge_l': avg_rouge, 'bleu': avg_bleu, 'num_samples': sample_count}

    def save_checkpoint(self, label):
        os.makedirs(self.log_dir, exist_ok=True)
        ckpt_path = os.path.join(self.log_dir, f'lora_state_{label}.pt')
        torch.save(self.get_lora_state_dict(), ckpt_path)
        print(f"[SAVE] Checkpoint saved to {ckpt_path}")

    def save_results(self, eval_history, train_time, rounds_completed, num_clients):
        os.makedirs(self.log_dir, exist_ok=True)
        results_path = os.path.join(self.log_dir, 'final_results.json')
        results_data = {
            'eval_history': eval_history,
            'train_time': train_time,
            'rounds_completed': rounds_completed,
            'num_clients': num_clients,
            'experiment_metrics': self.experiment_metrics,
            'config': {key: str(value) for key, value in vars(self.args).items()},
        }
        with open(results_path, 'w') as file:
            json.dump(results_data, file, indent=2)
        print(f"[SAVE] Results saved to {results_path}")
