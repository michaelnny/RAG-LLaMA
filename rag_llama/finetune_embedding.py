# Copyright (c) 2024 Michael Hu.
# This project is released under the MIT License.
# See the accompanying LICENSE file for details.

"""Script to fine-tune embedding model"""


from typing import List, Tuple, Mapping, Text, Any, Callable
import argparse
import os
import random
import pickle
import functools
import tqdm
import json
import math
import time
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter


from rag_llama.models.embedding import create_embedding_tokenizer, create_embedding_model, post_embedding_processing
from rag_llama.core.schedule import CosineDecayWithWarmupLRScheduler


os.environ['TOKENIZERS_PARALLELISM'] = 'false'


def load_and_prepare_dataset(tokenizer: Callable, data_source: str, max_seq_len: int = 512) -> List[Mapping[Text, Any]]:
    results = []

    samples = pickle.load(open(data_source, 'rb'))

    for item in samples:
        query_tokens = tokenizer(item['query'], padding=False, truncation=True, return_tensors='pt')
        positive_tokens = tokenizer(item['positive_match'], padding=False, truncation=True, return_tensors='pt')
        negative_tokens = tokenizer(item['negative_match'], padding=False, truncation=True, return_tensors='pt')

        # remove batch dimension
        query_tokens = torch.squeeze(query_tokens['input_ids'], dim=0)
        positive_tokens = torch.squeeze(positive_tokens['input_ids'], dim=0)
        negative_tokens = torch.squeeze(negative_tokens['input_ids'], dim=0)

        if len(query_tokens) > max_seq_len:
            query_tokens = query_tokens[:max_seq_len].clone()
        if len(positive_tokens) > max_seq_len:
            positive_tokens = positive_tokens[:max_seq_len].clone()
        if len(negative_tokens) > max_seq_len:
            negative_tokens = negative_tokens[:max_seq_len].clone()

        results.append({'query_tokens': query_tokens, 'positive_tokens': positive_tokens, 'negative_tokens': negative_tokens})

    return results


class FineTuneDataset(Dataset):
    """For supervised fune-tuning, for each sample, we have pair of question, positive answer, negative answer tokens."""

    def __init__(self, items: List[Mapping[Text, Any]]) -> None:
        """
        Args:
            items: a list of pre-processed (tokenized) samples.

        """
        assert items is not None and len(items) > 1
        self.data = items

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def log_statistics(tb_writer: SummaryWriter, epoch: int, stats: Mapping[Text, Any], is_training: bool) -> None:

    if tb_writer is not None:
        tb_tag = 'train_epochs' if is_training else 'val_epochs'
        for k, v in stats.items():
            if isinstance(v, (int, float)):
                tb_writer.add_scalar(f'{tb_tag}/{k}', v, epoch)


def compute_num_trainable_params(model: torch.nn.Module) -> Tuple[int, int]:
    num_trainable_params = 0
    num_frozen_params = 0

    for p_name, params in model.named_parameters():
        is_trainable = params.requires_grad
        is_quantized = hasattr(params, 'quant_state')

        # quantized layer is not trainable
        if not is_trainable and is_quantized:
            num_params = math.prod(params.quant_state.shape)
        else:
            num_params = params.numel()

        num_trainable_params += num_params if is_trainable else 0
        num_frozen_params += num_params if not is_trainable else 0

    return num_trainable_params, num_frozen_params


def create_optimizer(
    model: torch.nn.Module,
    lr: float,
    eps: float = 1e-8,
    weight_decay: float = 0.0,
    betas: Tuple[float] = (0.9, 0.95),
    fused: bool = False,
) -> torch.optim.AdamW:
    """
    Returns the PyTorch AdamW optimizer for the model,
    where we skip apply weight decay to layer norm, embedding, and all bias,
    and apply weight decay to the reset of parameters.
    """

    # Create empty lists to store parameters for weight decay and no weight decay.
    decay = []
    no_decay = []

    for p_name, params in model.named_parameters():
        is_trainable = params.requires_grad

        if is_trainable:
            # Check for parameters corresponding to torch.nn.LayerNorm or torch.nn.Embedding.
            if p_name.endswith('bias') or 'embeddings.weight' in p_name or 'LayerNorm' in p_name:
                no_decay.append(params)
            else:
                decay.append(params)

    if weight_decay > 0:
        num_decay_params = sum(p.numel() for p in decay)
        num_nodecay_params = sum(p.numel() for p in no_decay)
        print(f'Number of decayed parameters: {num_decay_params:,}')
        print(f'Number of non-decayed parameters: {num_nodecay_params:,}')

    # create the pytorch optimizer object
    optim_groups = [
        {'params': decay, 'weight_decay': weight_decay},
        {'params': no_decay, 'weight_decay': 0.0},
    ]

    optim_kwargs = {'lr': lr, 'eps': eps, 'betas': betas, 'fused': fused}
    optimizer = torch.optim.AdamW(optim_groups, **optim_kwargs)
    return optimizer


def custom_collate_fn(batch: List[dict], pad_id: int) -> Mapping[Text, torch.Tensor]:
    """
    Custom collate function to pad the sequence to maximum length in the batch.
    """

    batch_size = len(batch)
    max_query_len = max([len(item['query_tokens']) for item in batch])
    max_positive_len = max([len(item['positive_tokens']) for item in batch])
    max_negative_len = max([len(item['negative_tokens']) for item in batch])

    batch_queries = torch.full((batch_size, max_query_len), pad_id, dtype=torch.long)
    batch_positives = torch.full((batch_size, max_positive_len), pad_id, dtype=torch.long)
    batch_negatives = torch.full((batch_size, max_negative_len), pad_id, dtype=torch.long)

    for i, item in enumerate(batch):
        query = item['query_tokens']
        positive = item['positive_tokens']
        negative = item['negative_tokens']

        batch_queries[i, : len(query)] = query
        batch_positives[i, : len(positive)] = positive
        batch_negatives[i, : len(negative)] = negative

    # create attention masks, where 1s are valid tokens, and 0s are padding tokens
    batch_queries_attn_mask = torch.where(batch_queries == pad_id, 0, 1)
    batch_positives_attn_mask = torch.where(batch_positives == pad_id, 0, 1)
    batch_negatives_attn_mask = torch.where(batch_negatives == pad_id, 0, 1)

    return {
        'queries_tokens': batch_queries,
        'positives_tokens': batch_positives,
        'negatives_tokens': batch_negatives,
        'queries_attn_mask': batch_queries_attn_mask,
        'positives_attn_mask': batch_positives_attn_mask,
        'negatives_attn_mask': batch_negatives_attn_mask,
    }


def compute_embedding_loss(query_embeddings: torch.Tensor, positive_embeddings: torch.Tensor, negative_embeddings: torch.Tensor, margin: float = 0.5) -> torch.Tensor:
    """Compute cosine embedding loss"""
    concat_query_embeddings = torch.cat([query_embeddings, query_embeddings], dim=0)
    concat_response_embeddings = torch.cat([positive_embeddings, negative_embeddings], dim=0)

    # cosine_embedding_loss requires a target with shape of (batch_size, )
    # where 1s for positive embedding, and -1s for negative embedding
    targets = [1] * len(positive_embeddings) + [-1] * len(negative_embeddings)
    concat_targets = torch.tensor(targets, device=concat_query_embeddings.device, dtype=torch.long)

    loss = F.cosine_embedding_loss(concat_query_embeddings, concat_response_embeddings, concat_targets, margin=margin, reduction='mean')
    return loss


def run_train_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.AdamW,
    gradient_accum_steps: int,
    loss_margin: float,
    grad_clip: float,
    scheduler: CosineDecayWithWarmupLRScheduler = None,
    scaler: torch.cuda.amp.GradScaler = None,
) -> List[float]:

    assert gradient_accum_steps >= 1

    train_losses = []
    model.train()
    optimizer.zero_grad(set_to_none=True)

    def update_step() -> None:
        """Run a single parameter update step"""
        if grad_clip > 0.0:
            if scaler is not None:  # when using float16
                scaler.unscale_(optimizer)  # unscale before clip gradients

            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        if scaler is not None:  # when using float16
            scaler.step(optimizer)
            scaler.update()  # adjust scaling for next batch
        else:
            optimizer.step()

        # prepare for next update
        optimizer.zero_grad(set_to_none=True)

        if scheduler:
            scheduler.step()  # call lr scheduler on a step-by-step basis instead of epoch

    train_pbar = tqdm.tqdm(range(len(loader)), colour='blue', desc='Train steps')
    for i, batch in enumerate(loader):
        # move to cuda
        for k in batch.keys():
            batch[k] = batch[k].to('cuda', non_blocking=True)

        # Forward passing to compute embeddings
        query_output = model(
            batch['queries_tokens'],
            attention_mask=batch['queries_attn_mask'],
        )
        positive_output = model(
            batch['positives_tokens'],
            attention_mask=batch['positives_attn_mask'],
        )
        negative_output = model(
            batch['negatives_tokens'],
            attention_mask=batch['negatives_attn_mask'],
        )

        query_embeddings = post_embedding_processing(query_output.last_hidden_state, batch['queries_attn_mask'], True)
        positive_embeddings = post_embedding_processing(positive_output.last_hidden_state, batch['positives_attn_mask'], True)
        negative_embeddings = post_embedding_processing(negative_output.last_hidden_state, batch['negatives_attn_mask'], True)

        loss = compute_embedding_loss(query_embeddings, positive_embeddings, negative_embeddings, loss_margin)  # [batch_size]

        # scale the loss to account for gradient accumulation
        scaled_loss = loss / gradient_accum_steps

        if scaler is not None:  # when using float16
            scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()

        train_losses.append(loss.detach().item())

        train_pbar.update(1)

        if i % gradient_accum_steps == 0 or (i + 1) == len(loader):
            update_step()

    train_pbar.close()
    return train_losses


@torch.no_grad()
def run_validation_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    loss_margin: float,
) -> List[float]:

    val_losses = []

    model.eval()

    val_pbar = tqdm.tqdm(range(len(loader)), colour='green', desc='Evaluation steps')
    for i, batch in enumerate(loader):
        # move to cuda
        for k in batch.keys():
            batch[k] = batch[k].to('cuda', non_blocking=True)

        # Forward passing to compute embeddings
        query_output = model(
            batch['queries_tokens'],
            attention_mask=batch['queries_attn_mask'],
        )
        positive_output = model(
            batch['positives_tokens'],
            attention_mask=batch['positives_attn_mask'],
        )
        negative_output = model(
            batch['negatives_tokens'],
            attention_mask=batch['negatives_attn_mask'],
        )

        query_embeddings = post_embedding_processing(query_output.last_hidden_state, batch['queries_attn_mask'], True)
        positive_embeddings = post_embedding_processing(positive_output.last_hidden_state, batch['positives_attn_mask'], True)
        negative_embeddings = post_embedding_processing(negative_output.last_hidden_state, batch['negatives_attn_mask'], True)

        loss = compute_embedding_loss(query_embeddings, positive_embeddings, negative_embeddings, loss_margin)  # [batch_size]
        val_losses.append(loss.detach().item())
        val_pbar.update(1)

    val_pbar.close()
    return val_losses


def main():

    if not torch.version.cuda:
        raise SystemExit('This script requires Pytorch with CUDA.')

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    train_ds_file = os.path.join(args.dataset_dir, 'train.pk')
    val_ds_file = os.path.join(args.dataset_dir, 'validation.pk')
    meta_file = os.path.join(args.dataset_dir, 'meta.json')

    if not all([os.path.exists(f) for f in (train_ds_file, val_ds_file, meta_file)]):
        raise SystemExit('Invalid dataset files')

    os.makedirs(args.logs_dir, exist_ok=True)
    os.makedirs(args.ckpt_dir, exist_ok=True)

    tokenizer = create_embedding_tokenizer()

    added_special_tokens = False
    with open(meta_file, 'r', encoding='utf-8') as file:
        content = file.read()
        metadata = json.loads(content)
        if metadata and 'alert_codes' in metadata and len(metadata['alert_codes']) > 0:
            tokenizer.add_tokens(metadata['alert_codes'])
            tokenizer.save_pretrained(os.path.join(args.ckpt_dir, 'tokenizer'))
            added_special_tokens = True
            print('Added tokens to tokenizer')

    # BERT tokenizer doesn't have EOS token
    _collate_fn = functools.partial(
        custom_collate_fn,
        pad_id=tokenizer.pad_token_id,
    )

    cuda_kwargs = {
        'collate_fn': _collate_fn,
        'num_workers': 1,
        'pin_memory': False,
        'shuffle': True,
    }

    train_items = load_and_prepare_dataset(tokenizer=tokenizer, data_source=train_ds_file, max_seq_len=args.max_seq_len)
    train_dataset = FineTuneDataset(train_items)
    train_kwargs = {'batch_size': args.train_batch_size, 'sampler': None}
    train_kwargs.update(cuda_kwargs)
    train_loader = DataLoader(train_dataset, **train_kwargs)

    val_items = load_and_prepare_dataset(tokenizer=tokenizer, data_source=val_ds_file, max_seq_len=args.max_seq_len)
    val_dataset = FineTuneDataset(val_items)
    val_kwargs = {'batch_size': args.val_batch_size, 'sampler': None}
    val_kwargs.update(cuda_kwargs)
    val_loader = DataLoader(val_dataset, **val_kwargs)

    tb_writer = SummaryWriter(args.logs_dir)

    torch.cuda.set_device('cuda:0')

    compute_dtype = torch.float32
    scaler = None
    if args.mixed_precision:
        if torch.version.cuda and torch.cuda.is_bf16_supported():
            compute_dtype = torch.bfloat16
        else:
            compute_dtype = torch.float16
            scaler = torch.cuda.amp.GradScaler()
    else:
        print('Training in float32 mode, make sure you have enough GPU RAM')

    model = create_embedding_model()

    if added_special_tokens:
        model.resize_token_embeddings(len(tokenizer))

    for name, module in model.named_modules():
        module = module.to(dtype=compute_dtype)

    # make embedding layer trainable
    trainable_layers = [
        'word_embeddings',
        # 'position_embeddings',
        # 'attention.self.key',
        # 'attention.self.query',
        # 'attention.self.value',
        # 'attention.output',
    ]
    for n, p in model.named_parameters():
        if any([k in n for k in trainable_layers]):
            p.requires_grad = True
        else:
            p.requires_grad = False

    num_trainable, num_frozen = compute_num_trainable_params(model)
    print(f'Number of trainable parameters: {num_trainable:,}')
    print(f'Number of frozen parameters: {num_frozen:,}')

    assert num_trainable > 0, num_trainable

    model = model.cuda()

    optimizer = create_optimizer(model=model, lr=args.init_lr)

    batch_size = int(args.train_batch_size * args.gradient_accum_steps)
    steps_per_epoch = len(train_loader) // args.gradient_accum_steps
    max_train_steps = steps_per_epoch * args.num_epochs

    scheduler = CosineDecayWithWarmupLRScheduler(
        optimizer=optimizer,
        init_lr=args.init_lr,
        max_lr=args.max_lr,
        min_lr=args.min_lr,
        warmup_steps=int(args.warmup_ratio * max_train_steps),
        max_decay_steps=max_train_steps,
    )

    print(f'Starting to run {args.num_epochs} training epochs, total of {max_train_steps} steps, with batch size {batch_size}')
    val_epochs_loss = []

    for epoch in range(1, args.num_epochs + 1):  # for each epoch
        print(f'Start epoch {epoch}')

        t0 = time.time()
        train_losses = run_train_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            gradient_accum_steps=args.gradient_accum_steps,
            loss_margin=args.loss_margin,
            grad_clip=args.grad_clip,
            scheduler=scheduler,
            scaler=scaler,
        )

        t1 = time.time()

        train_stats = {'loss': torch.tensor(train_losses).mean().item()}
        train_stats['learning_rate'] = optimizer.param_groups[0]['lr']
        train_stats['time'] = t1 - t0
        log_statistics(tb_writer, epoch, train_stats, True)

        t2 = time.time()
        val_losses = run_validation_epoch(
            model=model,
            loader=val_loader,
            loss_margin=args.loss_margin,
        )
        t3 = time.time()

        val_stats = {'loss': torch.tensor(val_losses).mean().item()}
        val_stats['time'] = t3 - t2
        log_statistics(tb_writer, epoch, val_stats, False)
        val_epochs_loss.append(val_stats['loss'])

        # create checkpoint
        ckpt_path = os.path.join(args.ckpt_dir, f'epoch-{epoch}')
        model.save_pretrained(ckpt_path)
        print(f'Model checkpoint saved at {ckpt_path}')

        # Check if validation loss is increasing for the last N consecutive epochs
        if len(val_epochs_loss) > args.earlystop_window:
            check_results = []
            for i in range(len(val_epochs_loss) - 1, len(val_epochs_loss) - 1 - args.earlystop_window, -1):
                curr_loss = val_epochs_loss[i]
                prev_loss = val_epochs_loss[i - 1]
                if curr_loss > prev_loss:
                    check_results.append(True)
                else:
                    check_results.append(False)

            if all(check_results):
                print('Stopping training due to early stopping based on validation loss increases.')
                break


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', help='Dir contains the train.pk, validation.pk, and meta.json files for fine-tune the embedding model', type=str, default='./datasets/embed_alert_codes')
    parser.add_argument('--logs_dir', help='Tensorboard logs dir', type=str, default='./logs/finetune_embedding')
    parser.add_argument('--ckpt_dir', help='Checkpoints dir', type=str, default='./checkpoints/finetune_embedding')
    parser.add_argument('--max_seq_len', help='Max sequence length of the embedding model', type=int, default=512)
    parser.add_argument('--train_batch_size', help='Micro batch size for training', type=int, default=32)
    parser.add_argument('--gradient_accum_steps', help='Accumulate gradient steps during training', type=int, default=1)
    parser.add_argument('--val_batch_size', help='Batch size for validation', type=int, default=64)
    parser.add_argument('--mixed_precision', help='Use mixed precision during training', type=bool, default=False)
    parser.add_argument('--num_epochs', help='Number of epochs to train', type=int, default=50)
    parser.add_argument('--init_lr', help='Initial learning rate', type=float, default=1e-6)
    parser.add_argument('--max_lr', help='Maximum learning rate after warm up', type=float, default=2e-5)
    parser.add_argument('--min_lr', help='Minimum learning rate after decay', type=float, default=1e-5)
    parser.add_argument('--warmup_ratio', help='Learning rate warm up', type=float, default=0.05)
    parser.add_argument('--loss_margin', help='Constant for the cosine embedding loss, should be in [-1, 1]', type=float, default=0.0)
    parser.add_argument('--grad_clip', help='Clip gradient norm', type=float, default=1.0)
    parser.add_argument('--earlystop_window', help='Stop training if the validation loss is increasing over the last N consecutive epochs', type=int, default=5)
    parser.add_argument('--seed', help='Runtime seed', type=int, default=173)

    main()
