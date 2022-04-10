import math
import os
from typing import Any

import pytorch_lightning as pl
import torch
import torch.nn as nn
from deepspeed.profiling.flops_profiler import FlopsProfiler
from pytorch_lightning import Trainer, seed_everything, Callback
from pytorch_lightning.strategies import DeepSpeedStrategy
from pytorch_lightning.utilities import rank_zero_info
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, RandomSampler

from xformers.factory.model_factory import xFormer, xFormerConfig


class GPT(pl.LightningModule):

    def __init__(
            self,
            vocab_size,
            weight_decay=0.1,
            betas=(0.9, 0.95),
            learning_rate=6e-4,
            n_embd=512,
            block_size=128,
            n_layer=8,
            n_head=8,
            resid_pdrop=0.1,
            attn_pdrop=0.1,
            mlp_pdrop=0.1,
            attention="scaled_dot_product",
            hidden_layer_multiplier=4,
            warmup_tokens=20,
            final_tokens=1000,
    ):
        super().__init__()

        # auto creates self.hparams from the method signature
        self.save_hyperparameters()

        # A list of the encoder or decoder blocks which constitute the Transformer.
        xformer_config = [
            {
                "reversible": False,  # Turn on to test the effect of using reversible layers
                "block_type": "encoder",
                "num_layers": self.hparams.n_layer,
                "dim_model": self.hparams.n_embd,
                "layer_norm_style": "pre",
                "position_encoding_config": {
                    "name": "vocab",
                    "seq_len": self.hparams.block_size,
                    "vocab_size": self.hparams.vocab_size,
                },
                "multi_head_config": {
                    "num_heads": self.hparams.n_head,
                    "residual_dropout": self.hparams.resid_pdrop,
                    "use_rotary_embeddings": True,
                    "attention": {
                        "name": self.hparams.attention,
                        "dropout": self.hparams.attn_pdrop,
                        "causal": True,
                        "seq_len": self.hparams.block_size,
                        "num_rules": self.hparams.n_head,
                    },
                },
                "feedforward_config": {
                    "name": "FusedMLP",  # Use MLP if Triton is not available
                    "dropout": self.hparams.mlp_pdrop,
                    "activation": "gelu",
                    "hidden_layer_multiplier": self.hparams.hidden_layer_multiplier,
                },
            }
        ]

        config = xFormerConfig(xformer_config)
        self.model = xFormer.from_config(config)

        # decoder head
        self.ln_f = nn.LayerNorm(self.hparams.n_embd)
        self.head = nn.Linear(self.hparams.n_embd, self.hparams.vocab_size, bias=False)

        self.block_size = self.hparams.block_size
        self.apply(self._init_weights)

        self._tokens_seen = 0

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

        # Reset the token counter
        self._tokens_seen = 0

    def get_block_size(self):
        return self.block_size

    def configure_optimizers(self):
        # Create the optimizer and the training schedule:
        # - Handle the per-param weight decay
        no_decay = ["bias", "LayerNorm.weight"]
        params_decay = [
            p for n, p in self.named_parameters() if not any(nd in n for nd in no_decay)
        ]
        params_nodecay = [
            p for n, p in self.named_parameters() if any(nd in n for nd in no_decay)
        ]
        optim_groups = [
            {"params": params_decay, "weight_decay": self.hparams.weight_decay},
            {"params": params_nodecay, "weight_decay": 0.0},
        ]

        # - Start with a warm up, ramp up then cosine
        optimizer = torch.optim.AdamW(
            optim_groups, lr=self.hparams.learning_rate, betas=self.hparams.betas
        )

        def update_lr(*_):
            config = self.hparams

            if self._tokens_seen < config.warmup_tokens:
                # linear warmup
                lr_mult = float(self._tokens_seen) / float(max(1, config.warmup_tokens))
                lr_mult = max(lr_mult, 1e-2)  # could be that we've not seen any yet
            else:
                # cosine learning rate decay
                progress = float(self._tokens_seen - config.warmup_tokens) / float(
                    max(1, config.final_tokens - config.warmup_tokens)
                )
                lr_mult = max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))

            return lr_mult

        lr_scheduler = {
            "scheduler": torch.optim.lr_scheduler.LambdaLR(
                optimizer,
                lr_lambda=[update_lr, update_lr],
            ),
            "name": "learning_rate",
            "interval": "step",  # The unit of the scheduler's step size
            "frequency": 1,  # The frequency of the scheduler
        }
        return [optimizer], [lr_scheduler]

    def forward(self, src):
        # predict the next tokens (in latent space)
        prediction = self.model(src)

        # translate the predictions into tokens
        prediction = self.ln_f(prediction)
        logits = self.head(prediction)

        return logits

    def training_step(self, batch, _):
        src, targets = batch

        # Update the tokens we've seen (tracked for LR scheduling)
        self._tokens_seen += (src >= 0).numel()

        # same action as inference
        logits = self(src)

        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        self.logger.log_metrics(
            {
                "train_loss": loss.mean(),
                "learning_rate": self.lr_schedulers().get_last_lr()[0],
            },
            step=trainer.global_step,
        )

        return loss


class CharDataset(Dataset):
    def __init__(self, data, block_size):
        chars = list(set(data))
        data_size, vocab_size = len(data), len(chars)
        rank_zero_info("data has %d characters, %d unique." % (data_size, vocab_size))

        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}
        self.block_size = block_size
        self.vocab_size = vocab_size
        self.data = data

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, i):
        chunk = self.data[i: i + self.block_size + 1]
        dix = [self.stoi[s] for s in chunk]

        # src and target are off by one, we want the model to predict the next word
        x = torch.tensor(dix[:-1], dtype=torch.long)
        y = torch.tensor(dix[1:], dtype=torch.long)
        return x, y

    def to_tokens(self, message, device):
        return torch.tensor([self.stoi[s] for s in message], dtype=torch.long)[
            None, ...
        ].to(device)

    def from_tokens(self, tokens):
        return "".join([self.itos[int(i)] for i in tokens])


class DeepSpeedProfiler(Callback):

    def __init__(self, profile_step: int = 20):
        self.profile_step = profile_step
        self.prof = None

    def on_train_batch_start(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        batch: Any,
        batch_idx: int,
        unused: int = 0,
    ) -> None:
        if self.prof is None:
            self.prof = FlopsProfiler(pl_module)
        if batch_idx == self.profile_step:
            self.prof.start_profile()

    def on_train_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs,
        batch: Any,
        batch_idx: int,
        unused: int = 0,
    ) -> None:
        if batch_idx == self.profile_step:
            flops = self.prof.get_total_flops(as_string=True)
            params = self.prof.get_total_params(as_string=True)
            self.prof.print_model_profile(batch_idx, detailed=False)
            self.prof.end_profile()


if __name__ == "__main__":
    seed_everything(42)

    # Adjust batch depending on the available memory on your machine.
    # You can also use reversible layers to save memory
    REF_BATCH = 512
    BATCH = 256

    WORKERS = 4
    EPOCHS = 1
    BLOCK_SIZE = 128
    WARMUP = 20

    if not os.path.exists("input.txt"):
        os.system(
            "wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        )

    text = open("input.txt", "r").read()
    train_dataset = CharDataset(text, BLOCK_SIZE)
    train_loader = DataLoader(
        train_dataset,
        sampler=RandomSampler(train_dataset),
        batch_size=BATCH,
        num_workers=WORKERS,
        pin_memory=True,
    )

    model = GPT(
        vocab_size=train_dataset.vocab_size,
        block_size=train_dataset.block_size,
        attention="scaled_dot_product",
        warmup_tokens=REF_BATCH * WARMUP,
        final_tokens=EPOCHS * len(train_dataset) * BLOCK_SIZE,
    )

    trainer = Trainer(
        gpus=1,
        strategy=DeepSpeedStrategy(stage=2),
        callbacks=DeepSpeedProfiler(),
        limit_train_batches=50,
        max_epochs=1,
        # max_epochs=EPOCHS,
        precision=16,
        gradient_clip_val=1,  # Use to catch divergent gradients, if experimenting
        log_every_n_steps=1,
        accumulate_grad_batches=REF_BATCH // BATCH,
    )

    trainer.fit(model, train_loader)
