import logging
import os

import torch
from datasets import load_dataset
from torch.utils.data import IterableDataset
from transformers import AutoTokenizer
from transformers.utils import logging


class CharDataset(IterableDataset):
    def __init__(self, block_size):
        # some HF boilerplate
        logging.set_verbosity(40)
        os.environ["TOKENIZERS_PARALLELISM"] = "TRUE"
        ds = load_dataset(
            "oscar", "unshuffled_deduplicated_en", split="train", streaming=True
        )
        ds = ds.with_format("torch")
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        self.block_size = block_size

        # add one more token so that we can shift the labels (labels are the next word)
        block_size = block_size + 1

        def convert_to_features(examples):
            examples = self.tokenizer(examples["text"])
            # Concatenate all texts.
            concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
            total_length = len(concatenated_examples[list(examples.keys())[0]])
            # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
            # customize this part to your needs.
            total_length = (total_length // block_size) * block_size
            # Split by chunks of max_len.
            result = {
                k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
                for k, t in concatenated_examples.items()
            }
            return result

        ds = ds.map(convert_to_features, remove_columns=["text", "id"], batched=True)

        self.ds = ds

    def __iter__(self):
        self.ds.shuffle()
        for chunk in self.ds:
            chunk = chunk["input_ids"]
            # src and target are off by one, we want the model to predict the next word
            x = torch.tensor(chunk[:-1], dtype=torch.long)
            y = torch.tensor(chunk[1:], dtype=torch.long)
            yield x, y

    @property
    def vocab_size(self) -> int:
        return len(self.tokenizer)
