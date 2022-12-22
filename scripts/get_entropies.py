from typing import List, Optional

import fire
import torch
import sys
import numpy as np
import torch.nn.functional as F

import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import TensorDataset
from ner.dataset import convert_examples_to_features, read_examples_from_file
from torch.utils.data import SequentialSampler
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForTokenClassification
from ner.dataset import get_labels

def predictive_entropy(predictions):
    epsilon = sys.float_info.min
    predictive_entropy = -np.sum( np.mean(predictions, axis=0) * np.log(np.mean(predictions, axis=0) + epsilon),
            axis=-1)

    return predictive_entropy

def load_and_cache_examples(max_seq_length, data_path, model_type, tokenizer, labels, pad_token_label_id, mode):
    examples = read_examples_from_file(data_path, mode)
    features = convert_examples_to_features(
        examples,
        labels,
        max_seq_length,
        tokenizer,
        cls_token_at_end=bool(model_type in ["xlnet"]),
        # xlnet has a cls token at the end
        cls_token=tokenizer.cls_token,
        cls_token_segment_id=2 if model_type in ["xlnet"] else 0,
        sep_token=tokenizer.sep_token,
        sep_token_extra=bool(model_type in ["roberta"]),
        # roberta uses an extra separator b/w pairs of sentences, cf. github.com/pytorch/fairseq/commit/1684e166e3da03f5b600dbb7855cb98ddfcd0805
        pad_on_left=bool(model_type in ["xlnet"]),
        # pad on the left for xlnet
        pad_token=tokenizer.convert_tokens_to_ids(
            [tokenizer.pad_token])[0],
        pad_token_segment_id=4 if model_type in ["xlnet"] else 0,
        pad_token_label_id=pad_token_label_id,
    )

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor(
        [f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor(
        [f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor(
        [f.segment_ids for f in features], dtype=torch.long)
    all_label_ids = torch.tensor(
        [f.label_ids for f in features], dtype=torch.long)
    all_ids = torch.tensor(
        [int(f.idx.split('-')[-1]) for f in features], dtype=torch.long)

    dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_ids)
    return dataset

def get_k_predictions(model, dataset, model_type, k):
    device = torch.device("cuda")
    sampler = SequentialSampler(dataset)
    dataloader = DataLoader(dataset, sampler=sampler, shuffle=False, batch_size=64)

    model.eval()

    model.eval()
    for module in model.modules():
        if module.__class__.__name__.startswith('Dropout'):
            module.train()
    
    bag_predictions = torch.empty((3453, 200, k, 9), device=device)

    for _ in range(k):
        predictions = None # (batch_size, max seq length, different tokens labels)
        
        for batch in dataloader:
            batch = tuple(t.to(device) for t in batch)
            with torch.no_grad():
                inputs = {
                    "input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}

                if model_type != "distilbert":
                    inputs["token_type_ids"] = (
                        batch[2] if model_type in ["bert", "xlnet"] else None
                    )  # XLM and RoBERTa don"t use segment_ids
                outputs = model(**inputs)
                loss, logits = outputs[:2]
                    
                if predictions is not None:
                    predictions = torch.cat((predictions, F.softmax(logits, dim=-1)))
                else:
                    predictions = F.softmax(logits, dim=-1)
                
        
    return bag_predictions


def main(
    root_dir: str,
    corruption_name: str,
    param: str,
    model_conf: List[str],
    saved_things: str,
    number_of_predictions: int,
    language):
    
    tokenizer = AutoTokenizer.from_pretrained(saved_things)
    model = AutoModelForTokenClassification.from_pretrained(saved_things)
    labels = get_labels()
    
    dataset = load_and_cache_examples(
        200, 
        f"../data/{corruption_name}/{param}/{language}", 
        model_conf[0], 
        tokenizer,
        labels,
        CrossEntropyLoss().ignore_index,
        "test"
    )
    
    predictions = get_k_predictions(model, dataset, model_conf[0], number_of_predictions)
    
    entropies = []

    for i in range(len(predictions)):
        entropy = []
        for j in range(len(predictions[0])):
            entropy.append(predictive_entropy(predictions[i][j].detach().cpu().numpy()))
            
        entropies.append(entropy)

if __name__ == "__main__":
    fire.Fire(main)