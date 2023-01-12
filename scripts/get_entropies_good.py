from typing import List, Optional
import os
import pickle
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
DEVICE = 'cuda'
def predictive_entropy(predictions: torch.Tensor) -> float:
    """Entropy calculation

    Args:
        predictions (torch.Tensor): predictions to get the entropy from

    Returns:
        float: entropy of prediction
    """
    epsilon = sys.float_info.min
    predictive_entropy = -np.sum( np.mean(predictions, axis=0) * np.log(np.mean(predictions, axis=0) + epsilon),
            axis=-1)
#    print("Calculating", epsilon, "Entropy of shape", predictions.shape, 'and ans is', predictive_entropy)

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
        # roberta uses an extra separator b/w pairs of sentences, 
        # cf. github.com/pytorch/fairseq/commit/1684e166e3da03f5b600dbb7855cb98ddfcd0805
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

def get_k_predictions(model, dataset, model_type, num_labels, seq_length, k):
    
    # use gpu
    device = torch.device(DEVICE)
    
    # set dataset
    sampler = SequentialSampler(dataset)
    dataloader = DataLoader(
        dataset, 
        sampler=sampler, 
        shuffle=False, # suffle should be false
        batch_size=64
    )

    # freeze state of model model batch stats etc
    model.eval()
    
    # allow dropout to change for Monte Carlo stuffs
    for module in model.modules():
        if module.__class__.__name__.startswith('Dropout'):
            module.train()
    
    # bag of predictions
    bag_predictions = torch.empty((len(dataset), seq_length, k, num_labels), device=device)
    labels = torch.empty((len(dataset), seq_length), device=device)
    bag_labels = torch.empty((len(dataset), seq_length, k), device=device)

    for i in range(k):
        
        # (batch_size, max seq length, different tokens labels)
        predictions = None
        mylabels = None 
        # labels (batch_size, max seq length, token)
        
        for batch in dataloader:
            batch = tuple(t.to(device) for t in batch)
            with torch.no_grad():
                inputs = {
                    "input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}

                if model_type != "distilbert":
                    inputs["token_type_ids"] = (
                        batch[2] if model_type in ["bert", "xlnet"] else None
                    )  # XLM and RoBERTa don"t use segment_ids
                
                # forward pass
                outputs = model(**inputs)
                logits = outputs[1]
                
                # predictions to predictions
                if predictions is not None:
                    predictions = torch.vstack((predictions, F.softmax(logits, dim=-1)))
                else:
                    predictions = F.softmax(logits, dim=-1)
                if mylabels is None:
                    mylabels = batch[3]
                else:
                    mylabels = torch.vstack((mylabels, batch[3]))
        # append the predictions to 
        bag_predictions[:, :, i, :] = predictions
        bag_labels[:, :, i] = mylabels
    
    assert torch.all(torch.std(bag_labels, dim=-1) == 0)
    print("HERE I have my labels", mylabels.shape)
    return bag_predictions, mylabels


def main(
    model_type: str,
    corruption_name: str,
    param: str,
    language: str,
    seed: str,
    number_of_predictions: int,
):
    # initialize certain params
    seq_length = 200
    
    # set paths
    weights_path = f"results/{model_type}/{corruption_name}/{param}/{language}/{seed}"
    my_dir = weights_path.replace('results/', 'entropies_v2/')
    if os.path.exists(f"{my_dir}/entropies.npz"): 
        print("Exit Early")
        return
    data_path = f"data/{corruption_name}/{param}/{language}"
    data_path = f"data/original/1/{language}"
    
    # get weights and stuffs
    tokenizer = AutoTokenizer.from_pretrained(weights_path)
    model = AutoModelForTokenClassification.from_pretrained(weights_path).to(DEVICE)
    labels = get_labels()
    
    # get the test dataset for the model
    dataset = load_and_cache_examples(
        seq_length, 
        data_path, 
        model, 
        tokenizer,
        labels,
        CrossEntropyLoss().ignore_index,
        "test"
    )
    print("DATASET", len(dataset.tensors), "LEN", dataset.tensors[0].shape, dataset.tensors[1].shape)
    # get predictions as (num_samples x seq_length x k x num labels)
    predictions, all_my_labels = get_k_predictions(
        model, 
        dataset, 
        model_type, 
        len(labels),
        seq_length,
        number_of_predictions
    )
    print("HEY THERE", predictions.shape)
    # get entropies
    entropies = []
    
    # calculate entropy for each sequence in the dataset
    for i in range(len(predictions)):
        
        # calculate entropy for each tokens in a sequence
        entropy_seq_length = []
        for j in range(len(predictions[0])):
            entropy_seq_length.append(predictive_entropy(predictions[i][j].detach().cpu().numpy()))
        entropies.append(entropy_seq_length)
        
    entropies = np.array(entropies)
    
    
    # save the entropies
    #entropies_path = f"{weights_path}/entropies.npz"
    os.makedirs(my_dir, exist_ok=True)
    entropies_path = f"{my_dir}/entropies.npz"
    np.savez(entropies_path, entropies)
    
    # pred_path = f"{my_dir}/predictions.npz"
    # np.savez(pred_path, predictions.detach().cpu().numpy())

    mylabels_path = f"{my_dir}/labels.npz"
    np.savez(mylabels_path, all_my_labels.detach().cpu().numpy())
    
    labels_path = f"{my_dir}/labels.p"
    with open(labels_path, 'wb+') as f:
        pickle.dump(labels, f)
    # TODO:
    # save labels and label attached to entropies

if __name__ == "__main__":
    torch.manual_seed(42)
    fire.Fire(main)
