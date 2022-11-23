import copy
import os
from typing import List
import random

import numpy as np
from ner.dataset import InputExample, get_labels_position

labels_suffix = ["PER", "ORG", "LOC", "DATE"]


def write_modified_examples_general(mode: str, X: List[InputExample], new_filename: str,
                            function_to_use_to_process, func_kwargs = {}):
        
    folder_path = new_filename
    os.makedirs(folder_path, exist_ok=True)
        
    file_path =  os.path.join(folder_path, "{}.txt".format(mode))
    
    n_examples = function_to_use_to_process(X, **func_kwargs)
    
    with open(file_path, "w+", encoding="utf-8") as f:
        for ex in n_examples:
            for i in range(len(ex.labels)):
                f.write(f"{ex.words[i]} {ex.labels[i]}\n")
            f.write("\n")
            
    f.close()

def cap_number_of_labels(X: List[InputExample], number: int) -> List[InputExample]:
    n_examples = []
    
    for ex in X:
        ex = copy.deepcopy(ex)
        labels_pos = get_labels_position(ex.labels.copy())
        labels = ex.labels.copy()
        
        # randomly sample different entities
        for (x, y) in random.sample(labels_pos, max(len(labels_pos) - number, 0)):
            labels[x:y + 1] = ['O' for _ in range(len(labels[x:y + 1]))]
            
        ex.labels = labels
        n_examples.append(ex)
            
    return n_examples

def swap_number_of_labels(X: List[InputExample], number: int) -> List[InputExample]:
    n_examples = []
    
    for ex in X:
        ex = copy.deepcopy(ex)
        labels_pos = get_labels_position(ex.labels.copy())
        labels = ex.labels.copy()
        
        # randomly sample different entities
        for (x, y) in random.sample(labels_pos, min(len(labels_pos), number)):
            
            # choose a label at random excluding the label we currently have
            current = labels[x].split("-")[-1]
            available_labels = labels_suffix.copy()
            available_labels.remove(current)
            label = np.random.choice(available_labels)

            labels[x] = f"B-{label}"
            labels[x + 1:y + 1] = [f"I-{label}" for _ in range(len(labels[x + 1:y + 1]))]
            
        ex.labels = labels
        n_examples.append(ex)
            
    return n_examples

def keep_percentage_of_labels(X: List[InputExample], percentage: float) -> List[InputExample]:
    n_examples = []
    
    all_possible_values = []
    for example_index, ex in enumerate(X):
        copies = ex.labels.copy()
        labels_pos = get_labels_position(copies)
        all_possible_values.extend(
            [(example_index, temp) for temp in labels_pos]
        )
        n_examples.append(InputExample(ex.guid, ex.words.copy(), copies))
    # Now, we must keep only a portion of these ones.
    ones_to_delete = np.random.choice(np.arange(len(all_possible_values)), size=int(np.round((1 - percentage) * len(all_possible_values))), replace=False)
    for idx in ones_to_delete:
        example_index, temp = all_possible_values[idx]
        for j in range(temp[0], temp[1] + 1):
            n_examples[example_index].labels[j] = 'O'   
    return n_examples

def swap_percentage_of_labels(X: List[InputExample], percentage: float) -> List[InputExample]:
    n_examples = []
    
    all_possible_values = []
    for example_index, ex in enumerate(X):
        ex = copy.deepcopy(ex)
        copies = ex.labels.copy()
        labels_pos = get_labels_position(copies)
        all_possible_values.extend(
            [(example_index, temp) for temp in labels_pos]
        )
        n_examples.append(InputExample(ex.guid, ex.words.copy(), copies))
    # Now, we must keep only a portion of these ones.
    ones_to_swap = np.random.choice(np.arange(len(all_possible_values)), size=int(np.round((1 - percentage) * len(all_possible_values))), replace=False)
    for idx in ones_to_swap:
        example_index, temp = all_possible_values[idx]
        
        # choose a label at random excluding the label we currently have
        current = n_examples[example_index].labels[temp[0]].split("-")[-1]
        available_labels = labels_suffix.copy()
        available_labels.remove(current)
        label = np.random.choice(available_labels)
        
        # swap the labels
        n_examples[example_index].labels[temp[0]] = f"B-{label}"
        for j in range(temp[0] + 1, temp[1] + 1):
            n_examples[example_index].labels[j] = f'I-{label}'   
            
    return n_examples

def keep_percentage_of_sentences(X: List[InputExample], percentage: float) -> List[InputExample]:
    ones_to_delete = set(np.random.choice(np.arange(len(X)), size=int(np.round((1 - percentage) * len(X))), replace=False))
    return [copy.deepcopy(ex) for i, ex in enumerate(X) if i not in ones_to_delete]