import copy
import os
from typing import List

import numpy as np
from ner.dataset import InputExample, get_capped_labels, get_labels_position

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


def get_capped_examples(X: List[InputExample], cap):
    n_examples = []
    for ex in X:
        labels_pos = get_labels_position(ex.labels.copy())
        ex.labels = get_capped_labels(ex.labels.copy(), labels_pos, cap)
        n_examples.append(ex)
            
    return n_examples

def delete_percentage_of_labels(X: List[InputExample], perc_delete: float) -> List[InputExample]:
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
    ones_to_delete = np.random.choice(np.arange(len(all_possible_values)), size=int(np.round((perc_delete) * len(all_possible_values))), replace=False)
    for idx in ones_to_delete:
        example_index, temp = all_possible_values[idx]
        for j in range(temp[0], temp[1] + 1):
            n_examples[example_index].labels[j] = 'O'   
    return n_examples

def delete_percentage_of_sentences(X: List[InputExample], perc_delete: float) -> List[InputExample]:
    n_examples = []
    all_possible_values = []
    ones_to_delete = set(np.random.choice(np.arange(len(X)), size=int(np.round((perc_delete) * len(X))), replace=False))
    return [copy.deepcopy(ex) for i, ex in enumerate(X) if i not in ones_to_delete]
    for example_index, ex in enumerate(X):
        copies = ex.labels.copy()