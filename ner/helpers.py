import os

from ner.dataset import InputExample


def read_examples_from_file(data_dir, mode):
    file_path = os.path.join(data_dir, "{}.txt".format(mode))
    guid_index = 1
    examples = []
    with open(file_path, encoding="utf-8") as f:
        words = []
        labels = []
        for line in f:
            line = line.strip()
            if len(line) < 2  or line == "\n":
                # print(line, words)
                if words:
                    examples.append(InputExample(guid="{}-{}".format(mode, guid_index), words=words, labels=labels))
                    guid_index += 1
                    words = []
                    labels = []
            else:
                splits = line.split(" ")
                words.append(splits[0])
                if len(splits) > 1:
                    labels.append(splits[-1].replace("\n", ""))
                else:
                    # Examples could have no label for mode = "test"
                    labels.append("O")
        if words:
            examples.append(InputExample(guid="{}-{}".format(mode, guid_index), words=words, labels=labels))
    return examples


def get_labels_position(labels):
    label =  []
    in_word = False
    start_idx = -1

    for i in range(len(labels)):
        
        if labels[i].startswith('B-'):
            if in_word is True:
                label.append((start_idx, i-1))
                
            in_word = True
            start_idx = i
            
        elif labels[i] == 'O' and in_word == True:
            label.append((start_idx, i - 1))
            in_word = False
            
    return label

def get_capped_labels(labels, labels_pos, cap):
    import random
    random.seed(42)

    for (x, y) in random.sample(labels_pos, max(len(labels_pos) - cap, 0)):
        labels[x:y + 1] = ['O' for _ in range(len(labels[x:y + 1]))]
       
    return labels


def get_capped_examples(X, cap):
    n_examples = []
    for ex in X:
        labels_pos = get_labels_position(ex.labels.copy())
        ex.labels = get_capped_labels(ex.labels.copy(), labels_pos, cap)
        n_examples.append(ex)
            
    return n_examples


def write_modified_examples(data_dir, mode, X, cap):
        
    folder_path = os.path.join(data_dir, "cap-{}".format(cap))
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)
        
    file_path =  os.path.join(folder_path, "{}.txt".format(mode))
    
    n_examples = get_capped_examples(X, cap)
    
    with open(file_path, "w+", encoding="utf-8") as f:
        for ex in n_examples:
            for i in range(len(ex.labels)):
                f.write(f"{ex.words[i]} {ex.labels[i]}\n")
            f.write("\n")
            
    f.close()

def get_label_density_statistics(X):
    n_labels = []
    for ex in X:
        n_labels.append(len(get_labels_position(ex.labels.copy())))
        
    return np.mean(n_labels), np.std(n_labels), max(set(n_labels), key=n_labels.count), *np.unique(n_labels, return_counts=True), n_labels
