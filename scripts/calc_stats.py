from collections import defaultdict
from typing import List, Optional
from shutil import copyfile
import logging

import fire
from tqdm import tqdm
from ner.corruption.corruption import delete_percentage_of_labels, write_modified_examples_general

from ner.dataset import read_examples_from_file
from scripts.create_corrupted_datasets import ALL_FUNCS_PARAMS

log = logging.getLogger(__name__)

perc_delete = [{'perc_delete': i / 10} for i in range(0, 10)]


def main(
    root_dir: str,
    corruption_name,
    languages: Optional[List[str]] = ["swa", "kin", "pcm", "en-conll-2003"],):

    func, corruption_params = ALL_FUNCS_PARAMS[corruption_name]
    for lang in languages:
        for param in corruption_params:
            stats = defaultdict(lambda : 0)
            li = list(param.values())
            assert len(li) == 1
            examples = read_examples_from_file(f'{root_dir}/{corruption_name}/{li[0]}/{lang}/', 'train')
            for ex in examples:
                for l in ex.labels:
                    if l == 'O': continue
                    stats[l] += 1
            stats = dict(stats)
            print(f"{lang:<10} {li[0]} {stats}")
                


if __name__ == "__main__":
    fire.Fire(main)
    