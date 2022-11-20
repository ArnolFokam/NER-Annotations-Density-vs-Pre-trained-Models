from typing import List, Optional
from shutil import copyfile
import logging

import fire
from tqdm import tqdm
from ner.corruption.corruption import delete_percentage_of_labels, delete_percentage_of_sentences, write_modified_examples_general

from ner.dataset import read_examples_from_file

log = logging.getLogger(__name__)

perc_delete = [{'perc_delete': i / 10} for i in range(0, 10)]

ALL_FUNCS_PARAMS = {
    'global_corruption': (delete_percentage_of_labels, perc_delete),
    'remove_sentences' : (delete_percentage_of_sentences, perc_delete),
}


def main(
    root_dir: str,
    corruption_name,
    languages: Optional[List[str]] = ["swa", "kin", "pcm", "en-conll-2003"],):

    func, corruption_params = ALL_FUNCS_PARAMS[corruption_name]
    for lang in languages:
        for param in corruption_params:
            li = list(param.values())
            assert len(li) == 1
            examples = read_examples_from_file(f'{root_dir}/og/{lang}', 'train')
            new_filename = f'{root_dir}/{corruption_name}/{li[0]}/{lang}/'
            write_modified_examples_general('train', examples, new_filename, func, func_kwargs=param)
            copyfile(f'{root_dir}/og/{lang}/dev.txt', f'{new_filename}/dev.txt')
            copyfile(f'{root_dir}/og/{lang}/test.txt', f'{new_filename}/test.txt')


if __name__ == "__main__":
    fire.Fire(main)
    