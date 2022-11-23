from typing import List, Optional
from shutil import copyfile
import logging

import fire
from ner.corruption.corruption import keep_percentage_of_labels, keep_percentage_of_sentences, swap_percentage_of_labels, cap_number_of_labels, swap_number_of_labels, write_modified_examples_general
from ner.dataset import read_examples_from_file

log = logging.getLogger(__name__)

percentage = [{'percentage': i / 10} for i in range(1, 11)]
number = [{'number': i} for i in range(1, 11)]

ALL_FUNCS_PARAMS = {
    # global corruption
    'global_cap_labels':        (keep_percentage_of_labels, percentage),
    'global_cap_sentences' :    (keep_percentage_of_sentences, percentage),
    'global_swap_labels':       (swap_percentage_of_labels, percentage),
    
    # local corruption
    'local_cap_labels':         (cap_number_of_labels, number),
    'local_swap_labels':        (swap_number_of_labels, number),
}


def main(
    root_dir: str,
    corruption_name,
    # languages: Optional[List[str]] = ["swa", "kin", "pcm", "en-conll-2003"],):
    languages: Optional[List[str]] = ['amh','conll_2003_en','hau','ibo','kin','lug','luo','pcm','swa','wol','yor',],):

    func, corruption_params = ALL_FUNCS_PARAMS[corruption_name]
    for lang in languages:
        for param in corruption_params:
            
            log.info(f"Preprocessing using the function '{func.__name__}' with params {param}")
            
            li = list(param.values())
            assert len(li) == 1
            examples = read_examples_from_file(f'{root_dir}/original/{lang}', 'train')
            new_filename = f'{root_dir}/{corruption_name}/{li[0]}/{lang}/'
            write_modified_examples_general('train', examples, new_filename, func, func_kwargs=param)
            copyfile(f'{root_dir}/original/{lang}/dev.txt', f'{new_filename}/dev.txt')
            copyfile(f'{root_dir}/original/{lang}/test.txt', f'{new_filename}/test.txt')


if __name__ == "__main__":
    fire.Fire(main)

