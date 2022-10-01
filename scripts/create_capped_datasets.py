from typing import List, Optional
from shutil import copyfile
import logging

import fire
from tqdm import tqdm

from ner.dataset import read_examples_from_file, write_modified_examples

log = logging.getLogger(__name__)

def main(
    root_dir: str,
    caps: Optional[List[int]] = None,
    languages: Optional[List[str]] = ["swa", "kin", "pcm", "en-conllpp"],):

    caps = caps if caps else list(range(1, 11))

    log.info(f"Create {len(caps)} for each language dataset: {caps}")

    for lang in languages:

        pbar = tqdm(caps)
        pbar.set_description(f"Capping dataset for language {lang}")

        for cap in pbar:
            examples = read_examples_from_file(f'{root_dir}/{lang}', 'train')
            write_modified_examples(f'{root_dir}/{lang}', 'train', examples, cap=cap)
            copyfile(f'{root_dir}/{lang}/dev.txt', f'{root_dir}/{lang}/cap-{cap}/dev.txt')
            copyfile(f'{root_dir}/{lang}/test.txt', f'{root_dir}/{lang}/cap-{cap}/test.txt')


if __name__ == "__main__":
    fire.Fire(main)