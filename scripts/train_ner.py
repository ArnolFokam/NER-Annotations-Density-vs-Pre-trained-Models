import hydra
from omegaconf import DictConfi

@hydra.main(version_base=None, config_path=None)
def train_ner(cfg: DictConfig):
    """Trains NER models for different datasets"""

if __name__ == "__main__":
    train_ner()
