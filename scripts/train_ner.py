import logging

import hydra
from omegaconf import DictConfig
import wandb

from transformers import (
    WEIGHTS_NAME,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoConfig,
    AutoTokenizer,
    AutoModel,
    AdamW,
    BertConfig,
    BertForTokenClassification,
    BertTokenizer,
    RobertaConfig,
    RobertaForTokenClassification,
    RobertaTokenizer,
    get_linear_schedule_with_warmup,
    get_constant_schedule_with_warmup,
)

logger = logging.getLogger("Afri_NER_Log")
logging.basicConfig(level=logging.INFO)

MODEL_CLASSES = {
    "bert": (BertConfig, BertForTokenClassification, BertTokenizer),
    "roberta": (RobertaConfig, RobertaForTokenClassification, RobertaTokenizer),
}

@hydra.main(version_base=None, config_path=None)
def train_ner(cfg: DictConfig):
    """Trains NER models for different datasets"""

if __name__ == "__main__":
    train_ner()
