import glob
import logging
import os
import random
import shutil

import hydra
import wandb
import torch
import numpy as np
from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf
from seqeval.metrics import f1_score, precision_score, recall_score
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from transformers import (
    WEIGHTS_NAME,
    AutoModelForTokenClassification,
    AutoConfig,
    AutoTokenizer,
    BertConfig,
    BertForTokenClassification,
    BertTokenizer,
    RobertaConfig,
    RobertaForTokenClassification,
    RobertaTokenizer,
    get_linear_schedule_with_warmup,
    get_constant_schedule_with_warmup,
)

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

from ner.dataset import convert_examples_to_features, get_labels, read_examples_from_file
from ner.helpers import get_dir

load_dotenv()

logger = logging.getLogger("Afri_NER_Log")
logging.basicConfig(level=logging.INFO)

MODEL_CLASSES = {
    "bert": (BertConfig, BertForTokenClassification, BertTokenizer),
    "roberta": (RobertaConfig, RobertaForTokenClassification, RobertaTokenizer),
}


def set_seed(cfg):
    """Set seed for training"""
    random.seed(cfg.device.seed)
    np.random.seed(cfg.device.seed)
    torch.manual_seed(cfg.device.seed)
    if cfg.device.n_gpu > 0:
        torch.cuda.manual_seed_all(cfg.device.seed)


def load_and_cache_examples(cfg, tokenizer, labels, pad_token_label_id, mode):
    if cfg.device.local_rank not in [-1, 0] and not evaluate:
        # Make sure only the first process in distributed training process the dataset, and the others will use the cache
        torch.distributed.barrier()

    # Load data features from cache or dataset file
    cached_features_file = os.path.join(
        cfg.data.path,
        "cached_{}_{}_{}".format(
            mode, list(filter(None, cfg.model.name_or_path.split("/"))
                       ).pop(), str(cfg.data.max_seq_length)
        ),
    )
    if os.path.exists(cached_features_file) and not cfg.model.overwrite_cache:
        logger.info("Loading features from cached file %s",
                    cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", cfg.data.path)
        examples = read_examples_from_file(cfg.data.path, mode)
        features = convert_examples_to_features(
            examples,
            labels,
            cfg.data.max_seq_length,
            tokenizer,
            cls_token_at_end=bool(cfg.model.type in ["xlnet"]),
            # xlnet has a cls token at the end
            cls_token=tokenizer.cls_token,
            cls_token_segment_id=2 if cfg.model.type in ["xlnet"] else 0,
            sep_token=tokenizer.sep_token,
            sep_token_extra=bool(cfg.model.type in ["roberta"]),
            # roberta uses an extra separator b/w pairs of sentences, cf. github.com/pytorch/fairseq/commit/1684e166e3da03f5b600dbb7855cb98ddfcd0805
            pad_on_left=bool(cfg.model.type in ["xlnet"]),
            # pad on the left for xlnet
            pad_token=tokenizer.convert_tokens_to_ids(
                [tokenizer.pad_token])[0],
            pad_token_segment_id=4 if cfg.model.type in ["xlnet"] else 0,
            pad_token_label_id=pad_token_label_id,
        )
        if cfg.device.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s",
                        cached_features_file)
            torch.save(features, cached_features_file)

    if cfg.device.local_rank == 0 and not evaluate:
        # Make sure only the first process in distributed training process the dataset, and the others will use the cache
        torch.distributed.barrier()

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor(
        [f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor(
        [f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor(
        [f.segment_ids for f in features], dtype=torch.long)
    all_label_ids = torch.tensor(
        [f.label_ids for f in features], dtype=torch.long)

    dataset = TensorDataset(all_input_ids, all_input_mask,
                            all_segment_ids, all_label_ids)
    return dataset


def train(cfg, train_dataset, model, tokenizer, labels, pad_token_label_id, device, wandb_logger):
    """ Train the model """

    if cfg.device.local_rank in [-1, 0]:
        tb_writer = SummaryWriter()

    cfg.data.train_batch_size = cfg.data.per_gpu_train_batch_size * \
        max(1, cfg.device.n_gpu)
    train_sampler = RandomSampler(
        train_dataset) if cfg.device.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset, sampler=train_sampler, batch_size=cfg.data.train_batch_size)

    if cfg.optim.max_steps > 0:
        t_total = cfg.optim.max_steps
        cfg.optim.num_train_epochs = cfg.optim.max_steps // (
            len(train_dataloader) // cfg.optim.gradient_accumulation_steps) + 1
    else:
        t_total = len(
            train_dataloader) // cfg.optim.gradient_accumulation_steps * cfg.optim.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": cfg.optim.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(
            nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(
        optimizer_grouped_parameters,
        lr=cfg.optim.learning_rate, eps=cfg.optim.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=cfg.optim.warmup_steps, num_training_steps=t_total
    )

    # Check if saved optimizer or scheduler states exist
    if os.path.isfile(os.path.join(cfg.model.name_or_path, "optimizer.pt")) and os.path.isfile(
            os.path.join(cfg.model.name_or_path, "scheduler.pt")
    ):
        # Load in optimizer and scheduler states
        optimizer.load_state_dict(torch.load(
            os.path.join(cfg.model.name_or_path, "optimizer.pt")))
        scheduler.load_state_dict(torch.load(
            os.path.join(cfg.model.name_or_path, "scheduler.pt")))

    # multi-gpu training (should be after apex fp16 initialization)
    if cfg.device.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if cfg.device.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[
                cfg.device.local_rank], output_device=cfg.device.local_rank, find_unused_parameters=True
        )

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", cfg.optim.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d",
                cfg.data.per_gpu_train_batch_size)
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        cfg.data.train_batch_size
        * cfg.optim.gradient_accumulation_steps
        * (torch.distributed.get_world_size()
           if cfg.device.local_rank != -1 else 1),
    )
    logger.info("  Gradient Accumulation steps = %d",
                cfg.optim.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    epochs_trained = 0
    steps_trained_in_current_epoch = 0
    # Check if continuing training from a checkpoint
    if os.path.exists(cfg.model.name_or_path):
        # set global_step to gobal_step of last saved checkpoint from model path
        try:
            global_step = int(
                cfg.model.name_or_path.split("-")[-1].split("/")[0])
        except ValueError:
            global_step = 0
        epochs_trained = global_step // (len(train_dataloader) //
                                         cfg.optim.gradient_accumulation_steps)
        steps_trained_in_current_epoch = global_step % (
            len(train_dataloader) // cfg.optim.gradient_accumulation_steps)

        logger.info(
            "  Continuing training from checkpoint, will skip to saved global_step")
        logger.info("  Continuing training from epoch %d", epochs_trained)
        logger.info("  Continuing training from global step %d", global_step)
        logger.info("  Will skip the first %d steps in the first epoch",
                    steps_trained_in_current_epoch)

    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(
        epochs_trained, int(cfg.optim.num_train_epochs), desc="Epoch", position=0, leave=True, disable=cfg.device.local_rank not in [-1, 0]
    )
    set_seed(cfg)  # Added here for reproductibility
    for epoch in train_iterator:
        wandb_logger.log({"train-epoch": epoch, "step": global_step})
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", position=0,
                              leave=True, disable=cfg.device.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):

            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue

            model.train()
            batch = tuple(t.to(device) for t in batch)
            inputs = {
                "input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}
            if cfg.model.type != "distilbert":
                inputs["token_type_ids"] = (
                    batch[2] if cfg.model.type in ["bert", "xlnet"] else None
                )  # XLM and RoBERTa don"t use segment_ids

            outputs = model(**inputs)
            # model outputs are always tuple in pytorch-transformers (see doc)
            loss = outputs[0]

            if cfg.device.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if cfg.optim.gradient_accumulation_steps > 1:
                loss = loss / cfg.optim.gradient_accumulation_steps

            loss.backward()

            tr_loss += loss.item()
            if (step + 1) % cfg.optim.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), cfg.optim.max_grad_norm)

                scheduler.step()  # Update learning rate schedule
                optimizer.step()
                model.zero_grad()
                global_step += 1

                torch.cuda.empty_cache()

                if cfg.device.local_rank in [-1, 0] and cfg.logging.steps > 0 and global_step % cfg.logging.steps == 0:
                    # Log metrics
                    if (
                            cfg.device.local_rank == -1 and cfg.experiment.evaluate_during_training
                    ):  # Only evaluate when single GPU otherwise metrics may not average well
                        results, _ = evaluate(
                            cfg, model, tokenizer, labels, pad_token_label_id, mode="dev", device=device, wandb_logger=wandb_logger)
                        for key, value in results.items():
                            tb_writer.add_scalar(
                                "eval_{}".format(key), value, global_step)
                            wandb_logger.log(
                                {f"dev-{key}": value, "step": global_step})

                    tb_writer.add_scalar(
                        "lr", scheduler.get_lr()[0], global_step)
                    tb_writer.add_scalar(
                        "loss", (tr_loss - logging_loss) / cfg.logging.steps, global_step)
                    wandb_logger.log(
                        {"train-loss": (tr_loss - logging_loss) / cfg.logging.steps, "step": global_step})
                    wandb_logger.log(
                        {"train-lr": scheduler.get_lr()[0], "step": global_step})
                    logging_loss = tr_loss

                if cfg.device.local_rank in [-1, 0] and cfg.model.save_steps > 0 and global_step % cfg.model.save_steps == 0:
                    # Save model checkpoint
                    output_dir = os.path.join(
                        cfg.model.output_dir, "checkpoint-{}".format(global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = (
                        model.module if hasattr(model, "module") else model
                    )  # Take care of distributed/parallel training
                    model_to_save.save_pretrained(output_dir)
                    tokenizer.save_pretrained(output_dir)

                    torch.save(cfg, os.path.join(
                        output_dir, "training_cfg.bin"))
                    logger.info("Saving model checkpoint to %s", output_dir)

                    torch.save(optimizer.state_dict(), os.path.join(
                        output_dir, "optimizer.pt"))
                    torch.save(scheduler.state_dict(), os.path.join(
                        output_dir, "scheduler.pt"))
                    logger.info(
                        "Saving optimizer and scheduler states to %s", output_dir)

            if cfg.optim.max_steps > 0 and global_step > cfg.optim.max_steps:
                epoch_iterator.close()
                break
        if cfg.optim.max_steps > 0 and global_step > cfg.optim.max_steps:
            train_iterator.close()
            break

    if cfg.device.local_rank in [-1, 0]:
        tb_writer.close()

    return global_step, tr_loss / global_step


def evaluate(cfg, model, tokenizer, labels, pad_token_label_id, mode, device, wandb_logger=None, prefix=""):

    if cfg.logging.wandb.use and not wandb_logger:
        wandb_logger = wandb.init(
            project=str(cfg.experiment.name),
            dir=os.path.join(
                os.getcwd(),
                cfg.model.output_dir,
            ),
            group=str(cfg.experiment.group),
        )

    eval_dataset = load_and_cache_examples(
        cfg, tokenizer, labels, pad_token_label_id, mode=mode)

    cfg.data.eval_batch_size = cfg.data.per_gpu_eval_batch_size * \
        max(1, cfg.device.n_gpu)
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(
        eval_dataset) if cfg.device.local_rank == -1 else DistributedSampler(eval_dataset)
    eval_dataloader = DataLoader(
        eval_dataset, sampler=eval_sampler, batch_size=cfg.data.eval_batch_size)

    # multi-gpu evaluate
    if cfg.device.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running evaluation %s *****", prefix)
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", cfg.data.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    out_label_ids = None
    model.eval()
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        batch = tuple(t.to(device) for t in batch)

        with torch.no_grad():
            inputs = {
                "input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}
            if cfg.model.type != "distilbert":
                inputs["token_type_ids"] = (
                    batch[2] if cfg.model.type in ["bert", "xlnet"] else None
                )  # XLM and RoBERTa don"t use segment_ids
            outputs = model(**inputs)
            tmp_eval_loss, logits = outputs[:2]

            if cfg.device.n_gpu > 1:
                # mean() to average on multi-gpu parallel evaluating
                tmp_eval_loss = tmp_eval_loss.mean()

            eval_loss += tmp_eval_loss.item()
        nb_eval_steps += 1
        if preds is None:
            preds = logits.detach().cpu().numpy()
            out_label_ids = inputs["labels"].detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(
                out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)

    eval_loss = eval_loss / nb_eval_steps
    preds = np.argmax(preds, axis=2)

    label_map = {i: label for i, label in enumerate(labels)}

    out_label_list = [[] for _ in range(out_label_ids.shape[0])]
    preds_list = [[] for _ in range(out_label_ids.shape[0])]

    for i in range(out_label_ids.shape[0]):
        for j in range(out_label_ids.shape[1]):
            if out_label_ids[i, j] != pad_token_label_id:
                out_label_list[i].append(label_map[out_label_ids[i][j]])
                preds_list[i].append(label_map[preds[i][j]])

    results = {
        "loss": eval_loss,
        "precision": precision_score(out_label_list, preds_list),
        "recall": recall_score(out_label_list, preds_list),
        "f1": f1_score(out_label_list, preds_list),
    }

    logger.info("***** Eval results %s *****", prefix)
    for key in sorted(results.keys()):
        logger.info("  %s = %s", key, str(results[key]))

    return results, preds_list


def start_training(cfg: DictConfig, wandb_logger):
    """
    Start the actual training process
    """
    if (
        os.path.exists(cfg.model.output_dir)
        and os.listdir(cfg.model.output_dir)
        and cfg.experiment.do.train
        and not cfg.model.overwrite_output_dir
    ):
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                cfg.model.output_dir
            )
        )

    # Setup distant debugging if needed
    if cfg.device.server_ip and cfg.device.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd

        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(cfg.device.server_ip,
                            cfg.device.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    if cfg.device.local_rank == -1 or not cfg.device.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available()
                              and not cfg.device.no_cuda else "cpu")
        cfg.device.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(cfg.device.local_rank)
        device = torch.device("cuda", cfg.device.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        cfg.device.n_gpu = 1

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if cfg.device.local_rank in [
            -1, 0] else logging.WARN,
    )

    # Set seed
    set_seed(cfg)

    # Prepare CONLL-2003 task
    labels = get_labels(cfg.data.labels)
    num_labels = len(labels)
    # Use cross entropy ignore index as padding label id so that only real label ids contribute to the loss later
    pad_token_label_id = CrossEntropyLoss().ignore_index

    # Load pretrained model and tokenizer
    if cfg.device.local_rank not in [-1, 0]:
        # Make sure only the first process in distributed training will download model & vocab
        torch.distributed.barrier()

    cfg.model.type = cfg.model.type.lower()
    # MODEL_CLASSES[cfg.model.type]
    config_class, model_class, tokenizer_class = AutoConfig, AutoModelForTokenClassification, AutoTokenizer

    config = config_class.from_pretrained(
        cfg.model.config_name if cfg.model.config_name else cfg.model.name_or_path,
        num_labels=num_labels,
        id2label={str(i): label for i, label in enumerate(labels)},
        label2id={label: i for i, label in enumerate(labels)},
        cache_dir=cfg.model.cache_dir if cfg.model.cache_dir else None,
    )
    tokenizer = tokenizer_class.from_pretrained(
        cfg.model.tokenizer_name if cfg.model.tokenizer_name else cfg.model.name_or_path,
        # do_lower_case=cfg.model.do_lower_case,
        cache_dir=cfg.model.cache_dir if cfg.model.cache_dir else None,
        # use_fast=cfg.use_fast,
    )
    model = model_class.from_pretrained(
        cfg.model.name_or_path,
        from_tf=bool(".ckpt" in cfg.model.name_or_path),
        config=config,
        cache_dir=cfg.model.cache_dir if cfg.model.cache_dir else None,
    )

    if cfg.device.local_rank == 0:
        # Make sure only the first process in distributed training will download model & vocab
        torch.distributed.barrier()

    model.to(device)

    logger.info("Training/evaluation parameters %s", cfg)

    # Training
    if cfg.experiment.do.train:
        train_dataset = load_and_cache_examples(
            cfg, tokenizer, labels, pad_token_label_id, mode="train")
        #train_dataset = load_examples(cfg, mode="train")
        global_step, tr_loss = train(
            cfg, train_dataset, model, tokenizer, labels, pad_token_label_id, device, wandb_logger=wandb_logger)
        #global_step, tr_loss = train_ner(cfg, train_dataset, model, tokenizer, labels, pad_token_label_id)
        logger.info(" global_step = %s, average loss = %s",
                    global_step, tr_loss)
        torch.cuda.empty_cache()

    # Fine-tuning
    if cfg.experiment.do.finetune:
        tokenizer = tokenizer_class.from_pretrained(
            cfg.model.input_dir, do_lower_case=cfg.model.do_lower_case)
        model = model_class.from_pretrained(cfg.model.input_dir)
        model.to(device)
        result, predictions = evaluate(
            cfg, model, tokenizer, labels, pad_token_label_id, mode="test", device=device, wandb_logger=wandb_logger)
        train_dataset = load_and_cache_examples(
            cfg, tokenizer, labels, pad_token_label_id, mode="train", wandb_logger=wandb_logger)

        # train_dataset = load_examples(cfg, mode="train")
        global_step, tr_loss = train(
            cfg, train_dataset, model, tokenizer, labels, pad_token_label_id, device, wandb_logger=wandb_logger)
        # global_step, tr_loss = train_ner(cfg, train_dataset, model, tokenizer, labels, pad_token_label_id)
        logger.info(" global_step = %s, average loss = %s",
                    global_step, tr_loss)
        torch.cuda.empty_cache()

    # Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()
    if (cfg.experiment.do.train or cfg.experiment.do.finetune) and (cfg.device.local_rank == -1 or torch.distributed.get_rank() == 0):
        # Create output directory if needed
        if not os.path.exists(cfg.model.output_dir) and cfg.device.local_rank in [-1, 0]:
            os.makedirs(cfg.model.output_dir)

        logger.info("Saving model checkpoint to %s", cfg.model.output_dir)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        model_to_save = (
            model.module if hasattr(model, "module") else model
        )  # Take care of distributed/parallel training
        model_to_save.save_pretrained(cfg.model.output_dir)
        tokenizer.save_pretrained(cfg.model.output_dir)
        torch.cuda.empty_cache()

        # Good practice: save your training arguments together with the trained model
        torch.save(cfg, os.path.join(cfg.model.output_dir, "training_cfg.bin"))

    results = {}
    if cfg.experiment.do.eval and cfg.device.local_rank in [-1, 0]:
        tokenizer = tokenizer_class.from_pretrained(
            cfg.model.output_dir, do_lower_case=cfg.model.do_lower_case)
        checkpoints = [cfg.model.output_dir]
        if cfg.experiment.evaluate_all_checkpoints:
            checkpoints = list(
                os.path.dirname(c) for c in sorted(glob.glob(cfg.model.output_dir + "/**/" + WEIGHTS_NAME, recursive=True))
            )
            logging.getLogger("pytorch_transformers.modeling_utils").setLevel(
                logging.WARN)  # Reduce logging
        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            global_step = checkpoint.split(
                "-")[-1] if len(checkpoints) > 1 else ""
            model = model_class.from_pretrained(checkpoint)
            model.to(device)
            result, _ = evaluate(cfg, model, tokenizer, labels, pad_token_label_id,
                                 mode="dev", device=device, prefix=global_step, wandb_logger=wandb_logger)
            if global_step:
                result = {"{}_{}".format(global_step, k): v for k, v in result.items()}
            results.update(result)
        output_eval_file = os.path.join(
            cfg.model.output_dir, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            for key in sorted(results.keys()):
                writer.write("{} = {}\n".format(key, str(results[key])))
                wandb_logger.log({f"dev-{key}": str(results[key])})
        torch.cuda.empty_cache()

    if cfg.experiment.do.predict and cfg.device.local_rank in [-1, 0]:
        tokenizer = tokenizer_class.from_pretrained(
            cfg.model.output_dir, do_lower_case=cfg.model.do_lower_case)
        model = model_class.from_pretrained(cfg.model.output_dir)
        model.to(device)
        result, predictions = evaluate(
            cfg, model, tokenizer, labels, pad_token_label_id, mode="test", device=device, wandb_logger=wandb_logger)
        # Save results
        output_test_results_file = os.path.join(
            cfg.model.output_dir, "test_results.txt")
        with open(output_test_results_file, "w") as writer:
            for key in sorted(result.keys()):
                writer.write("{} = {}\n".format(key, str(result[key])))
                wandb_logger.log({f"test-{key}": str(results[key])})
        # Save predictions
        output_test_predictions_file = os.path.join(
            cfg.model.output_dir, "test_predictions.txt")
        with open(output_test_predictions_file, "w") as writer:
            with open(os.path.join(cfg.data.path, "test.txt"), "r") as f:
                example_id = 0
                for line in f:
                    if line.startswith("-DOCSTART-") or line == "" or line == "\n":
                        writer.write(line)
                        if not predictions[example_id]:
                            example_id += 1
                    elif predictions[example_id]:
                        output_line = line.split(
                        )[0] + " " + predictions[example_id].pop(0) + "\n"
                        writer.write(output_line)
                    else:
                        logger.warning(
                            "Maximum sequence length exceeded: No prediction for '%s'.", line.split()[0])
    torch.cuda.empty_cache()

    logger.info(results)


def copydir(src, dst):
    os.makedirs(dst, exist_ok=True)

    for filename in glob.glob(os.path.join(src, '*.*')):
        shutil.copyfile(filename, os.path.join(dst, filename.split('/')[-1]))


@hydra.main(version_base=None, config_path=None)
def train_eval_ner(cfg: DictConfig):
    """Trains NER models for different datasets"""

    wandb_logger = wandb.init(
        project=str(cfg.experiment.name),
        dir=get_dir(os.path.join(
            os.getcwd(),
            cfg.model.output_dir,
        )),
        group=str(cfg.experiment.group)
    )

    wandb_logger.config.update(OmegaConf.to_container(
        cfg, resolve=True,
        throw_on_missing=True
    ))

    start_training(cfg, wandb_logger)

    if cfg.experiment.do.eval:
        cfg.experiment.do.train = False

        # copy data for evaluation on eval cap
        copydir(f"{cfg.model.output_dir}",
                f"{cfg.model.output_dir}/eval-cap-{cfg.data.eval_cap}")

        start_training(cfg)


if __name__ == "__main__":
    train_eval_ner()
