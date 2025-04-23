import logging
import os
import random
import sys
import warnings
from dataclasses import dataclass, field
from typing import List, Optional

import torch
import torch.nn as nn
import datasets
import evaluate
import numpy as np
from datasets import load_dataset, concatenate_datasets
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import precision_score, recall_score, f1_score

import transformers
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    EvalPrediction,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import send_example_telemetry

logger = logging.getLogger(__name__)

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    do_regression: bool = field(
        default=None,
        metadata={
            "help": "Whether to do regression instead of classification. If None, will be inferred from the dataset."
        },
    )
    text_column_names: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The name of the text column in the input dataset or a CSV/JSON file. "
                'If not specified, will use the "sentence" column for single/multi-label classifcation task.'
            )
        },
    )
    text_column_delimiter: Optional[str] = field(
        default=" ", metadata={"help": "THe delimiter to use to join text columns into a single sentence."}
    )
    train_split_name: Optional[str] = field(
        default=None,
        metadata={
            "help": 'The name of the train split in the input dataset. If not specified, will use the "train" split when do_train is enabled'
        },
    )
    validation_split_name: Optional[str] = field(
        default=None,
        metadata={
            "help": 'The name of the validation split in the input dataset. If not specified, will use the "validation" split when do_eval is enabled'
        },
    )
    test_split_name: Optional[str] = field(
        default=None,
        metadata={
            "help": 'The name of the test split in the input dataset. If not specified, will use the "test" split when do_predict is enabled'
        },
    )
    remove_splits: Optional[str] = field(
        default=None,
        metadata={"help": "The splits to remove from the dataset. Multiple splits should be separated by commas."},
    )
    remove_columns: Optional[str] = field(
        default=None,
        metadata={"help": "The columns to remove from the dataset. Multiple columns should be separated by commas."},
    )
    label_column_name: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The name of the label column in the input dataset or a CSV/JSON file. "
                'If not specified, will use the "label" column for single/multi-label classifcation task'
            )
        },
    )
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": (
                "Whether to pad all samples to `max_seq_length`. "
                "If False, will pad the samples dynamically when batching to the maximum length in the batch."
            )
        },
    )
    shuffle_train_dataset: bool = field(
        default=False, metadata={"help": "Whether to shuffle the train dataset or not."}
    )
    shuffle_seed: int = field(
        default=42, metadata={"help": "Random seed that will be used to shuffle the train dataset."}
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of prediction examples to this "
                "value if set."
            )
        },
    )
    metric_name: Optional[str] = field(default=None, metadata={"help": "The metric to use for evaluation."})
    train_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the training data."}
    )
    validation_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the validation data."}
    )
    test_file: Optional[str] = field(default=None, metadata={"help": "A csv or a json file containing the test data."})
    do_augment: bool = field(default=False, metadata={"help": "Whether to augment with provided synthetic dataset."})
    do_weighted_loss: bool = field(default=False, metadata={"help": "Whether to use weighted cross-entropy loss."})
    weight_multiplier: float = field(default=1.0, metadata={"help": "Weighted loss multiplier factor."})

    def __post_init__(self):
        if self.dataset_name is None:
            if self.train_file is None or self.validation_file is None:
                raise ValueError(" training/validation file or a dataset name.")

            train_extension = self.train_file.split(".")[-1]
            assert train_extension in ["csv", "json"], "`train_file` should be a csv or a json file."
            validation_extension = self.validation_file.split(".")[-1]
            assert (
                validation_extension == train_extension
            ), "`validation_file` should have the same extension (csv or json) as `train_file`."

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    token: str = field(
        default=None,
        metadata={
            "help": (
                "The token to use as HTTP bearer authorization for remote files. If not specified, will use the token "
                "generated when running `huggingface-cli login` (stored in `~/.huggingface`)."
            )
        },
    )
    use_auth_token: bool = field(
        default=None,
        metadata={
            "help": "The `use_auth_token` argument is deprecated and will be removed in v4.34. Please use `token` instead."
        },
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether or not to allow for custom models defined on the Hub in their own modeling files. This option"
                "should only be set to `True` for repositories you trust and in which you have read the code, as it will "
                "execute code present on the Hub on your local machine."
            )
        },
    )
    ignore_mismatched_sizes: bool = field(
        default=False,
        metadata={"help": "Will enable to load a pretrained model whose head dimensions are different."},
    )

def get_label_list(raw_dataset, split="train") -> List[str]:
    """Get the list of labels from a mutli-label dataset"""

    if isinstance(raw_dataset[split]["label"][0], list):
        label_list = [label for sample in raw_dataset[split]["label"] for label in sample]
        label_list = list(set(label_list))
    else:
        label_list = raw_dataset[split].unique("label")
    # we will treat the label list as a list of string instead of int, consistent with model.config.label2id
    label_list = [str(label) for label in label_list]
    return label_list

"""# Model and Dataset

## Model

Terdapat 2 model yang akan digunakan:
1. IndoNLU IndoBERT
2. XLM-RoBERTa

## Dataset Asli dan Dataset Sintetis

Dataset yang digunakan dalam pelatihan model ini terbagi menjadi dua jenis: dataset asli (original) dan dataset sintetis (synthetic).

### **Dataset Asli:**
1. **Twitter-Indonesian-Sarcastic-Fix-Text** (Dataset asli Twitter)
2. **Reddit-Indonesian-Sarcastic-Fix-Text** (Dataset asli Reddit)

### **Dataset Sintetis:**
Dataset sintetis akan digunakan untuk augmentasi data.

1. **Twitter-Indonesian-Sarcastic-Synthetic-Zero-Shot-Topic**
2. **Twitter-Indonesian-Sarcastic-Synthetic-One-Shot**
3. **Twitter-Indonesian-Sarcastic-Synthetic-Few-Shot**
4. **Reddit-Indonesian-Sarcastic-Synthetic-Zero-Shot-Topic**
5. **Reddit-Indonesian-Sarcastic-Synthetic-One-Shot**
6. **Reddit-Indonesian-Sarcastic-Synthetic-Few-Shot**

## Variabel `do_augment_var`

Variabel **`do_augment_var`** adalah sebuah flag (penanda) yang digunakan untuk menentukan apakah **data augmentasi** akan dilakukan pada dataset yang digunakan untuk pelatihan model.

- **Jika `do_augment_var = True`**: Data augmentasi akan dilakukan. Dataset sintetis akan digabungkan dengan dataset asli.
- **Jika `do_augment_var = False`**: Data augmentasi tidak dilakukan. Hanya dataset asli yang digunakan untuk pelatihan model.

## Variabel `ratios`

Variabel `ratios` merujuk pada **persentase data sintetis** yang akan digunakan dalam proses augmentasi data.

"""

# List of model names (1: IndoNLU IndoBERT, 2: XLM-RoBERTa)
list_of_model_name = [
    "indobenchmark/indobert-base-p1",   # 1
    "xlm-roberta-large"                 # 2
]

# List of original datasets (Mapped by numbers for easier reference)
list_of_ori_dataset = [
    "enoubi/Twitter-Indonesian-Sarcastic-Fix-Text",                     # 1
    "enoubi/Reddit-Indonesian-Sarcastic-Fix-Text"                       # 2
]

# List of synthetic datasets (Mapped by numbers for easier reference)
list_of_syn_dataset = [
    "enoubi/Twitter-Indonesian-Sarcastic-Synthetic-Zero-Shot-Topic",    # 1
    "enoubi/Twitter-Indonesian-Sarcastic-Synthetic-One-Shot",           # 2
    "enoubi/Twitter-Indonesian-Sarcastic-Synthetic-Few-Shot",           # 3
    "enoubi/Reddit-Indonesian-Sarcastic-Synthetic-Zero-Shot-Topic",     # 4
    "enoubi/Reddit-Indonesian-Sarcastic-Synthetic-One-Shot",            # 5
    "enoubi/Reddit-Indonesian-Sarcastic-Synthetic-Few-Shot"             # 6
]

# Model number selection: (1 -> IndoNLU IndoBERT, 2 -> XLM-RoBERTa)
model_number = 1

# Flag to enable data augmentation (set to True for augmentation)
do_augment_var = True

# List of augmentation percentages (e.g., 10%, 20%, 30%, ...)
ratios = [10, 20, 30, 40, 50]

# List of original datasets being used (can be extended as needed)
list_ori_data_number = [1]

# Mapping between original datasets and corresponding synthetic datasets
ori_to_syn_mapping = {
    1: [1, 2, 3],  # Twitter dataset mapping to synthetic datasets (Zero-Shot Topic, One-Shot, Few-Shot)
    2: [4, 5, 6]   # Reddit dataset mapping to synthetic datasets (Zero-Shot Topic, One-Shot, Few-Shot)
}

"""# Train and Test"""

for ori_data_number in list_ori_data_number:
    list_syn_data_number = ori_to_syn_mapping[ori_data_number]

    for syn_data_number in list_syn_data_number:
        for ratio in ratios:
            model_args = ModelArguments(model_name_or_path=list_of_model_name[model_number - 1])

            if do_augment_var:
                syn_dataset = list_of_syn_dataset[syn_data_number - 1]

            ori_dataset_name = list_of_ori_dataset[ori_data_number - 1]

            data_args = DataTrainingArguments(
                dataset_name=ori_dataset_name,
                dataset_config_name="default",
                text_column_names="tweet" if "Twitter" in ori_dataset_name else "text",
                label_column_name="label",
                metric_name="f1",
                max_seq_length=128,
                shuffle_train_dataset=True,
                do_augment=do_augment_var
            )
            training_args = TrainingArguments(
                output_dir="./outputs",
                eval_strategy="epoch",
                logging_strategy="epoch",
                save_strategy="epoch",
                save_total_limit=2,
                lr_scheduler_type="cosine",
                learning_rate=1e-5,
                per_device_train_batch_size=32,
                per_device_eval_batch_size=64,
                label_smoothing_factor=0.0,
                num_train_epochs=100,
                do_train=True,
                do_eval=True,
                do_predict=True,
                weight_decay=0.03,
                logging_dir="./logs",
                load_best_model_at_end=True,
                metric_for_best_model="f1",
                seed=42,
                report_to="none",
                fp16=True,
                overwrite_output_dir=True,
            )

            if model_args.use_auth_token is not None:
                warnings.warn(
                    "The `use_auth_token` argument is deprecated and will be removed in v4.34. Please use `token` instead.",
                    FutureWarning,
                )
                if model_args.token is not None:
                    raise ValueError("`token` and `use_auth_token` are both specified. Please set only the argument `token`.")
                model_args.token = model_args.use_auth_token

            # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
            # information sent is the one passed as arguments along with your Python/PyTorch versions.
            send_example_telemetry("run_classification", model_args, data_args)

            # Setup logging
            logging.basicConfig(
                format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
                datefmt="%m/%d/%Y %H:%M:%S",
                handlers=[logging.StreamHandler(sys.stdout)],
            )

            if training_args.should_log:
                # The default of training_args.log_level is passive, so we set log level at info here to have that default.
                transformers.utils.logging.set_verbosity_info()

            log_level = training_args.get_process_log_level()
            logger.setLevel(log_level)
            datasets.utils.logging.set_verbosity(log_level)
            transformers.utils.logging.set_verbosity(log_level)
            transformers.utils.logging.enable_default_handler()
            transformers.utils.logging.enable_explicit_format()

            # Log on each process the small summary:
            logger.warning(
                f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
                + f"distributed training: {training_args.parallel_mode.value == 'distributed'}, 16-bits training: {training_args.fp16}"
            )
            logger.info(f"Training/evaluation parameters {training_args}")

            # Detecting last checkpoint.
            last_checkpoint = None
            if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
                last_checkpoint = get_last_checkpoint(training_args.output_dir)
                if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
                    raise ValueError(
                        f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                        "Use --overwrite_output_dir to overcome."
                    )
                elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
                    logger.info(
                        f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                        "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
                    )

            # Set seed before initializing model.
            set_seed(training_args.seed)

            # Get the datasets: you can either provide your own CSV/JSON training and evaluation files, or specify a dataset name
            # to load from huggingface/datasets. In ether case, you can specify a the key of the column(s) containing the text and
            # the key of the column containing the label. If multiple columns are specified for the text, they will be joined togather
            # for the actual text value.
            # In distributed training, the load_dataset function guarantee that only one local process can concurrently
            # download the dataset.
            if data_args.dataset_name is not None:
                # Downloading and loading a dataset from the hub.
                raw_datasets = load_dataset(
                    data_args.dataset_name,
                    data_args.dataset_config_name,
                    cache_dir=model_args.cache_dir,
                    token=model_args.token,
                )
                # Try print some info about the dataset
                logger.info(f"Dataset loaded: {raw_datasets}")
                logger.info(raw_datasets)
            else:
                # Loading a dataset from your local files.
                # CSV/JSON training and evaluation files are needed.
                data_files = {"train": data_args.train_file, "validation": data_args.validation_file}

                # Get the test dataset: you can provide your own CSV/JSON test file
                if training_args.do_predict:
                    if data_args.test_file is not None:
                        train_extension = data_args.train_file.split(".")[-1]
                        test_extension = data_args.test_file.split(".")[-1]
                        assert (
                            test_extension == train_extension
                        ), "`test_file` should have the same extension (csv or json) as `train_file`."
                        data_files["test"] = data_args.test_file
                    else:
                        raise ValueError("Need either a dataset name or a test file for `do_predict`.")

                for key in data_files.keys():
                    logger.info(f"load a local file for {key}: {data_files[key]}")

                if data_args.train_file.endswith(".csv"):
                    # Loading a dataset from local csv files
                    raw_datasets = load_dataset(
                        "csv",
                        data_files=data_files,
                        cache_dir=model_args.cache_dir,
                        token=model_args.token,
                    )
                else:
                    # Loading a dataset from local json files
                    raw_datasets = load_dataset(
                        "json",
                        data_files=data_files,
                        cache_dir=model_args.cache_dir,
                        token=model_args.token,
                    )

            if data_args.do_augment:
                print("== AUGMENTING DATA ==")
                print(f"Data augmentation (percentage): {ratio}%")

                # Load the synthetic dataset
                synthetic = load_dataset(syn_dataset)

                # Calculate the total number of sarcasm samples in the training set
                total_sarcasm = sum(1 for x in raw_datasets["train"] if x["label"] == 1)

                # Determine the number of synthetic samples to add based on the augmentation ratio
                num_samples = int((ratio / 100) * total_sarcasm)

                # Ensure that the number of samples doesn't exceed the available synthetic dataset size
                num_samples = min(num_samples, len(synthetic["train"]))

                print(f"Total original sarcasm data: {total_sarcasm}")
                print(f"Total synthetic data added: {num_samples}")

                # Select the first 'num_samples' data from the synthetic dataset
                synthetic["train"] = synthetic["train"].select(range(num_samples))

                # normalize columns
                if data_args.text_column_names is not None:
                    text_column_name = data_args.text_column_names.split(",")[0]

                # Remove any columns from the raw dataset that are not 'text/tweet' or 'label'
                raw_datasets = raw_datasets.remove_columns(
                    set(raw_datasets["train"].features.keys()) - set([text_column_name, "label"])
                )

                print(f"Total training data before augmentation: {len(raw_datasets['train'])}")

                # Concatenate the original training dataset with the augmented synthetic dataset
                raw_datasets["train"] = concatenate_datasets([raw_datasets["train"], synthetic["train"]])

                print(f"Total training data after augmentation: {len(raw_datasets['train'])}")

            # See more about loading any type of standard or custom dataset at
            # https://huggingface.co/docs/datasets/loading_datasets.

            if data_args.remove_splits is not None:
                for split in data_args.remove_splits.split(","):
                    logger.info(f"removing split {split}")
                    raw_datasets.pop(split)

            if data_args.train_split_name is not None:
                logger.info(f"using {data_args.validation_split_name} as validation set")
                raw_datasets["train"] = raw_datasets[data_args.train_split_name]
                raw_datasets.pop(data_args.train_split_name)

            if data_args.validation_split_name is not None:
                logger.info(f"using {data_args.validation_split_name} as validation set")
                raw_datasets["validation"] = raw_datasets[data_args.validation_split_name]
                raw_datasets.pop(data_args.validation_split_name)

            if data_args.test_split_name is not None:
                logger.info(f"using {data_args.test_split_name} as test set")
                raw_datasets["test"] = raw_datasets[data_args.test_split_name]
                raw_datasets.pop(data_args.test_split_name)

            if data_args.remove_columns is not None:
                for split in raw_datasets.keys():
                    for column in data_args.remove_columns.split(","):
                        logger.info(f"removing column {column} from split {split}")
                        raw_datasets[split].remove_columns(column)

            if data_args.label_column_name is not None and data_args.label_column_name != "label":
                for key in raw_datasets.keys():
                    raw_datasets[key] = raw_datasets[key].rename_column(data_args.label_column_name, "label")

            # Trying to have good defaults here, don't hesitate to tweak to your needs.

            # We have to deal with common cases that labels appear in the training set but not in the validation/test set.
            # So we build the label list from the union of labels in train/val/test.
            label_list = get_label_list(raw_datasets, split="train")
            for split in ["validation", "test"]:
                if split in raw_datasets:
                    val_or_test_labels = get_label_list(raw_datasets, split=split)
                    diff = set(val_or_test_labels).difference(set(label_list))
                    if len(diff) > 0:
                        # add the labels that appear in val/test but not in train, throw a warning
                        logger.warning(
                            f"Labels {diff} in {split} set but not in training set, adding them to the label list"
                        )
                        label_list += list(diff)
            # if label is -1, we throw a warning and remove it from the label list
            for label in label_list:
                if label == -1:
                    logger.warning("Label -1 found in label list, removing it.")
                    label_list.remove(label)

            label_list.sort()
            num_labels = len(label_list)
            if num_labels <= 1:
                raise ValueError("You need more than one label to do classification.")

            # Load pretrained model and tokenizer
            # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
            # download model & vocab.
            config = AutoConfig.from_pretrained(
                model_args.config_name if model_args.config_name else model_args.model_name_or_path,
                num_labels=num_labels,
                finetuning_task="text-classification",
                cache_dir=model_args.cache_dir,
                revision=model_args.model_revision,
                token=model_args.token,
                trust_remote_code=model_args.trust_remote_code,
            )

            config.problem_type = "single_label_classification"
            logger.info("setting problem type to single label classification")

            tokenizer = AutoTokenizer.from_pretrained(
                model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
                cache_dir=model_args.cache_dir,
                use_fast=model_args.use_fast_tokenizer,
                revision=model_args.model_revision,
                token=model_args.token,
                trust_remote_code=model_args.trust_remote_code,
            )
            model = AutoModelForSequenceClassification.from_pretrained(
                model_args.model_name_or_path,
                from_tf=bool(".ckpt" in model_args.model_name_or_path),
                config=config,
                cache_dir=model_args.cache_dir,
                revision=model_args.model_revision,
                token=model_args.token,
                trust_remote_code=model_args.trust_remote_code,
                ignore_mismatched_sizes=model_args.ignore_mismatched_sizes,
            )

            # Padding strategy
            if data_args.pad_to_max_length:
                padding = "max_length"
            else:
                # We will pad later, dynamically at batch creation, to the max sequence length in each batch
                padding = False

            # for training ,we will update the config with label infos,
            # if do_train is not set, we will use the label infos in the config
            if training_args.do_train:  # classification, training
                label_to_id = {v: i for i, v in enumerate(label_list)}
                # update config with label infos
                if model.config.label2id != label_to_id:
                    logger.warning(
                        "The label2id key in the model config.json is not equal to the label2id key of this "
                        "run. You can ignore this if you are doing finetuning."
                    )
                model.config.label2id = label_to_id
                model.config.id2label = {id: label for label, id in config.label2id.items()}
            else:  # classification, but not training
                logger.info("using label infos in the model config")
                logger.info("label2id: {}".format(model.config.label2id))
                label_to_id = model.config.label2id

            if data_args.max_seq_length > tokenizer.model_max_length:
                logger.warning(
                    f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the "
                    f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
                )
            max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

            def preprocess_function(examples):
                if data_args.text_column_names is not None:
                    text_column_names = data_args.text_column_names.split(",")
                    # join together text columns into "sentence" column
                    examples["sentence"] = examples[text_column_names[0]]
                    for column in text_column_names[1:]:
                        for i in range(len(examples[column])):
                            if examples["sentence"][i] is None:
                                examples["sentence"][i] = ""
                            if examples[column][i] is None:
                                examples[column][i] = ""
                            examples["sentence"][i] += data_args.text_column_delimiter + examples[column][i]

                # Pastikan format yang benar sebelum masuk tokenizer
                if not isinstance(examples["sentence"], list) or not all(isinstance(x, str) for x in examples["sentence"]):
                    raise TypeError(f"Expected List[str], but got {type(examples['sentence'])}: {examples['sentence']}")

                # Tokenize the texts
                result = tokenizer(examples["sentence"], padding=padding, max_length=max_seq_length, truncation=True)
                if label_to_id is not None and "label" in examples:
                    result["label"] = [(label_to_id[str(l)] if l != -1 else -1) for l in examples["label"]]
                return result

            # Running the preprocessing pipeline on all the datasets
            with training_args.main_process_first(desc="dataset map pre-processing"):
                raw_datasets = raw_datasets.map(
                    preprocess_function,
                    batched=True,
                    load_from_cache_file=not data_args.overwrite_cache,
                    desc="Running tokenizer on dataset",
                )

            if training_args.do_train:
                if "train" not in raw_datasets:
                    raise ValueError("--do_train requires a train dataset.")
                train_dataset = raw_datasets["train"]
                if data_args.shuffle_train_dataset:
                    logger.info("Shuffling the training dataset")
                    train_dataset = train_dataset.shuffle(seed=data_args.shuffle_seed)
                if data_args.max_train_samples is not None:
                    max_train_samples = min(len(train_dataset), data_args.max_train_samples)
                    train_dataset = train_dataset.select(range(max_train_samples))

            if training_args.do_eval:
                if "validation" not in raw_datasets and "validation_matched" not in raw_datasets:
                    if "test" not in raw_datasets and "test_matched" not in raw_datasets:
                        raise ValueError("--do_eval requires a validation or test dataset if validation is not defined.")
                    else:
                        logger.warning("Validation dataset not found. Falling back to test dataset for validation.")
                        print("Validation dataset not found. Falling back to test dataset for validation.")
                        eval_dataset = raw_datasets["test"]
                else:
                    print("Eval Dataset using Validation dataset.")
                    eval_dataset = raw_datasets["validation"]

                if data_args.max_eval_samples is not None:
                    max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
                    eval_dataset = eval_dataset.select(range(max_eval_samples))

            if training_args.do_predict or data_args.test_file is not None:
                if "test" not in raw_datasets:
                    raise ValueError("--do_predict requires a test dataset")
                predict_dataset = raw_datasets["test"]
                # remove label column if it exists
                if data_args.max_predict_samples is not None:
                    max_predict_samples = min(len(predict_dataset), data_args.max_predict_samples)
                    predict_dataset = predict_dataset.select(range(max_predict_samples))

            # Log a few random samples from the training set:
            if training_args.do_train:
                for index in random.sample(range(len(train_dataset)), 3):
                    logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

            accuracy = evaluate.load("accuracy")
            f1 = evaluate.load("f1")
            precision = evaluate.load("precision")
            recall = evaluate.load("recall")

            def compute_metrics(p: EvalPrediction):
                preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
                preds = np.argmax(preds, axis=1)
                return {
                    "accuracy": accuracy.compute(predictions=preds, references=p.label_ids)["accuracy"],
                    "f1": f1_score(p.label_ids, preds),
                    "precision": precision_score(p.label_ids, preds),
                    "recall": recall_score(p.label_ids, preds)
                }

            # Data collator will default to DataCollatorWithPadding when the tokenizer is passed to Trainer, so we change it if
            # we already did the padding.
            if data_args.pad_to_max_length:
                data_collator = default_data_collator
            elif training_args.fp16:
                data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
            else:
                data_collator = None

            early_stopping = EarlyStoppingCallback(early_stopping_patience=3, early_stopping_threshold=0.01)

            # get all label classes
            classes = sorted([int(l) for l in label_list])
            weights = (
                compute_class_weight(class_weight="balanced", classes=np.array(classes), y=train_dataset["label"])
                * data_args.weight_multiplier
            )

            class WeightedTrainer(Trainer):
                def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
                    labels = inputs.pop("labels")
                    outputs = model(**inputs)
                    logits = outputs.get("logits")
                    loss_fct = nn.CrossEntropyLoss(
                        weight=torch.tensor(weights, device=model.device, dtype=torch.float)
                        if data_args.do_weighted_loss
                        else None
                    )
                    loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
                    return (loss, outputs) if return_outputs else loss

            # Initialize our Trainer
            trainer = WeightedTrainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset if training_args.do_train else None,
                eval_dataset=eval_dataset if training_args.do_eval else None,
                compute_metrics=compute_metrics,
                tokenizer=tokenizer,
                data_collator=data_collator,
                callbacks=[early_stopping],
            )

            # Training
            if training_args.do_train:
                checkpoint = None
                if training_args.resume_from_checkpoint is not None:
                    checkpoint = training_args.resume_from_checkpoint
                elif last_checkpoint is not None:
                    checkpoint = last_checkpoint
                train_result = trainer.train(resume_from_checkpoint=checkpoint)
                metrics = train_result.metrics
                max_train_samples = (
                    data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
                )
                metrics["train_samples"] = min(max_train_samples, len(train_dataset))
                trainer.save_model()  # Saves the tokenizer too for easy upload
                trainer.log_metrics("train", metrics)
                trainer.save_metrics("train", metrics)
                trainer.save_state()

            # Evaluation
            if training_args.do_eval:
                logger.info("*** Evaluate ***")
                metrics = trainer.evaluate(eval_dataset=predict_dataset)
                # max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
                # metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))
                trainer.log_metrics("eval", metrics)
                trainer.save_metrics("eval", metrics)

            if training_args.do_predict:
                logger.info("*** Predict ***")
                # Removing the `label` columns if exists because it might contains -1 and Trainer won't like that.
                if "label" in predict_dataset.features:
                    predict_dataset = predict_dataset.remove_columns("label")
                predictions = trainer.predict(predict_dataset, metric_key_prefix="predict").predictions
                predictions = np.argmax(predictions, axis=1)
                output_predict_file = os.path.join(training_args.output_dir, "predict_results.txt")
                if trainer.is_world_process_zero():
                    with open(output_predict_file, "w") as writer:
                        logger.info("***** Predict results *****")
                        writer.write("index\tprediction\n")
                        for index, item in enumerate(predictions):
                            item = label_list[item]
                            writer.write(f"{index}\t{item}\n")
                logger.info("Predict results saved at {}".format(output_predict_file))
            kwargs = {"finetuned_from": model_args.model_name_or_path, "tasks": "text-classification"}

            if training_args.push_to_hub:
                trainer.push_to_hub(**kwargs)
            else:
                trainer.create_model_card(**kwargs)

            # Clean up model name by replacing '/' with '_'
            file_model_name = list_of_model_name[model_number - 1].replace('/', '_')
            print(f"Model used: {file_model_name}")
            print(f"Original dataset used: {ori_dataset_name}")

            # Print synthetic dataset details only if augmentation is enabled
            if do_augment_var:
                print(f"Synthetic dataset used: {syn_dataset}")
                print(f"Percentage of synthetic data used: {ratio}%")

