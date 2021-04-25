import logging
from load_data import load_data, convert_to_features, convert_to_QA
from transformers import XLMRobertaTokenizer
import pickle
from XLM_Trainer import Trainer
import torch
import random
import numpy as np


def init_logger():
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )


def main():
    random_seed = 329
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)

    init_logger()

    train_dataset = load_data("/opt/ml/input/data/train/train.tsv")
    test_dataset = load_data("/opt/ml/input/data/test/test.tsv")

    ADDITIONAL_SPECIAL_TOKENS = ["<e1>", "</e1>", "<e2>", "</e2>"]
    MODEL_NAME = "xlm-roberta-large"
    tokenizer = XLMRobertaTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.add_special_tokens({"additional_special_tokens": ADDITIONAL_SPECIAL_TOKENS})

    train_Dataset = convert_to_features(train_dataset, tokenizer, max_len=476 + 2)
    test_Dataset = convert_to_features(test_dataset, tokenizer, max_len=476 + 2)
    with open('/opt/ml/input/data/label_type.pkl', 'rb') as f:
        label_type = pickle.load(f)

    trainer = Trainer(eval_batch_size=8, train_batch_size=8, num_labels=42,
                      max_steps=-1, weight_decay=0.001, learning_rate=1e-5,
                      adam_epsilon=1e-8, warmup_steps=0, num_train_epochs=8,
                      logging_steps=1000, save_steps=550, max_grad_norm=1.0,
                      model_dir='./XLM_model_Focal', gradient_accumulation_steps=1, dropout_rate=0.3,
                      label_dict=label_type, Model_name=MODEL_NAME, train_dataset=train_Dataset,
                      test_dataset=test_Dataset, activate_LSTM=None)

    do_train = True
    do_test = True

    if do_train:
        trainer.train()

    if do_test:
        trainer.test_pred()


if __name__ == "__main__":
    main()
