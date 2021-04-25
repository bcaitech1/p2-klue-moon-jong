import logging
import numpy as np
import pandas as pd
from transformers import XLMRobertaConfig, AdamW, get_linear_schedule_with_warmup
from XLM_model import RXLMRoberta, compute_metrics
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from tqdm.auto import tqdm
from torch.cuda.amp import GradScaler

import os

import torch

logger = logging.getLogger(__name__)


scaler = GradScaler()
class Trainer(object):
    def __init__(self, num_labels, label_dict, logging_steps, save_steps, max_steps,
                 num_train_epochs, warmup_steps, adam_epsilon, learning_rate, gradient_accumulation_steps,
                 max_grad_norm, eval_batch_size, train_batch_size, model_dir, dropout_rate,
                 weight_decay, Model_name, train_dataset=None, dev_dataset=None, test_dataset=None, activate_LSTM=None):
        # self.args = args
        self.train_dataset = train_dataset
        self.eval_batch_size = eval_batch_size
        self.train_batch_size = train_batch_size
        self.dev_dataset = dev_dataset
        self.test_dataset = test_dataset
        self.Model_name = Model_name
        self.label_lst = label_dict
        self.num_labels = num_labels
        self.max_steps = max_steps
        self.weight_decay = weight_decay
        self.learning_rate = learning_rate
        self.adam_epsilon = adam_epsilon
        self.warmup_steps = warmup_steps
        self.num_train_epochs = num_train_epochs
        self.logging_steps = logging_steps
        self.save_steps = save_steps
        self.max_grad_norm = max_grad_norm
        self.model_dir = model_dir
        self.dropout_rate = dropout_rate
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.activate_LSTM = activate_LSTM
        self.config = XLMRobertaConfig.from_pretrained(
            self.Model_name,
            num_labels=self.num_labels,
            # id2label={str(i): label for i, label in enumerate(self.label_lst)},
            id2label=self.label_lst,
            # label2id={label: i for key, label in self.label_lst},
            label2id={value: key for key, value in self.label_lst.items()}
        )
        self.model = RXLMRoberta(
            self.Model_name, config=self.config, dropout_rate=self.dropout_rate, activate_LSTM=self.activate_LSTM
        )

        # GPU or CPU
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

    def train(self):
        train_sampler = RandomSampler(self.train_dataset)
        train_dataloader = DataLoader(
            self.train_dataset,
            sampler=train_sampler,
            batch_size=self.train_batch_size,
        )

        if self.max_steps > 0:
            t_total = self.max_steps
            self.num_train_epochs = (
                    self.max_steps // (len(train_dataloader) // self.gradient_accumulation_steps) + 1
            )
        else:
            t_total = len(train_dataloader) // self.gradient_accumulation_steps * self.num_train_epochs

        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=self.learning_rate,
            eps=self.adam_epsilon,
        )
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=t_total,
        )

        # Train!
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(self.train_dataset))
        logger.info("  Num Epochs = %d", self.num_train_epochs)
        logger.info("  Total train batch size = %d", self.train_batch_size)
        logger.info("  Gradient Accumulation steps = %d", self.gradient_accumulation_steps)
        logger.info("  Total optimization steps = %d", t_total)
        logger.info("  Logging steps = %d", self.logging_steps)
        logger.info("  Save steps = %d", self.save_steps)

        global_step = 0
        tr_loss = 0.0
        self.model.zero_grad()

        train_iterator = tqdm(range(int(self.num_train_epochs)), desc="Epoch")

        for _ in train_iterator:
            epoch_iterator = tqdm(train_dataloader, desc="Iteration")
            for step, batch in enumerate(epoch_iterator):
                self.model.train()
                batch = tuple(batch[t].to(self.device) for t in batch)  # GPU or CPU
                inputs = {
                    "input_ids": batch[0],
                    "attention_mask": batch[1],
                    "token_type_ids": batch[2],
                    "labels": batch[5],
                    "e1_mask": batch[3],
                    "e2_mask": batch[4],
                    # "sep_mask": batch[5]
                }
                outputs = self.model(**inputs)
                loss = outputs[0]

                if self.gradient_accumulation_steps > 1:
                    loss = loss / self.gradient_accumulation_steps

                loss.backward()

                tr_loss += loss.item()
                if (step + 1) % self.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

                    optimizer.step()
                    # scheduler.step()  # Update learning rate schedule
                    # scaler.update()
                    self.model.zero_grad()
                    global_step += 1

                    if self.logging_steps > 0 and global_step % self.logging_steps == 0:
                        logger.info("  global steps = %d", global_step)
                        self.evaluate("train")  # There is no dev set for semeval task

                    if self.save_steps > 0 and global_step % self.save_steps == 0:
                        self.save_model()

                if 0 < self.max_steps < global_step:
                    epoch_iterator.close()
                    break

            if 0 < self.max_steps < global_step:
                train_iterator.close()
                break

        return global_step, tr_loss / global_step

    def evaluate(self, mode):
        # We use test dataset because semeval doesn't have dev dataset
        if mode == "test":
            dataset = self.test_dataset
        elif mode == "dev":
            dataset = self.dev_dataset
        elif mode == "train":
            dataset = self.train_dataset
        else:
            raise Exception("Only dev and test dataset available")

        eval_sampler = SequentialSampler(dataset)
        eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=self.eval_batch_size)

        # Eval!
        logger.info("***** Running evaluation on %s dataset *****", mode)
        logger.info("  Num examples = %d", len(dataset))
        logger.info("  Batch size = %d", self.eval_batch_size)
        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None

        self.model.eval()

        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            batch = tuple(batch[t].to(self.device) for t in batch)
            with torch.no_grad():
                inputs = {
                    "input_ids": batch[0],
                    "attention_mask": batch[1],
                    "token_type_ids": batch[2],
                    "labels": batch[5],
                    "e1_mask": batch[3],
                    "e2_mask": batch[4],
                    # "sep_mask": batch[5]
                }
                outputs = self.model(**inputs)
                tmp_eval_loss, logits = outputs[:2]

                eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1

            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = inputs["labels"].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)

        eval_loss = eval_loss / nb_eval_steps
        results = {"loss": eval_loss}
        preds = np.argmax(preds, axis=1)

        result = compute_metrics(preds, out_label_ids)
        results.update(result)

        logger.info("***** Eval results *****")
        for key in sorted(results.keys()):
            logger.info("  {} = {:.4f}".format(key, results[key]))

        return results

    def test_pred(self):
        test_dataset = self.test_dataset
        test_sampler = SequentialSampler(test_dataset)
        test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=self.eval_batch_size)

        # Eval!
        logger.info("***** Running evaluation on %s dataset *****", "test")
        # logger.info("  Num examples = %d", len(dataset))
        logger.info("  Batch size = %d", self.eval_batch_size)

        nb_eval_steps = 0
        preds = None
        out_label_ids = None

        self.model.eval()

        for batch in tqdm(test_dataloader, desc="Predicting"):
            batch = tuple(batch[t].to(self.device) for t in batch)
            with torch.no_grad():
                inputs = {
                    "input_ids": batch[0],
                    "attention_mask": batch[1],
                    "token_type_ids": batch[2],
                    "labels": None,
                    "e1_mask": batch[3],
                    "e2_mask": batch[4],
                    # "sep_mask": batch[5]
                }
                outputs = self.model(**inputs)
                # print(outputs)
                pred = outputs[0]

            nb_eval_steps += 1

            if preds is None:
                preds = pred.detach().cpu().numpy()
            else:
                preds = np.append(preds, pred.detach().cpu().numpy(), axis=0)

        preds = np.argmax(preds, axis=1)
        df = pd.DataFrame(preds, columns=['pred'])
        df.to_csv('./prediction/RXLMRoberta_QA_subtract.csv', index=False)


    def save_model(self):
        # Save model checkpoint (Overwrite)
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        model_to_save = self.model.module if hasattr(self.model, "module") else self.model
        model_to_save.save_pretrained(self.model_dir)

        # Save training arguments together with the trained model
        # torch.save(self.args, os.path.join(self.args.model_dir, "training_args.bin"))
        logger.info("Saving model checkpoint to %s", self.model_dir)

    def load_model(self):
        # Check whether model exists
        if not os.path.exists(self.model_dir):
            raise Exception("Model doesn't exists! Train first!")

        self.model = RXLMRoberta.from_pretrained(self.model_dir)
        self.model.to(self.device)
        logger.info("***** Model Loaded *****")
