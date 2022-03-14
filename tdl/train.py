from typing import Dict
import joblib
import numpy as np
from os.path import join

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

from transformers import BertTokenizerFast as BertTokenizer, BertModel, AdamW, get_linear_schedule_with_warmup


RANDOM_SEED = 42
MAX_TOKEN_COUNT = 512
N_EPOCHS = 3
BATCH_SIZE = 1
GPUS = 4
BERT_MODEL_NAME = 'bert-large-cased'

EPS = 1e-9

# TYPES = ['character', 'setting', 'feeling', 'action', 'causal relationship', 'outcome resolution', 'prediction']
TYPES = ['action', 'causal relationship', 'outcome resolution']
TYPE2CONTROL_SIGNAL = {'character': '<CHARACTER>', 'setting': '<SETTING>', 'feeling': '<FEELING>',
                       'action': '<ACTION>', 'causal relationship': '<CAUSAL_RELATIONSHIP>',
                       'outcome resolution': '<OUTCOME_RESOLUTION>', 'prediction': '<PREDICTION>'}

NUMBER_CONTROL_SIGNAL = ['<FIRST>', '<SECOND>', '<THIRD>', '<FOURTH>', '<FIFTH>', '<SIXTH>', '<SEVENTH>']

pl.seed_everything(RANDOM_SEED)

tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)

class EDL_Loss(torch.nn.Module):

    def __init__(self):
        super(EDL_Loss, self).__init__()

    def forward(self, true, pred):
        true = torch.clamp(true, min=EPS, max=1 - EPS)
        pred = torch.clamp(pred, min=EPS, max=1 - EPS)

        F.normalize(true, p=1, dim=1)
        F.normalize(pred, p=1, dim=1)

        beta = 0.7
        N = true.size()[0]

        # classification loss
        max_val, _ = torch.max(true, 1)
        max_val = max_val.view(max_val.shape[0], 1)

        label = (true == max_val).float()
        ce = -torch.sum(torch.mul(label, torch.log(pred))) / N

        # kl loss
        kl = torch.sum(torch.mul(true, torch.log(torch.div(true, pred)))) / N

        return kl + (1 - beta) * ce
    
class FairytaleQADataset(Dataset):
    def __init__(
        self,
        data: Dict,
        tokenizer: BertTokenizer,
        max_token_len: int = 128
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.max_token_len = max_token_len
        
        self.src_lines = []
        self.tgt_lines = []
        self.question_list = []
        self.inference_list = []
        self.gt_section_list = []
        self.gt_answer_list = []
        for v in self.data:
            type_counter = {}
            section = v['input']
            questions = []
            answers = []
            inferences = []
            gt_sections = []
            has_data = False
            for item in v['output']:
                print(item.keys())
                question_type = item['question_type']
                question = item['question']
                answer = item['answer']
                inference = item['inference']
                
                if question_type not in ['action', 'causal relationship', 'outcome resolution']:
                    continue
                    
                has_data = True
                
                if question_type in type_counter:
                    type_counter[question_type] += 1
                else:
                    type_counter[question_type] = 1
                
                gt_sections.append('{} {} '.format(TYPE2CONTROL_SIGNAL[question_type],
                                                 NUMBER_CONTROL_SIGNAL[type_counter[question_type]-1]) + section)
                questions.append('{} {} '.format(TYPE2CONTROL_SIGNAL[question_type],
                                                 NUMBER_CONTROL_SIGNAL[type_counter[question_type]-1]) + question)
                answers.append(answer)
                inferences.append('{} {} '.format(TYPE2CONTROL_SIGNAL[question_type],
                                                 NUMBER_CONTROL_SIGNAL[type_counter[question_type]-1]) + inference)
            if has_data==False:
                continue
               
            self.src_lines.append(section)
            tgt = [type_counter[t] if t in type_counter else 1 for t in TYPES] + [1]
            tgt_sum = np.sum(tgt)
            self.tgt_lines.append([t/tgt_sum for t in tgt])
            self.question_list.append(questions)
            self.inference_list.append(inferences)
            self.gt_section_list.append(gt_sections)
            self.gt_answer_list.append(answers)
            

    def __len__(self):
        return len(self.src_lines)

    def __getitem__(self, index: int):
        section = self.src_lines[index]
        label = self.tgt_lines[index]

        encoding = self.tokenizer.encode_plus(
            section,
            add_special_tokens=True,
            max_length=self.max_token_len,
            return_token_type_ids=False,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return dict(
            section=section,
            gt_section=self.gt_section_list[index],
            question=self.question_list[index],
            answer=self.gt_answer_list[index],
            inference=self.inference_list[index],
            input_ids=encoding["input_ids"].flatten(),
            attention_mask=encoding["attention_mask"].flatten(),
            labels=torch.FloatTensor(label)
        )


class FairytaleQADataModule(pl.LightningDataModule):
    def __init__(self, tokenizer, batch_size=8, max_token_len=128):
        super().__init__()
        self.data = joblib.load(join('data', 'data_list.pkl'))
        self.batch_size = batch_size
        self.train_data = self.data['train']
        self.val_data = self.data['val']
        self.test_data = self.data['test']
        self.tokenizer = tokenizer
        self.max_token_len = max_token_len

    def setup(self, stage=None):
        self.train_dataset = FairytaleQADataset(
            self.train_data,
            self.tokenizer,
            self.max_token_len
        )

        self.val_dataset = FairytaleQADataset(
            self.val_data,
            self.tokenizer,
            self.max_token_len
        )
        
        self.test_dataset = FairytaleQADataset(
            self.test_data,
            self.tokenizer,
            self.max_token_len
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=2
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=2
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=2
        )


data_module = FairytaleQADataModule(
    tokenizer,
    batch_size=BATCH_SIZE,
    max_token_len=MAX_TOKEN_COUNT
)


class FairytaleTDL(pl.LightningModule):
    def __init__(self, n_classes: int, n_training_steps=None, n_warmup_steps=None):
        super().__init__()
        self.bert = BertModel.from_pretrained(BERT_MODEL_NAME, return_dict=True)
        self.classifier = nn.Linear(self.bert.config.hidden_size, n_classes)
        self.n_training_steps = n_training_steps
        self.n_warmup_steps = n_warmup_steps
        self.criterion = EDL_Loss()

    def forward(self, input_ids, attention_mask, labels=None):
        output = self.bert(input_ids, attention_mask=attention_mask)
        output = self.classifier(output.pooler_output)
        # output = torch.sigmoid(output)
        output = F.softmax(output)
        loss = 0
        if labels is not None:
            loss = self.criterion(labels, output)
        return loss, output

    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        loss, outputs = self(input_ids, attention_mask, labels)
        self.log("train_loss", loss, prog_bar=True, logger=True)
        return {"loss": loss, "predictions": outputs, "labels": labels}

    def validation_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        loss, outputs = self(input_ids, attention_mask, labels)
        self.log("val_loss", loss, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        loss, outputs = self(input_ids, attention_mask, labels)
        self.log("test_loss", loss, prog_bar=True, logger=True)
        return loss

    def training_epoch_end(self, outputs):
        labels = []
        predictions = []
        for output in outputs:
            for out_labels in output["labels"].detach().cpu():
                labels.append(out_labels)
            for out_predictions in output["predictions"].detach().cpu():
                predictions.append(out_predictions)

        labels = torch.stack(labels).int()
        predictions = torch.stack(predictions)

        # for i, name in enumerate(LABEL_COLUMNS):
        #     class_roc_auc = auroc(predictions[:, i], labels[:, i])
        #     self.logger.experiment.add_scalar(
        #         f"{name}_roc_auc/Train", class_roc_auc, self.current_epoch)

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=2e-5)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.n_warmup_steps,
            num_training_steps=self.n_training_steps
        )

        return dict(
            optimizer=optimizer,
            lr_scheduler=dict(
                scheduler=scheduler,
                interval='step'
            )
        )


if __name__=='__main__':
    data = joblib.load(join('/home/amax/zzhaoao/BookComprehension/FinetuneBART/data', 'data_list.pkl'))['train']
    num_training_data = len(FairytaleQADataset(data, None, None).src_lines)

    steps_per_epoch = num_training_data // BATCH_SIZE
    total_training_steps = steps_per_epoch * N_EPOCHS
    warmup_steps = total_training_steps // 5


    model = FairytaleTDL(
        n_classes=len(TYPES) + 1,
        n_warmup_steps=warmup_steps,
        n_training_steps=total_training_steps
    )


    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints",
        # filename="best-checkpoint",
        save_top_k=-1,
        verbose=True,
        monitor="val_loss",
        mode="min"
    )


    logger = TensorBoardLogger("lightning_logs", name="fairy_tales")

    early_stopping_callback = EarlyStopping(monitor='val_loss', patience=2)

    trainer = pl.Trainer(
        logger=logger,
        callbacks=[early_stopping_callback, checkpoint_callback],
        max_epochs=N_EPOCHS,
        gpus=1,
        progress_bar_refresh_rate=30,
        # accelerator='ddp'
        )

    trainer.fit(model, data_module)


    trainer.test()
