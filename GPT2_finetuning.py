import os
import re
import sys
import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import pytorch_lightning as pl
from pytorch_lightning import loggers
import pytorch_lightning.callbacks as plc
from pytorch_lightning.callbacks import TQDMProgressBar, RichProgressBar
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW, get_linear_schedule_with_warmup
# 重写进度条，解决dynamic_ncols=True的问题
from pytorch_lightning.callbacks.progress.tqdm_progress import Tqdm


# 设置随机数种子
def set_seed(args: argparse.Namespace):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    # if args.devices > 0:
    #     torch.cuda.manual_seed_all(args.seed)


def data_preprocess():
    train = pd.read_csv('./data/train.csv').dropna()
    val = pd.read_csv('./data/valid.csv')
    test = pd.read_csv('./data/test.csv')

    # 证明训练集和测试集没有重复
    # print(set(test.textID.values).intersection(train.textID.values))

    train['processed_text'] = "question:" + train.sentiment + ' context:' + train.text + " answer:" + train.selected_text
    val['processed_text'] = "question:" + val.sentiment + ' context:' + val.text + " answer:" + val.selected_text
    test['processed_text'] = "question:" + test.sentiment + ' context:' + test.text + " answer:" + test.selected_text

    # val 要多一个没有标签的特征，用于 validation 时 generate，如果用有标签的 input_ids 会直接 generate 原答案
    val['unlabeled_text'] = "question:" + val.sentiment + ' context:' + val.text + " answer:"
    test['unlabeled_text'] = "question:" + test.sentiment + ' context:' + test.text + " answer:"

    print(train.shape, val.shape, test.shape)
    # print(train.columns, test.columns)

    return train, val, test


class GPT2Dataset(Dataset):
    def __init__(self, tokenizer, dataframe, max_length: int, train_type: str):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.train_type = train_type

    def __len__(self):
        return len(self.data['processed_text'])

    def __getitem__(self, item):
        text = self.data['processed_text'][item]
        selected_text = []  # 用于 validation 计算 jaccard
        unlabeled_input_ids = []  # 用于 validation 时 generate
        unlabeled_attention_mask = []  # 用于 validation 时 generate

        # if self.train_type == 'test':
        #     self.tokenizer.padding_side = 'left'  # generate 的时候需要 padding 在左边，否则结果可能全为eos
        #     encodings_dict = self.tokenizer('<|startoftext|>' + text, max_length=self.max_length, padding='max_length',
        #                                     truncation=True, return_tensors='pt')
        #
        # else:
        #     self.tokenizer.padding_side = 'right'
        #     encodings_dict = self.tokenizer('<|startoftext|>' + text, max_length=self.max_length, padding='max_length',
        #                                     truncation=True, return_tensors='pt')
        #
        #     if self.train_type == 'val':
        #         unlabeled_text = self.data['unlabeled_text'][item]
        #         self.tokenizer.padding_side = 'left'
        #         unlabeled_encodings_dict = self.tokenizer('<|startoftext|>' + unlabeled_text,
        #                                                   max_length=self.max_length, padding='max_length',
        #                                                   truncation=True, return_tensors='pt')
        #         selected_text = self.data['selected_text'][item]
        #         unlabeled_input_ids = unlabeled_encodings_dict['input_ids'].squeeze()
        #         unlabeled_attention_mask = unlabeled_encodings_dict['attention_mask'].squeeze()
        self.tokenizer.padding_side = 'right'
        encodings_dict = self.tokenizer('<|startoftext|>' + text, max_length=self.max_length, padding='max_length',
                                        truncation=True, return_tensors='pt')

        if self.train_type != 'train':
            unlabeled_text = self.data['unlabeled_text'][item]
            self.tokenizer.padding_side = 'left'
            unlabeled_encodings_dict = self.tokenizer('<|startoftext|>' + unlabeled_text,
                                                      max_length=self.max_length, padding='max_length',
                                                      truncation=True, return_tensors='pt')
            selected_text = self.data['selected_text'][item]
            unlabeled_input_ids = unlabeled_encodings_dict['input_ids'].squeeze()
            unlabeled_attention_mask = unlabeled_encodings_dict['attention_mask'].squeeze()

        input_ids = encodings_dict['input_ids'].squeeze()
        attention_mask = encodings_dict['attention_mask'].squeeze()

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'selected_text': selected_text,
            'unlabeled_input_ids': unlabeled_input_ids,
            'unlabeled_attention_mask': unlabeled_attention_mask,
        }


class GPT2Model(pl.LightningModule):
    def __init__(self, hparams):
        super(GPT2Model, self).__init__()

        # self.hparams = hparams 不能用了
        self.save_hyperparameters(hparams)

        self.model = GPT2LMHeadModel.from_pretrained(self.hparams.model_name_or_path)
        self.tokenizer = GPT2Tokenizer.from_pretrained(self.hparams.tokenizer_name_or_path,
                                                       bos_token='<|startoftext|>',
                                                       pad_token='<|endoftext|>')
        # print(self.tokenizer.special_tokens_map)

        # vocab中添加了新的token，vocab大小和embedding大小不匹配。需要调整模型的embedding的大小
        # print(self.model.transformer.wte.weight.shape[0] == len(self.tokenizer))  # 50257 50258
        self.model.resize_token_embeddings(len(self.tokenizer))

    def forward(self,
                input_ids,
                attention_mask=None,
                labels=None
                ):
        return self.model(
            input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

    # 计算 batch 的损失，用于 training_step 和 validation_step
    def _step(self, batch):

        outputs = self.model(input_ids=batch["input_ids"],
                       attention_mask=batch["attention_mask"],
                       labels=batch["input_ids"],
                       )
        loss = outputs[0]
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._step(batch)
        # 进度条默认记录的损失在几个值的窗口中平滑，而自己记录的是确切值。 https://github.com/Lightning-AI/lightning/issues/15831
        self.log("train_loss", loss, prog_bar=True, logger=True)

        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        loss = self._step(batch)

        generated_ids = self.model.generate(
            input_ids=batch["unlabeled_input_ids"],
            attention_mask=batch["unlabeled_attention_mask"],
            max_new_tokens=100,
            num_return_sequences=1,
            do_sample=True,
            top_k=50,
            temperature=0.7,
            top_p=0.95,
        )
        generated_text = [self.tokenizer.decode(g, skip_special_tokens=True) for g in generated_ids]
        predict_text = self.intercept_answer(generated_text=generated_text)

        jaccard_score = [jaccard(p, t) for p, t in zip(predict_text, batch['selected_text'])]
        jaccard_score = sum(jaccard_score) / len(jaccard_score)

        # 验证的损失不能按 step 输出， https://github.com/Lightning-AI/lightning/issues/15165
        self.log("val_loss", loss, logger=True, batch_size=self.hparams.train_batch_size)
        self.log("val_jaccard_score", jaccard_score, logger=True, batch_size=self.hparams.train_batch_size)

        return {"val_loss": loss, "val_jaccard_score": jaccard_score}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        jaccard_scores = np.array([x["val_jaccard_score"] for x in outputs])
        avg_jaccard_score = jaccard_scores.mean()

        return {"avg_val_loss": avg_loss, "avg_val_jaccard_score": avg_jaccard_score}

    def test_step(self, batch, batch_idx):
        loss = self._step(batch)

        generated_ids = self.model.generate(
            input_ids=batch["unlabeled_input_ids"],
            attention_mask=batch["unlabeled_attention_mask"],
            max_new_tokens=100,
            num_return_sequences=1,
            do_sample=True,
            top_k=50,
            temperature=0.7,
            top_p=0.95,
        )
        generated_text = [self.tokenizer.decode(g, skip_special_tokens=True) for g in generated_ids]
        predict_text = self.intercept_answer(generated_text=generated_text)

        jaccard_score = [jaccard(p, t) for p, t in zip(predict_text, batch['selected_text'])]
        jaccard_score = sum(jaccard_score) / len(jaccard_score)

        # 验证的损失不能按 step 输出， https://github.com/Lightning-AI/lightning/issues/15165
        self.log("test_loss", loss, logger=True, batch_size=self.hparams.train_batch_size)
        self.log("test_jaccard_score", jaccard_score, logger=True, batch_size=self.hparams.train_batch_size)

        return {"preds": predict_text, "test_loss": loss, "test_jaccard_score": jaccard_score}

    def test_epoch_end(self, outputs):
        output_test_predictions_file = os.path.join(self.hparams.output_dir, "GPT2_predictions.txt")

        with open(output_test_predictions_file, "w+") as p_writer:
            for output_batch in outputs[0]["preds"]:
                p_writer.writelines(output_batch + "\n")
            p_writer.close()

        avg_loss = torch.stack([x["test_loss"] for x in outputs]).mean()
        jaccard_scores = np.array([x["test_jaccard_score"] for x in outputs])
        avg_jaccard_score = jaccard_scores.mean()

        return {"avg_test_loss": avg_loss, "avg_test_jaccard_score": avg_jaccard_score}

    def configure_optimizers(self):
        model = self.model
        # When training neural networks, it is common to use "weight decay," where after each update,
        no_decay = ["bias", "LayerNorm.weight"]
        # Group parameters to those that will and will not have weight decay applied
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]

        optimizer = AdamW(optimizer_grouped_parameters, lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon)

        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=self.hparams.warmup_steps,
            num_training_steps=self.trainer.estimated_stepping_batches
        )
        self.lr_scheduler = scheduler

        return [optimizer], [scheduler]

    # Adjust weights based on calculated gradients and learning rate scheduler, and refresh gradients
    def optimizer_step(self,
                       epoch=None,
                       batch_idx=None,
                       optimizer=None,
                       optimizer_idx=None,
                       optimizer_closure=None,
                       on_tpu=None,
                       using_lbfgs=None
                       ):
        # Adjust weights based on calculated gradients
        optimizer.step(closure=optimizer_closure)
        # Refresh gradients (to zero)
        optimizer.zero_grad()
        # Update the learning rate scheduler
        self.lr_scheduler.step()

    def get_dataloader(self, train_type: str, batch_size: int, shuffle: bool = False) -> DataLoader:
        train, val, test = data_preprocess()

        if train_type == 'train':
            data = train
        elif train_type == 'val':
            data = val
        else:
            data = test

        dataset = GPT2Dataset(self.tokenizer, data, self.hparams.max_length, train_type)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=4)

        return dataloader

    def train_dataloader(self) -> DataLoader:
        dataloader = self.get_dataloader('train', batch_size=self.hparams.train_batch_size, shuffle=True)
        return dataloader

    def val_dataloader(self) -> DataLoader:
        return self.get_dataloader('val', batch_size=self.hparams.eval_batch_size)

    def test_dataloader(self) -> DataLoader:
        return self.get_dataloader('test', batch_size=self.hparams.eval_batch_size)

    # 删除生成文本中的原文本，截取生成的答案
    @staticmethod
    def intercept_answer(generated_text: list) -> list:
        predict_text = []
        for i, generated in enumerate(generated_text):
            matchobj = re.match(r"(.*)answer:(.*)", generated)
            predict_text.append(matchobj.group(2))  # (0)是全部句子 (1)是问题 (2)是答案

        return predict_text


def jaccard(str1, str2):
    a = set(str1.lower().split())
    b = set(str2.lower().split())
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))


# 为了解决进度条显示不全的问题（dynamic_ncols=True只能显示 80 列， 应该和pycharm或服务器有关）
# miniconda3\envs\lxl\lib\python3.8\site-packages\pytorch_lightning\callbacks\progress\tqdm_progress.py
class MyProgressBar(TQDMProgressBar):
    def init_train_tqdm(self):
        bar = Tqdm(
            desc=self.train_description,
            initial=self.train_batch_idx,
            position=(2 * self.process_position),
            disable=self.is_disabled,
            leave=True,
            ncols=150,
            file=sys.stdout,
            smoothing=0,
        )
        return bar

    # 健全性检查：先验证2个 step（可以改）-> 训练开始 -> 验证开始 -> 验证结束 -> 训练结束
    def on_validation_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        super().on_validation_epoch_end(trainer, pl_module)
        if not trainer.sanity_checking:
            metrics_dict = trainer.callback_metrics
            trainer.progress_bar_callback.print(f'valid end: {metrics_dict}\r')

    def on_test_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        super().on_test_end(trainer, pl_module)
        metrics_dict = trainer.callback_metrics
        logged_metrics = trainer.logged_metrics
        trainer.progress_bar_callback.print(f'test end: {metrics_dict}\r')
        trainer.progress_bar_callback.print(f'test end: {logged_metrics}\r')


def load_callbacks(args: argparse.Namespace):
    callbacks = []
    callbacks.append(plc.EarlyStopping(
        monitor='val_loss',
        mode='min',
        patience=3,
        min_delta=0.001
    ))

    callbacks.append(plc.ModelCheckpoint(
        dirpath=args.output_dir,
        monitor='val_loss',
        filename='GPT2-{epoch:02d}-{val_loss:.3f}',
        save_top_k=1,
        mode='min'
    ))

    callbacks.append(plc.LearningRateMonitor(
        logging_interval='step'))

    # callbacks.append(RichProgressBar())
    callbacks.append(MyProgressBar())

    return callbacks


if __name__ == '__main__':
    # 初始化参数列表
    args_dict = dict(
        output_dir="output",
        model_name_or_path='gpt2',
        tokenizer_name_or_path='gpt2',

        max_length=128,
        learning_rate=5e-4,
        train_batch_size=32,
        eval_batch_size=32,
        num_train_epochs=10,

        weight_decay=0.01,
        adam_epsilon=1e-8,
        warmup_steps=100,
        max_grad_norm=1.0,

        accelerator="gpu",
        devices=[1],
        logger=loggers.TensorBoardLogger(save_dir='logs', name='GPT2_logs'),

        seed=42,
        do_train=True,
        do_predict=True
    )
    args = argparse.Namespace(**args_dict)

    train_params = dict(
        max_epochs=args.num_train_epochs,
        gradient_clip_val=args.max_grad_norm,
        accelerator=args.accelerator,
        devices=args.devices,
        logger=args.logger,
        callbacks=load_callbacks(args)
    )

    set_seed(args)

    GPT2 = GPT2Model(args)
    trainer = pl.Trainer(**train_params)

    # 验证时进度条逐行打印是tqdm本身的问题，https://github.com/Lightning-AI/lightning/issues/16691
    # https://github.com/Lightning-AI/lightning/issues/14237
    if args.do_train:
        trainer.fit(GPT2)

    # T5 = T5.load_from_checkpoint("")
    if args.do_predict:
        trainer.test(GPT2)
