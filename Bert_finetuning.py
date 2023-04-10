import numpy as np
import pandas as pd
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import transformers
import tokenizers
from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from collections import defaultdict
import datetime


class TweetDataset(Dataset):
    def __init__(self, text, sentiment, tokenizer, max_len, is_training=False, selected_text=None):
        self.text = text
        self.sentiment = sentiment
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.is_training = is_training
        if is_training:
            self.selected_text = selected_text

    def __len__(self):
        return len(self.text)

    def __getitem__(self, item):
        text = self.text[item]
        sentiment = self.sentiment[item]

        encoding = self.tokenizer.encode_plus(
            text=sentiment, text_pair=text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=True,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        # 训练的时候需要根据selected_text计算起始位置和结束位置
        if self.is_training:
            selected_text = self.selected_text[item]
            # 找到selected在text中的开始位置和结束位置
            target_start = None
            target_end = None
            for i, char in enumerate(text):
                if char == selected_text[0] and text[i: i + len(selected_text)] == selected_text:
                    target_start = i
                    target_end = i + len(selected_text)
                    #分类的结果在0~man_len-1之间，如果实际结果超出范围，损失没法计算
                    if target_start >= self.max_len:
                        target_start = self.max_len - 1
                    if target_end >= self.max_len:
                        target_end = self.max_len - 1

            return {
                'text': text,
                'selected_text': selected_text,
                'target_start': target_start,
                'target_end': target_end,
                'sentiment': sentiment,
                'input_ids': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten(),
                'token_type_ids': encoding['token_type_ids'].flatten()
            }
        else:
            return {
                'text': text,
                'sentiment': sentiment,
                'input_ids': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten(),
                'token_type_ids': encoding['token_type_ids'].flatten()
            }


# dataloader
def create_data_loader(df, tokenizer, max_len, batch_size, is_training):
    if is_training:
        ds = TweetDataset(
            text=df['text'],
            selected_text=df['selected_text'],
            sentiment=df['sentiment'],
            tokenizer=tokenizer,
            max_len=max_len,
            is_training=is_training
        )
    else:
        ds = TweetDataset(
            text=df['text'],
            sentiment=df['sentiment'],
            tokenizer=tokenizer,
            max_len=max_len,
            is_training=is_training
        )

    return DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=4
    )


# 定义模型
class TweetExtraction(nn.Module):
    def __init__(self, output_dim):
        super(TweetExtraction, self).__init__()
        self.bert = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME, return_dict=False)
        self.drop = nn.Dropout(p=0.3)
        self.out = nn.Linear(self.bert.config.hidden_size, output_dim)

    def forward(self, input_ids, attention_mask, token_type_ids):
        last_hidden_state, _ = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )

        output = self.drop(last_hidden_state)
        logits = self.out(output)  # (bz, max_len, 768) -> (bz, max_len, 2)
        # 起始位置和结束位置的logit
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)  # (bz, max_len, 1) -> (bz, max_len)
        end_logits = end_logits.squeeze(-1)
        return start_logits, end_logits
# '''
# test
# '''
# tmp = TweetDataset(df_train['text'], df_train['selected_text'], df_train['sentiment'], tokenizer, 60)
# tmp_model = TweetExtraction(2)
# tmp_result = tmp_model(tmp.__getitem__(2)['input_ids'].reshape(1,-1),tmp.__getitem__(2)['attention_mask'].reshape(1,-1),tmp.__getitem__(2)['token_type_ids'].reshape(1,-1))


# 定义损失函数
def self_loss(start_logist, end_logist, start_position, end_position):
    loss_fn = nn.CrossEntropyLoss().to(device)
    start_loss = loss_fn(start_logist, start_position)
    end_loss = loss_fn(end_logist, end_position)
    return start_loss + end_loss


def jaccard(str1, str2):
    a = set(str1.lower().split())
    b = set(str2.lower().split())
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))


def calculate_jaccard_score(text, orig_selected, pred_start, pred_end):
    if pred_start > pred_end:
        pred_start = pred_end
    pred_selected = text[pred_start:pred_end]
    return jaccard(orig_selected, pred_selected)


# 定义EarlyStopping
class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


# 定义一次训练
def train_epoch(model, data_loader, loss_fn, optimizer, scheduler):
    model = model.train()
    losses = []
    jaccards = []
    step = 0

    for data in data_loader:
        input_ids = data["input_ids"].to(device)
        attention_mask = data["attention_mask"].to(device)
        token_type_ids = data["token_type_ids"].to(device)
        target_start = data["target_start"].to(device)
        target_end = data["target_end"].to(device)

        start_logits, end_logits = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        pred_start = torch.argmax(start_logits, dim=-1)
        pred_end = torch.argmax(end_logits, dim=-1)

        # 保留一个batch的总loss
        loss = loss_fn(start_logits, end_logits, target_start, target_end)
        losses.append(loss.item())

        # 保存一个batch的平均jaccard
        tmp_jaccards = []
        for i, text in enumerate(data['text']):
            jaccard = calculate_jaccard_score(text, data['selected_text'][i], pred_start[i], pred_end[i])
            tmp_jaccards.append(jaccard)
        jaccards.append(np.mean(tmp_jaccards))

        loss.backward()
        # clip_grad_norm_裁剪模型的梯度来避免梯度爆炸。
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        if step % 100 == 0:
            print(f'setp {step}: loss:{loss:.6f} jaccard:{np.mean(tmp_jaccards):.6f}')
        step += 1

    return losses, jaccards


# 评估模型
def eval_model(model, data_loader, loss_fn, return_predictions=False):
    model = model.eval()
    losses = []
    jaccards = []
    pred_selected_texts = []

    with torch.no_grad():
        for data in data_loader:
            text = data["text"]

            input_ids = data["input_ids"].to(device)
            attention_mask = data["attention_mask"].to(device)
            token_type_ids = data["token_type_ids"].to(device)
            target_start = data["target_start"].to(device)
            target_end = data["target_end"].to(device)

            start_logits, end_logits = model(input_ids=input_ids, attention_mask=attention_mask,
                                             token_type_ids=token_type_ids)
            pred_start = torch.argmax(start_logits, dim=-1)
            pred_end = torch.argmax(end_logits, dim=-1)

            # 保留一个batch的总loss
            loss = loss_fn(start_logits, end_logits, target_start, target_end)
            losses.append(loss.item())

            # 保存一个batch的平均jaccard
            tmp_jaccards = []
            for i, one_piece in enumerate(text):
                jaccard = calculate_jaccard_score(one_piece, data['selected_text'][i], pred_start[i], pred_end[i])
                tmp_jaccards.append(jaccard)
            jaccards.append(np.mean(tmp_jaccards))

            if return_predictions:
                # 从text中截取selected_text
                for i, one_piece in enumerate(text):
                    selected_text = one_piece[pred_start[i]:pred_end[i]]
                    pred_selected_texts.append(selected_text)

    return losses, jaccards, pred_selected_texts


# 做预测
def get_predictions(model, data_loader):
    model = model.eval()
    pred_selected_texts = []
    print('predicting...')

    with torch.no_grad():
        for data in data_loader:
            text = data["text"]
            input_ids = data["input_ids"].to(device)
            attention_mask = data["attention_mask"].to(device)
            token_type_ids = data["token_type_ids"].to(device)

            start_logits, end_logits = model(input_ids=input_ids, attention_mask=attention_mask,
                                             token_type_ids=token_type_ids)
            pred_start = torch.argmax(start_logits, dim=-1)
            pred_end = torch.argmax(end_logits, dim=-1)

            # 从text中截取selected_text
            for i, one_piece in enumerate(text):
                selected_text = one_piece[pred_start[i]:pred_end[i]]
                pred_selected_texts.append(selected_text)
    return pred_selected_texts


# 生成submission
def create_submission(pred_selected_texts):
    submission = pd.DataFrame(data=df_test['textID'], columns=['textID'])
    submission.insert(loc=1, column='selected_text', value=pred_selected_texts)
    time = datetime.datetime.now().strftime('%H%M%S')  # ('%Y-%m-%d %H:%M:%S')加个时间好区分
    submission.to_csv(f'./output/{PRE_TRAINED_MODEL_NAME}_{time}.csv', index=False)
    print('submission finished')

'''
加载数据, 创建Dataloader
'''
df_train = pd.read_csv('./data/train.csv')
df_valid = pd.read_csv('./data/valid.csv')
df_test = pd.read_csv('./data/test.csv')

print(f'train_data.shape:{df_train.shape}, valid_data.shape:{df_valid.shape}, test_data.shape:{df_test.shape}')

# 加载tokenizer
PRE_TRAINED_MODEL_NAME = 'bert-base-cased'
tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)

# # 各情感的数量和长度分布
# sentiment_counts = df_train['sentiment'].value_counts()
# print(sentiment_counts)
# sns.countplot(x=df_train['sentiment'])
# plt.show()
# # # 各文本的长度分布（bert的tokenizer划分的不是词的数量）
# # df_train['word_counts'] = df_train['text'].apply(lambda x: len(str(x).split()))
# # df_train['word_counts_ST'] = df_train['selected_text'].apply(lambda x: len(str(x).split()))
# # sns.kdeplot(x=df_train.word_counts, hue=df_train.sentiment)
# # plt.title('word count in text by sentiment group')
# # plt.show()
# text_lens = []
# selected_text_lens = []
# for _, data in df_train.iterrows():
#     text_tokens = tokenizer.encode(data['text'], max_length=512, truncation=True)
#     text_lens.append(len(text_tokens))
#     selected_text_tokens = tokenizer.encode(data['selected_text'], max_length=512, truncation=True)
#     selected_text_lens.append(len(selected_text_tokens))
# sns.histplot(text_lens)
# plt.xlabel('Length of text')
# plt.show()
# sns.histplot(selected_text_lens)
# plt.xlabel('Length of selected-text')
# plt.show()

# 划分数据，构建dataloader
max_len = 128
batch_size = 32

train_dataloader = create_data_loader(df_train, tokenizer, max_len=max_len, batch_size=batch_size, is_training=True)
vaild_dataloader = create_data_loader(df_valid, tokenizer, max_len=max_len, batch_size=batch_size, is_training=True)
# 如果测试集没有标签is_training=False
test_dataloader = create_data_loader(df_test, tokenizer, max_len=max_len, batch_size=batch_size, is_training=True)

'''
构建model optimizer scheduler
'''
EPOCHS = 10
learning_rate = 2e-5
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
model = TweetExtraction(output_dim=2).to(device)
optimizer = AdamW(model.parameters(), lr=learning_rate, correct_bias=False)

total_steps = len(train_dataloader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps=total_steps
)

# early stopping patience; how long to wait after last time validation loss improved.
patience = 3
model_save_path = f'./output/{PRE_TRAINED_MODEL_NAME}.pt'
early_stopping = EarlyStopping(path=model_save_path, patience=patience, verbose=True)

'''
训练
'''
history = defaultdict(list)
for epoch in range(EPOCHS):
    print('\n ', f'Epoch {epoch + 1}/{EPOCHS}', '\n', '-' * 10)

    train_losses, train_jaccards = train_epoch(model, train_dataloader, self_loss, optimizer, scheduler)
    print('\n', f'Train loss: {train_losses[-1]:.6f} jaccard: {train_jaccards[-1]:.6f}')

    vaild_losses, vaild_jaccards, _ = eval_model(model, vaild_dataloader, self_loss)
    print(f'Vaild loss:{np.mean(vaild_losses):.6f} jaccard:{np.mean(vaild_jaccards):.6f}')

    history['train_jaccard'].append(train_jaccards)
    history['train_loss'].append(train_losses)
    history['vaild_jaccard'].append(np.mean(vaild_jaccards))
    history['vaild_loss'].append(np.mean(vaild_losses))

    # early_stopping needs the validation loss to check if it has decresed,
    # and if it has, it will make a checkpoint of the current model
    early_stopping(np.mean(vaild_losses), model)
    if early_stopping.early_stop:
        print("Early stopping")
        break

'''
训练过程可视化
'''
# 展平
history['train_jaccard'] = np.array(history['train_jaccard']).flat
history['train_loss'] = np.array(history['train_loss']).flat
history['vaild_jaccard'] = np.array(history['vaild_jaccard']).flat
history['vaild_loss'] = np.array(history['vaild_loss']).flat

plt.plot(history['train_jaccard'])
plt.ylabel('Train jaccard')
plt.xlabel('Step')
plt.show()
plt.plot(history['vaild_jaccard'])
plt.ylabel('Vaild jaccard')
plt.xlabel('Step')
plt.show()

plt.plot(history['train_loss'])
plt.ylabel('Train loss')
plt.xlabel('Step')
plt.show()
plt.plot(history['vaild_loss'])
plt.ylabel('Vaild loss')
plt.xlabel('Step')
plt.show()

'''
预测
'''
# load the last checkpoint with the best model
model.load_state_dict(torch.load(model_save_path))
test_loss, test_jaccards, pred_selected_texts = eval_model(model, test_dataloader, self_loss, return_predictions=True)
print(f'test_loss:{np.mean(vaild_jaccards)}, test_jaccards:{np.mean(test_jaccards)}')
create_submission(pred_selected_texts)
