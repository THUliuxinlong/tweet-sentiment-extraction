import os
import numpy as np
import pandas as pd
import torch
import argparse
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, TrainingArguments, Trainer, \
    EarlyStoppingCallback, EvalPrediction, BloomForQuestionAnswering
from datasets import load_dataset
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType, PeftType

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 设置随机数种子
def set_seed(args: argparse.Namespace):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    # if args.devices > 0:
    #     torch.cuda.manual_seed_all(args.seed)


def preprocess(examples):
    sentiment = pd.Series([q.strip() for q in examples["sentiment"]])
    text = pd.Series([q.strip() for q in examples["text"]])
    selected_text = pd.Series([q.strip() for q in examples["selected_text"]])
    processed_text = "question: " + sentiment + " context: " + text

    inputs = tokenizer(processed_text.to_list(), max_length=args.max_len, padding='max_length',
                            truncation=True, return_tensors='pt')

    start_positions = np.array([processed_text[i].find(selected_text[i]) for i in range(len(selected_text))])
    end_positions = np.array([start_positions[i] + len(selected_text[i]) for i in range(len(selected_text))])

    start_positions = np.where(start_positions >= args.max_len, args.max_len - 1, start_positions)
    end_positions = np.where(end_positions >= args.max_len, args.max_len - 1, end_positions)

    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions

    return inputs


# EvalPrediction: predictions(np.ndarray), label_ids(np.ndarray, inputs(np.ndarray, optional)
# predictions:tuple 2ndarray(2748, 128)   label_ids: tuple 2ndarray(2748,)  inputs: ndarray(2748, 128) encodings
def compute_metrics(p: EvalPrediction):
    pred_start, pred_end = np.argmax(p.predictions, axis=-1)
    target_start, target_end = p.label_ids
    processed_text = tokenizer.batch_decode(p.inputs, skip_special_tokens=True, clean_up_tokenization_spaces=True)

    jaccard_scores = calculate_jaccard_score(processed_text, pred_start, pred_end, target_start, target_end)
    avg_jaccard = sum(jaccard_scores) / len(jaccard_scores)
    return {'jaccard': avg_jaccard}  # return a dictionary string to metric values.


def calculate_jaccard_score(text, pred_start, pred_end, target_start, target_end):
    mask = pred_start > pred_end
    pred_start[mask] = pred_end[mask]
    target_selected = [s[target_start[i]:target_end[i]] for i, s in enumerate(text)]
    pred_selected = [s[pred_start[i]:pred_end[i]] for i, s in enumerate(text)]

    jaccard_scores = [jaccard(p, t) for p, t in zip(target_selected, pred_selected)]
    return jaccard_scores


def jaccard(str1, str2):
    a = set(str1.lower().split())
    b = set(str2.lower().split())
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))


if __name__ == '__main__':
    # 初始化参数列表
    args_dict = dict(
        output_dir="output",
        model_name_or_path="bigscience/bloom-560m",  # "bigscience/bloom-3b"
        tokenizer_name_or_path="bigscience/bloom-560m",

        num_train_epochs=10,
        learning_rate=2e-4,
        max_len=200,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,

        weight_decay=0.01,
        adam_epsilon=1e-8,
        warmup_steps=100,
        max_grad_norm=1.0,

        logging_dir='logs/Bloom_logs',
        seed=42,
        do_train=True,
        do_predict=True
    )
    args = argparse.Namespace(**args_dict)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,

        evaluation_strategy='epoch',
        label_names=["start_positions", "end_positions"],
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        dataloader_num_workers=4,

        weight_decay=args.weight_decay,
        adam_epsilon=args.adam_epsilon,
        max_grad_norm=args.max_grad_norm,
        warmup_steps=args.warmup_steps,

        # no_cuda=True,  # run cpu
        seed=args.seed,
        include_inputs_for_metrics=True,
        logging_dir=args.logging_dir,
        logging_strategy='steps',
        logging_steps=20,
        save_strategy="epoch",  # 保存ckpt
        metric_for_best_model='eval_loss',
        load_best_model_at_end=True,
        greater_is_better=False,
    )

    set_seed(args)

    # peft_type = PeftType.LORA
    # peft_config = LoraConfig(r=8, lora_alpha=32, lora_dropout=0.1, task_type=TaskType.TOKEN_CLS,
    #                          inference_mode=False, target_modules=["query_key_value"])  # r:中间层神经元的个数 alpha:scale参数

    # peft_type = PeftType.PREFIX_TUNING
    # peft_config = PrefixTuningConfig(task_type="SEQ_CLS", num_virtual_tokens=20)  # num_virtual_tokens prompt的长度(10-20)

    # peft_type = PeftType.P_TUNING
    # peft_config = PromptEncoderConfig(task_type="SEQ_CLS", num_virtual_tokens=20, encoder_hidden_size=128)  # MLP中间层的参数

    # peft_type = PeftType.PROMPT_TUNING
    # peft_config = PromptTuningConfig(task_type="SEQ_CLS", num_virtual_tokens=20)

    model = BloomForQuestionAnswering.from_pretrained(args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    # model = get_peft_model(model, peft_config)
    # model.print_trainable_parameters()

    train = load_dataset("csv", data_files="./data/train.csv")
    valid = load_dataset("csv", data_files="./data/valid.csv")
    test = load_dataset("csv", data_files="./data/test.csv")
    print(train, valid, test)

    train_dataset = train['train'].map(preprocess, batched=True, remove_columns=train["train"].column_names)
    valid_dataset = valid['train'].map( preprocess, batched=True, remove_columns=valid["train"].column_names)
    test_dataset = test['train'].map(preprocess, batched=True, remove_columns=test["train"].column_names)

    trainer = Trainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3, early_stopping_threshold=0.01)],
        compute_metrics=compute_metrics
    )

    if args.do_train:
        print('Training')
        trainer.train()

    if args.do_predict:
        print('Predict')
        predictions, label_ids, metrics = trainer.predict(test_dataset=test_dataset)
        print(metrics)
