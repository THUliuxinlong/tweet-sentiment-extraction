import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import pandas as pd
import torch
import argparse
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, Seq2SeqTrainingArguments, Seq2SeqTrainer, \
    EarlyStoppingCallback, EvalPrediction, DataCollatorForSeq2Seq, AutoConfig
from datasets import load_dataset
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType, PeftType
from peft import prepare_model_for_int8_training


def main():
    # 设置随机数种子
    def set_seed(args: argparse.Namespace):
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        # if args.devices > 0:
        #     torch.cuda.manual_seed_all(args.seed)


    def preprocess_function(examples):
        sentiment = pd.Series([q.strip() for q in examples["sentiment"]])
        text = pd.Series([q.strip() for q in examples["text"]])
        processed_text = "question: " + sentiment + " context: " + text

        inputs = tokenizer(processed_text.to_list(), max_length=args.max_input_length, padding='max_length',
                                truncation=True, return_tensors='pt')

        labels = tokenizer(examples["selected_text"], max_length=args.max_target_length, truncation=True,
                           padding='max_length', return_tensors='pt')
        labels["input_ids"][labels["input_ids"][:, :] == tokenizer.pad_token_id] = -100

        inputs["labels"] = labels["input_ids"]
        return inputs


    # EvalPrediction: predictions(np.ndarray), label_ids(np.ndarray, inputs(np.ndarray, optional)
    # predictions:tuple 2ndarray(N, )   label_ids: tuple 2ndarray(N, L)  inputs: ndarray(N, L)
    def compute_metrics(p:EvalPrediction):
        decoded_preds = tokenizer.batch_decode(p.predictions, skip_special_tokens=True)
        labels = np.where(p.label_ids != -100, p.label_ids, tokenizer.pad_token_id)  # Replace -100 in the labels
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        jaccard_scores = [jaccard(p, t) for p, t in zip(decoded_preds, decoded_labels)]
        avg_jaccard = sum(jaccard_scores) / len(jaccard_scores)
        return {'jaccard': avg_jaccard}  # return a dictionary string to metric values.


    def jaccard(str1, str2):
        a = set(str1.lower().split())
        b = set(str2.lower().split())
        c = a.intersection(b)
        return float(len(c)) / (len(a) + len(b) - len(c))


    # 初始化参数列表
    args_dict = dict(
        output_dir="output",
        model_name_or_path='t5-3b',  # t5-small t5-3b
        tokenizer_name_or_path='t5-3b',

        max_input_length=128,
        max_target_length=128,
        seed=42,
        do_train=True,
        do_predict=True
    )
    args = argparse.Namespace(**args_dict)

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=10,
        learning_rate=2e-4,

        evaluation_strategy='epoch',
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,
        dataloader_num_workers=4,
        auto_find_batch_size=True,
        # gradient_checkpointing=True,
        # fp16=True,

        lr_scheduler_type='linear',
        weight_decay=0.01,
        adam_epsilon=1e-8,
        max_grad_norm=1.0,
        # warmup_ratio=0.1,
        # warmup_steps=100,

        # no_cuda=True,  # run cpu
        seed=args.seed,
        # include_inputs_for_metrics=True,
        logging_dir='logs/T5_3B_lora_logs',
        logging_strategy='steps',
        logging_steps=1,
        save_strategy="epoch",  # 保存ckpt
        save_total_limit=1,
        metric_for_best_model='eval_loss',
        load_best_model_at_end=True,
        greater_is_better=False,
        predict_with_generate=True,
        generation_max_length=128,
    )

    set_seed(args)

    peft_type = PeftType.LORA
    peft_config = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, task_type=TaskType.SEQ_2_SEQ_LM, bias="none",
                             inference_mode=False, target_modules=["q", "v"])  # r:中间层神经元的个数 alpha:scale参数

    # peft_type = PeftType.PREFIX_TUNING
    # peft_config = PrefixTuningConfig(task_type="SEQ_CLS", num_virtual_tokens=20)  # num_virtual_tokens prompt的长度(10-20)

    # peft_type = PeftType.P_TUNING
    # peft_config = PromptEncoderConfig(task_type="SEQ_CLS", num_virtual_tokens=20, encoder_hidden_size=128)  # MLP中间层的参数

    # peft_type = PeftType.PROMPT_TUNING
    # peft_config = PromptTuningConfig(task_type="SEQ_CLS", num_virtual_tokens=20)


    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name_or_path)
    # model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name_or_path, load_in_8bit=True, device_map="auto")
    # model = prepare_model_for_int8_training(model)
    print(model.get_memory_footprint() / 1024 / 1024 / 1024, 'GB')
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, model_max_length=512)
    train = load_dataset("csv", data_files="./data/train.csv")
    valid = load_dataset("csv", data_files="./data/valid.csv")
    test = load_dataset("csv", data_files="./data/test.csv")
    print(train, valid, test)

    train_dataset = train['train'].map(preprocess_function, batched=True, remove_columns=train["train"].column_names)
    valid_dataset = valid['train'].map( preprocess_function, batched=True, remove_columns=valid["train"].column_names)
    test_dataset = test['train'].map(preprocess_function, batched=True, remove_columns=test["train"].column_names)

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
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

if __name__ == '__main__':
    main()

# class SchedulerType(ExplicitEnum):
#     LINEAR = "linear"
#     COSINE = "cosine"
#     COSINE_WITH_RESTARTS = "cosine_with_restarts"
#     POLYNOMIAL = "polynomial"
#     CONSTANT = "constant"
#     CONSTANT_WITH_WARMUP = "constant_with_warmup"
#     INVERSE_SQRT = "inverse_sqrt"
