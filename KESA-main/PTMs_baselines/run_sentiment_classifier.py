import os
import glob
import logging
import numpy as np
import torch
import argparse

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, DistributedSampler
from tqdm import tqdm, trange
from load_data import cache_examples_train, cache_examples_dev, cache_examples_test
from transformers.models.bert.modeling_bert import BertForSequenceClassification
from transformers.models.roberta.modeling_roberta import RobertaForSequenceClassification
from transformers.models.xlnet.modeling_xlnet import XLNetForSequenceClassification
from transformers import (
    WEIGHTS_NAME, AdamW, BertConfig, RobertaConfig, XLNetConfig, BertTokenizer, RobertaTokenizer, XLNetTokenizer,
    get_linear_schedule_with_warmup
)
from transformers import glue_compute_metrics as compute_metrics
from data_utils import output_modes, processors


# 设置日志记录器
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

MODEL_CLASSES = {
    "bert_raw": (BertConfig, BertForSequenceClassification, BertTokenizer),
    "roberta_raw": (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer),
    "xlnet_raw": (XLNetConfig, XLNetForSequenceClassification, XLNetTokenizer),
}

# 设置随机种子
def set_seed(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True

# 训练函数
def train(args, train_dataset, model, tokenizer):
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    optimizer = AdamW(model.parameters(), lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                num_training_steps=t_total)

    global_step = 0
    model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0])

    for epoch in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):
            model.train()
            batch = [t.to(args.device) for t in batch]
            inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}

            outputs = model(**inputs)
            loss = outputs[0]

            if args.n_gpu > 1:
                loss = loss.mean()

            loss.backward()
            optimizer.step()
            scheduler.step()
            model.zero_grad()
            # 打印每个 iteration 的 loss 值
            if step % 10 == 0:  # 每 10 个 iteration 打印一次 loss，可以根据需要调整频率
                logger.info(f"Epoch: {epoch}, Step: {step}, Loss: {loss.item()}")

            global_step += 1
            torch.cuda.empty_cache()

# 评估函数
def evaluate(args, model, tokenizer, prefix="", evaluate="dev"):
    eval_task_names = (args.task_name,)
    eval_outputs_dirs = (args.output_dir,)

    results = {}
    for eval_task, eval_output_dir in zip(eval_task_names, eval_outputs_dirs):
        eval_dataset = cache_examples_dev(args, eval_task, tokenizer) if evaluate == "dev" else cache_examples_test(
            args, eval_task, tokenizer)
        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.per_gpu_eval_batch_size)

        if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
            model = torch.nn.DataParallel(model)

        nb_eval_steps = 0
        preds = None
        out_label_ids = None
        for batch in eval_dataloader:
            model.eval()
            batch = tuple(t.to(args.device) for t in batch)
            with torch.no_grad():
                inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}
                outputs = model(**inputs)
                logits = outputs[1]

            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = inputs["labels"].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)

        preds = np.argmax(preds, axis=1)
        result = compute_metrics(eval_task, preds, out_label_ids)
        results.update(result)

        logger.info(f"***** Eval results {prefix} *****")
        for key in sorted(result.keys()):
            logger.info(f"  {key} = {result[key]}")

    return results

# 主函数
def main():
    # 设置参数
    args = argparse.Namespace(
        do_train=True,
        do_eval=True,
        model_type="bert_raw",
        model_name_or_path="F:/Github_learning/kesa/bert_base_uncased",
        task_name="sst-2",
        data_dir="F:/Github_learning/kesa/dataset/SST_2/",
        num_train_epochs=3.0,
        per_gpu_eval_batch_size=32,
        per_gpu_train_batch_size=32,
        max_seq_length=128,
        learning_rate=5e-5,
        seed=11,
        output_dir="F:/Github_learning/kesa/output/",
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        n_gpu=torch.cuda.device_count(),
        local_rank=-1,
        gradient_accumulation_steps=1,
        max_steps=-1,
        weight_decay=0.0,
        adam_epsilon=1e-8,
        warmup_steps=0,
        fp16=False,
        fp16_opt_level="O1",
        overwrite_cache = False
    )

    # 设置随机种子
    set_seed(args)

    # 检查任务名
    if args.task_name not in processors:
        raise ValueError("Task not found: %s" % (args.task_name))

    processor = processors[args.task_name]()
    args.output_mode = output_modes[args.task_name]
    label_list = processor.get_labels()
    num_labels = len(label_list)

    # 加载预训练模型和分词器
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path)
    config = config_class.from_pretrained(args.model_name_or_path, num_labels=num_labels)
    model = model_class.from_pretrained(args.model_name_or_path, config=config)
    model.to(args.device)

    # 训练
    if args.do_train:
        logger.info("Loading training data from %s", args.data_dir)
        train_dataset = cache_examples_train(args, args.task_name, tokenizer)
        logger.info("Training data loaded successfully.")
        train(args, train_dataset, model, tokenizer)

    # 评估
    if args.do_eval:
        logger.info("Evaluation process started...")
        evaluate(args, model, tokenizer)

if __name__ == "__main__":
    main()
