# 导入需要的库和模块
import os
import glob
import logging
import numpy as np
import torch
import argparse

# 设置环境变量 CUDA_LAUNCH_BLOCKING，确保 CUDA 错误能够被立即报告
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# 导入数据加载模块
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, DistributedSampler

# tqdm 用于进度条显示，trange 生成一个指定长度的进度条对象
from tqdm import tqdm, trange

# 从 load_data 模块导入训练、验证、测试集的缓存加载函数
from load_data import cache_examples_train, cache_examples_dev, cache_examples_test

# 从 transformers 库中导入不同模型用于分类任务的实现
from transformers.models.bert.modeling_bert import BertForSequenceClassification
from transformers.models.roberta.modeling_roberta import RobertaForSequenceClassification
from transformers.models.xlnet.modeling_xlnet import XLNetForSequenceClassification

# 导入配置类、分词器和优化器
from transformers import (
    WEIGHTS_NAME, AdamW, BertConfig, RobertaConfig, XLNetConfig, BertTokenizer, RobertaTokenizer, XLNetTokenizer,
    get_linear_schedule_with_warmup
)

# 导入评估函数
from transformers import glue_compute_metrics as compute_metrics

# 从数据工具模块中导入输出模式和任务处理器
from data_utils import output_modes, processors

# 设置日志记录器，指定日志格式和级别
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO
)
logger = logging.getLogger(__name__)  # 获取日志记录器

# 定义支持的模型类型及其对应的配置类、模型类、分词器类
MODEL_CLASSES = {
    "bert_raw": (BertConfig, BertForSequenceClassification, BertTokenizer),
    "roberta_raw": (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer),
    "xlnet_raw": (XLNetConfig, XLNetForSequenceClassification, XLNetTokenizer),
}


# 设置随机种子以确保实验的可重复性
def set_seed(args):
    np.random.seed(args.seed)  # 设置 numpy 的随机种子
    torch.manual_seed(args.seed)  # 设置 torch 的随机种子
    torch.cuda.manual_seed_all(args.seed)  # 为所有 GPU 设置随机种子
    torch.backends.cudnn.deterministic = True  # 确保 CUDNN 的行为确定性


# 训练函数，用于模型的训练过程
def train(args, train_dataset, model, tokenizer):
    # 设置训练批次大小（根据 GPU 数量调整）
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)

    # 定义数据采样器，如果分布式训练，使用 DistributedSampler，否则使用 RandomSampler
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)

    # 创建 DataLoader，按照指定的采样器和批次大小加载数据
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    # 如果设置了最大训练步数，则计算总训练步数，否则根据数据量和梯度累积步数计算
    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # 定义 AdamW 优化器并设置学习率和 epsilon
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, eps=args.adam_epsilon)

    # 设置学习率调度器，使用线性 warm-up 策略
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                num_training_steps=t_total)

    global_step = 0  # 初始化全局步数
    model.zero_grad()  # 清空模型梯度
    # trange 是 tqdm 的一种，提供 epoch 进度条
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0])

    # 开始每个 epoch 的训练循环
    for epoch in train_iterator:
        # tqdm 显示每个 iteration 的进度条
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])

        for step, batch in enumerate(epoch_iterator):
            model.train()  # 设置模型为训练模式
            batch = [t.to(args.device) for t in batch]  # 将 batch 数据转移到指定设备（GPU 或 CPU）

            # 将输入打包成字典传入模型
            inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}

            outputs = model(**inputs)  # 前向传播
            loss = outputs[0]  # 获取 loss 值

            # 如果有多个 GPU，计算平均 loss
            if args.n_gpu > 1:
                loss = loss.mean()

            # 反向传播计算梯度
            loss.backward()
            optimizer.step()  # 更新模型参数
            scheduler.step()  # 更新学习率调度器
            model.zero_grad()  # 清空梯度

            # 每 10 个 iteration 打印一次 loss
            if step % 10 == 0:
                logger.info(f"Epoch: {epoch}, Step: {step}, Loss: {loss.item()}")

            global_step += 1
            torch.cuda.empty_cache()  # 清空 CUDA 缓存


# 评估函数，用于模型的验证或测试过程
def evaluate(args, model, tokenizer, prefix="", evaluate="dev"):
    # 设置评估任务和输出目录
    eval_task_names = (args.task_name,)
    eval_outputs_dirs = (args.output_dir,)

    results = {}
    # 遍历任务列表（通常只有一个任务）
    for eval_task, eval_output_dir in zip(eval_task_names, eval_outputs_dirs):
        # 加载验证集或测试集的数据
        eval_dataset = cache_examples_dev(args, eval_task, tokenizer) if evaluate == "dev" else cache_examples_test(
            args, eval_task, tokenizer)
        eval_sampler = SequentialSampler(eval_dataset)  # 顺序采样器用于评估
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.per_gpu_eval_batch_size)

        # 如果有多个 GPU，使用 DataParallel 以并行执行
        if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
            model = torch.nn.DataParallel(model)

        nb_eval_steps = 0
        preds = None
        out_label_ids = None
        # 评估循环，逐批次计算模型输出
        for batch in eval_dataloader:
            model.eval()  # 设置模型为评估模式
            batch = tuple(t.to(args.device) for t in batch)  # 将数据移动到指定设备
            with torch.no_grad():  # 在评估过程中不计算梯度
                inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}
                outputs = model(**inputs)  # 前向传播
                logits = outputs[1]  # 获取模型的 logits 输出

            # 如果 preds 是 None，则初始化预测结果数组
            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = inputs["labels"].detach().cpu().numpy()
            else:
                # 将当前批次的结果追加到 preds 中
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)

        preds = np.argmax(preds, axis=1)  # 获取预测结果中概率最大的类别
        result = compute_metrics(eval_task, preds, out_label_ids)  # 计算评估指标
        results.update(result)  # 更新评估结果

        # 输出评估结果
        logger.info(f"***** Eval results {prefix} *****")
        for key in sorted(result.keys()):
            logger.info(f"  {key} = {result[key]}")

    return results


# 主函数
def main():
    # 设置训练和评估的参数
    args = argparse.Namespace(
        do_train=True,
        do_eval=True,
        model_type="bert_raw",
        model_name_or_path="F:/LLMtrainingfollowgithub/kesa/bert_base_uncased",
        task_name="sst-2",
        data_dir="F:/LLMtrainingfollowgithub/kesa/dataset/SST_2/",
        num_train_epochs=3.0,
        per_gpu_eval_batch_size=32,
        per_gpu_train_batch_size=32,
        max_seq_length=128,
        learning_rate=5e-5,
        seed=11,
        output_dir="F:/LLMtrainingfollowgithub/kesa/output/",
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),  # 设置设备（GPU 或 CPU）
        n_gpu=torch.cuda.device_count(),  # 获取可用 GPU 数量
        local_rank=-1,  # 分布式训练相关，默认不启用
        gradient_accumulation_steps=1,  # 梯度累积步数
        max_steps=-1,  # 默认不设置最大训练步数
        weight_decay=0.0,  # 权重衰减
        adam_epsilon=1e-8,  # Adam 优化器的 epsilon
        warmup_steps=0,  # 预热步数
        fp16=False,  # 是否启用混合精度训练
        fp16_opt_level="O1",  # 混合精度优化级别
        overwrite_cache=False  # 是否覆盖缓存数据
    )

    # 设置随机种子
    set_seed(args)

    # 检查任务名是否在支持的任务中
    if args.task_name not in processors:
        raise ValueError("Task not found: %s" % (args.task_name))

    # 获取任务的处理器和标签
    processor = processors[args.task_name]()
    args.output_mode = output_modes[args.task_name]
    label_list = processor.get_labels()
    num_labels = len(label_list)

    # 加载预训练模型、配置和分词器
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path)  # 加载分词器
    config = config_class.from_pretrained(args.model_name_or_path, num_labels=num_labels)  # 加载配置
    model = model_class.from_pretrained(args.model_name_or_path, config=config)  # 加载模型
    model.to(args.device)  # 将模型转移到指定设备

    # 如果启用训练，加载训练数据并执行训练
    if args.do_train:
        logger.info("Loading training data from %s", args.data_dir)
        train_dataset = cache_examples_train(args, args.task_name, tokenizer)
        logger.info("Training data loaded successfully.")
        train(args, train_dataset, model, tokenizer)  # 调用训练函数

    # 如果启用评估，执行评估流程
    if args.do_eval:
        logger.info("Evaluation process started...")
        evaluate(args, model, tokenizer)  # 调用评估函数


# 脚本入口，执行主函数
if __name__ == "__main__":
    main()
