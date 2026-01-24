import gc
import json
import os
import argparse
import fnmatch

import numpy as np
import torch
import random
import time

np.random.seed(0)
torch.manual_seed(0)

from transformers import AutoModelForCausalLM, AutoTokenizer

from pruner import *
from lm_eval import tasks, evaluator

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def pattern_match(patterns, source_list):
    task_names = set()
    for pattern in patterns:
        for matching in fnmatch.filter(source_list, pattern):
            task_names.add(matching)
    return list(task_names)


class MultiChoice:
    def __init__(self, choices):
        self.choices = choices

    # Simple wildcard support (linux filename patterns)
    def __contains__(self, values):
        for value in values.split(","):
            if len(fnmatch.filter(self.choices, value)) == 0:
                return False

        return True

    def __iter__(self):
        for choice in self.choices:
            yield choice


@torch.no_grad()
def eval_zero(args, model, tokenizer):
    task_names = pattern_match(args.tasks.split(","), tasks.ALL_TASKS)

    print(f"Selected Tasks: {task_names}")

    description_dict = {}
    if args.description_dict_path:
        with open(args.description_dict_path, "r") as f:
            description_dict = json.load(f)

    results = evaluator.simple_evaluate(
        model_type=args.model_type,
        model=(tokenizer, model),
        model_args=args.model_args,
        tasks=task_names,
        num_fewshot=args.num_fewshot,
        batch_size=args.batch_size,
        device=args.device,
        no_cache=args.no_cache,
        limit=args.limit,
        description_dict=description_dict,
        decontamination_ngrams_path=args.decontamination_ngrams_path,
        check_integrity=args.check_integrity,
    )

    if results is None:
        return

    dumped = json.dumps(results, indent=2)
    print(dumped)

    if args.output_path:
        import os
        directory_path = os.path.dirname(args.output_path)
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)

        with open(args.output_path, "w") as f:
            f.write(dumped)

    print(
        f"{args.model_type} ({args.model_args}), limit: {args.limit}, provide_description: {args.provide_description}, "
        f"num_fewshot: {args.num_fewshot}, batch_size: {args.batch_size}"
    )
    print(evaluator.make_table(results))

    task_acc = results["results"]
    acc_res = []
    if "storycloze" in task_names:
        acc_res.append(task_acc["storycloze"]["acc"] * 100)
        task_names.pop(task_names.index("storycloze"))
    if "rte" in task_names:
        acc_res.append(task_acc["rte"]["acc"] * 100)
        task_names.pop(task_names.index("rte"))
    if len(acc_res) > 0:
        print("|".join(["%.2f" % c for c in acc_res]))
        acc_res = []

    need_task_name = ["winogrande", "hellaswag", "boolq", "piqa", "openbookqa", "arc_easy", "arc_challenge"]
    for task_name in need_task_name:
        if task_name not in task_acc:
            acc_res.append(0)
            continue
        if "acc_norm" in task_acc[task_name] and task_name != "piqa":
            acc_res.append(task_acc[task_name]["acc_norm"] * 100)
        else:
            acc_res.append(task_acc[task_name]["acc"] * 100)
    mean_acc = sum(acc_res) / len(acc_res)
    acc_res.append(mean_acc)
    print("|".join(["%.2f" % c for c in acc_res]))


def make_parser():
    parser = argparse.ArgumentParser("Pruner Processor")

    parser.add_argument("-p", "--pruner", default=None, type=str, help="pruner")
    parser.add_argument("-s", "--final_s", default=0., type=float, help="final sparsity")
    parser.add_argument("-d", "--dataset", default="c4", type=str, help="dataset")

    parser.add_argument('--base_model', type=str, help='model name')

    # remove
    parser.add_argument("--remove_list", nargs='+', default=[], help='remove transformer list')

    # control
    parser.add_argument("--fp16", action='store_true', help='use fp16')
    parser.add_argument("--bf16", action='store_true', help='use bf16')
    parser.add_argument("--origin", action='store_true', help='original stream')

    ## eval zero
    parser.add_argument('--model_type', type=str, default="hf-causal-experimental", help='model type')
    parser.add_argument("--model_args", default="pretrained=facebook/opt-125m")
    parser.add_argument("--tasks", default=None, choices=MultiChoice(tasks.ALL_TASKS))
    parser.add_argument("--provide_description", action="store_true")
    parser.add_argument("--num_fewshot", type=int, default=0)
    parser.add_argument("--batch_size", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output_path", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no_cache", action="store_true")
    parser.add_argument("--decontamination_ngrams_path", default=None)
    parser.add_argument("--description_dict_path", default=None)
    parser.add_argument("--check_integrity", action="store_true")

    parser.add_argument("--seed", default=0, type=int, help='seed')

    ## spsr
    parser.add_argument("--output_dir", default="/media/data/yzg", type=str, help="output_dir")
    parser.add_argument('--epochs', type=int, default=10, help='epochs')
    parser.add_argument("--lr", default=3e-4, type=float, help="lr")

    #linear_patch
    parser.add_argument("--train_size", type=int, default=5000, help="Number of training data samples.")
    parser.add_argument("--val_size", type=int, default=16, help="Number of validation data samples.")
    parser.add_argument('--insert_type', type=str, default='rotate', help='insert type')
    parser.add_argument("--min_lr_factor", type=float, default=20, help="min_lr = lr/min_lr_factor")
    parser.add_argument("--wd", type=float, default=1e-4, help="weight decay")
    parser.add_argument("--early_stop", type=int, default=0, help="early stoping after validation loss do not decrease")

    return parser


def load_model_tokenizer(args):
    ckpt_dir = args.base_model
    tokenizer = AutoTokenizer.from_pretrained(ckpt_dir, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

    dtype = torch.float32
    if args.fp16:
        dtype = torch.float16
    if args.bf16:
        dtype = torch.bfloat16

    model = AutoModelForCausalLM.from_pretrained(
        ckpt_dir,
        trust_remote_code=True,
        # quantization_config=bnb_config,  # 上面本地模型的配置
        # device_map="cpu" if args.deepsparse else "auto",  # 使用GPU的编号
        device_map="auto",
        torch_dtype=dtype,
    )
    return tokenizer, model


def remove_layers_and_update_config(model, remove_indices, model_type="llama"):
    # remove_indices: 可迭代的要删除的层索引（int）
    remove_indices = [int(i) for i in remove_indices]

    # 找到原始层列表和 config 的字段
    if "Llama" in model_type or "llama" in model_type:
        layers = model.model.layers
        cfg = model.config
        num_layers_attr = "num_hidden_layers"
    elif "opt" in model_type:
        layers = model.model.decoder.layers
        cfg = model.config
        num_layers_attr = "num_hidden_layers"
    elif "Qwen" in model_type:
        layers = model.transformer.h
        cfg = model.config
        num_layers_attr = "n_layer"  # 视 Qwen config 而定
    else:
        layers = model.model.layers
        cfg = model.config
        num_layers_attr = "num_hidden_layers"

    # 构造新的 ModuleList，只保留未被移除的层
    for idx in sorted(remove_indices)[::-1]:
        del layers[idx]

    model.model.layers = layers
    gc.collect()

    # 更新 config 中记录的层数
    new_n = len(layers)
    if hasattr(cfg, num_layers_attr):
        setattr(cfg, num_layers_attr, new_n)
    else:
        # 如果找不到常见字段，打印提醒，并可选择设置 config.layers 或其它自定义
        print(f"Warning: config has no attribute {num_layers_attr}; config keys: {list(cfg.__dict__.keys())}")

    return layers


if __name__ == '__main__':
    args = make_parser().parse_args()

    tokenizer, model = load_model_tokenizer(args)

    pruner_dic = {
        "short": ShortGptPruner,
        "laco": LacoPruner,
        "cl": CLPruner,
        "reme": ReplaceMePruner,
        "spsrci": SPSRCIPruner, "spsrp": SPSRPlusPruner, "diag": DiagCalPruner, "spsrs": SPSRPlusPruner,
        "stream": StreamLinePruner,
        "patch": LinearPatchPruner, "patchp": LinearPatchPlusPruner,
        "sleb": SLEBPruner,
        "block": BlockPruner,
        "taylor": TaylorPruner, "taylori": TaylorIterPruner,
    }
    random.seed(args.seed)
    if "Qwen3" or "Qwen2.5" or "Qwen1.5" in args.base_model:
        args.base_model = args.base_model.replace("Qwen", "LlamaQwen")
    if args.pruner is not None:
        args.remove_list = [int(i) for i in args.remove_list]

        start_time = time.time()
        pruner = pruner_dic[args.pruner](model, tokenizer)
        pruner.prune(args)
        prune_time = time.time() - start_time
        print("overall time cost: %.5f sec" % prune_time)
        del pruner
    elif len(args.remove_list) > 0:
        remove_layers_and_update_config(model, args.remove_list, model_type=args.base_model)
    torch.cuda.empty_cache()

    if args.tasks is not None:
        from utils import eval_ppl

        eval_tasks = pattern_match(args.tasks.split(","), tasks.ALL_TASKS)
        print(f"Selected Tasks: {eval_tasks}")

        if "wikitext" in eval_tasks:
            eval_tasks.pop(eval_tasks.index("wikitext"))
            eval_ppl(model, tokenizer, "wikitext")
        if "ptb" in eval_tasks:
            eval_tasks.pop(eval_tasks.index("ptb"))
            eval_ppl(model, tokenizer, "ptb")
        if "c4" in eval_tasks:
            eval_tasks.pop(eval_tasks.index("c4"))
            eval_ppl(model, tokenizer, "c4")
        if len(eval_tasks) > 0:
            eval_zero(args, model, tokenizer, eval_tasks)
