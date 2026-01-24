import json

import fire
import warnings


import torch
from torch import nn
from torch.utils.data import Subset
from tqdm import tqdm
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
from transformers import (
    AutoTokenizer,
    LlamaForCausalLM,
    LlamaConfig,
    AutoConfig,
    AutoModelForCausalLM,
    BertConfig,
    BertModel,
    BertForSequenceClassification
)
from transformers.modeling_outputs import CausalLMOutputWithPast

from datas import get_examples

import csv
import ast
import re


class CustomBertForSequenceClassification(BertForSequenceClassification):
    _no_split_modules = ["bert"]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class IdentityNormLike(nn.Module):
    def __init__(self, s, bias):
        super().__init__()

        self.s = s
        self.bias = bias

    def forward(self, hidden_states, *args, **kwargs):
        use_cache = kwargs["use_cache"] if "use_cache" in kwargs else False
        output_attentions = kwargs["output_attentions"] if "output_attentions" in kwargs else False
        past_key_value = kwargs["past_key_value"] if "past_key_value" in kwargs else None
        outputs = (
            (hidden_states * self.s.to(device=hidden_states.device) + self.bias.to(device=hidden_states.device)),)

        if output_attentions:
            outputs += (None,)

        if use_cache:
            outputs += (past_key_value,)
        return outputs


class SPSRLossTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_fct = nn.CrossEntropyLoss()  # top-k label
        self.start_idx = -1
        self.position_ids = None
        print("direct loss")

    def compute_loss(self, model, inputs, num_items_in_batch=None, return_outputs=False):
        """
        We override the compute_loss method of the Trainer class
        to use our custom loss instead of the default
        cross entropy.
        """
        # res = model(**inputs)
        #
        # return (res.loss, res.logits) if return_outputs else res.loss

        labels = inputs.get("labels")[:, 1:]
        hidden_states = inputs.get("input_ids")
        layers = model.model.layers
        for decoder_layer in layers[self.start_idx:]:
            layer_outputs = decoder_layer(
                hidden_states,
                position_ids=self.position_ids,
            )

            hidden_states = layer_outputs[0]

        hidden_states = model.model.norm(hidden_states)

        logits = model.lm_head(hidden_states)[:, :-1, :].contiguous()
        # # print(logits)
        loss = self.loss_fct(logits.reshape(-1, logits.size(-1)).to(labels.device), labels.reshape(-1))

        return (loss, logits) if return_outputs else loss


def origin_cal(model, remove_list):
    pruning_lst = sorted(remove_list)
    pruning_num = len(pruning_lst)
    start = pruning_lst[0]
    candidates = []
    for i in range(1, pruning_num):  # merge
        if pruning_lst[i] - pruning_lst[i - 1] > 1:
            candidates.append((start, pruning_lst[i - 1] + 1))
            start = pruning_lst[i]
    candidates.append((start, pruning_lst[pruning_num - 1] + 1))
    print(candidates)

    new_layers = model.model.layers
    for idx_lst in candidates[::-1]:
        start_index, end_index = idx_lst
        new_layers = nn.ModuleList(new_layers[:start_index] + new_layers[end_index:])
    return new_layers


def spsr_cal(model, tokenizer, dataloader, remove_list):
    use_cache = model.config.use_cache
    model.config.use_cache = False

    pruning_lst = sorted(remove_list)
    pruning_num = len(pruning_lst)
    start = pruning_lst[0]
    candidates = []
    for i in range(1, pruning_num):  # merge
        if pruning_lst[i] - pruning_lst[i - 1] > 1:
            candidates.append((start, pruning_lst[i - 1] + 1))
            start = pruning_lst[i]
    candidates.append((start, pruning_lst[pruning_num - 1] + 1))
    print(candidates)
    target_idx = pruning_lst[0]

    device = model.device
    norm_ratio = {}
    num_examples = len(dataloader)
    hidden_inputs = torch.zeros((len(dataloader), len(dataloader[0]), model.config.hidden_size), device=device,
                                dtype=torch.bfloat16)
    out_norms = {i: torch.zeros((model.config.hidden_size,), device=device, dtype=torch.float32) for i in candidates}
    inp_norms = {i: torch.zeros_like(out_norms[i]) for i in candidates}
    target_idx = pruning_lst[0]
    for j in tqdm(range(len(dataloader)), desc="Collecting hidden states"):
        input_ids = dataloader[j].unsqueeze(0).to(device)
        outputs = model(input_ids, output_hidden_states=True)
        hidden_states = outputs.hidden_states
        hidden_inputs[j] = hidden_states[target_idx][0]

        for idx_lst in candidates:
            start_index, end_index = idx_lst
            inp_norms[idx_lst] += torch.mean(hidden_states[start_index][0].to(dtype=torch.float32) ** 2, dim=0).to(
                device) / num_examples
            out_norms[idx_lst] += torch.mean(hidden_states[end_index][0].to(dtype=torch.float32) ** 2,
                                             dim=0).to(device) / num_examples

    for idx_lst in candidates:
        s = (out_norms[idx_lst] / inp_norms[idx_lst]) ** 0.5
        b = torch.zeros_like(s)
        s = s.to(torch.bfloat16)
        b = b.to(torch.bfloat16)
        norm_ratio[idx_lst] = (s, b)


    model.config.use_cache = use_cache

    origin_layers = model.model.layers
    new_layers = model.model.layers
    for idx_lst in candidates[::-1]:
        start_index, end_index = idx_lst
        s, b = norm_ratio[idx_lst]
        new_layers = nn.ModuleList(new_layers[:start_index] +
                                   [IdentityNormLike(s, b)] + new_layers[end_index:])

    for layer in new_layers:
        if isinstance(layer, IdentityNormLike):
            layer.s = nn.Parameter(layer.s)
            layer.bias = nn.Parameter(layer.bias)
    model.model.layers = new_layers

    # training
    dataloader = [(inps, outs) for inps, outs in zip(hidden_inputs.cpu(), dataloader.cpu())]
    total_size = len(dataloader)
    train_size = int(0.9 * total_size)
    train_dataset = Subset(dataloader, range(train_size))
    eval_dataset = Subset(dataloader, range(train_size, total_size))

    data_collator = DataCollator(tokenizer)

    output_dir = '/media/data/yzg'
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=10,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        warmup_steps=int(10 * len(train_dataset) * 0.175),
        weight_decay=1e-4,
        logging_dir='./logs',
        eval_strategy='epoch',
        logging_steps=500,
        # lr_scheduler_type="cosine",
        # warmup_ratio=0.175,
        # lr_scheduler_kwargs={"num_cycles": 5},
        learning_rate=3e-2,
        save_strategy="no",
    )

    trainer = SPSRLossTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )
    trainer.position_ids = torch.arange(0, 2048).unsqueeze(0).to(device)
    trainer.start_idx = target_idx

    trainer.train()

    for layer in new_layers:
        if isinstance(layer, IdentityNormLike):
            layer.s = nn.Parameter(layer.s.data.to(torch.float16))
            layer.bias = nn.Parameter(layer.bias.data.to(torch.float16))

    # recover
    model.model.layers = origin_layers
    return new_layers


class DataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, examples):
        labels = torch.cat([example[1].unsqueeze(0) for example in examples], dim=0)
        input_ids = torch.cat([example[0].unsqueeze(0) for example in examples], dim=0)
        output_dict = dict(labels=labels, input_ids=input_ids)
        return output_dict


class AdaptiveSLEBForCausalLM(LlamaForCausalLM):
    def __init__(self, config, name=None, base_model=None, router_model=None, router_path="./model/v3/jy_diff_all/", origin=False):

        super().__init__(config)
        self.n_layers = config.num_hidden_layers

        self.tokenizer = AutoTokenizer.from_pretrained("/home/jim/nas/yzg/Llama-3-8b/base")
        self.bert_tokenizer = AutoTokenizer.from_pretrained(router_path)

        if base_model is not None:
            self.model = base_model
        else:
            print('no llm model loaded...')
        if router_model is not None:

            self.router = router_model
        else:
            print('no trained router loaded...')
        self.router.eval()

        if name is None:
            self.name = 'v23_adaptSLEB'
        self.seqlen = 2048
        self.input_set = {}
        # self.excluded_indices=[0,2,4,6,7]
        self.init_count()

        if origin:
            self.spsr_layers = self.prepare_origin()
        else:
            self.spsr_layers = self.prepare_spsr()

    def prepare_origin(self, num_labels=10):
        skip_layers = [None] * num_labels
        with open('pudding/llama_layer_list_6_advanced_tasks.csv', mode="r", newline="", encoding="utf-8") as f:
            # with open('codes/5_adaptive_cluster/clustered_layer_list_10.csv', mode="r", newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            for row in reader:
                row_set_str = row[0]
                row_idx_str = row[1]
                if row_set_str == "Set": continue
                my_tuple = ast.literal_eval(row_set_str)
                skip_layers[int(row_idx_str)] = list(my_tuple)

        spsr_layers = [None] * len(skip_layers)
        for i in range(len(spsr_layers)):
            skip_layer = skip_layers[i]
            spsr_layers[i] = origin_cal(self.model, skip_layer)
        return spsr_layers

    def prepare_spsr(self, num_labels=10):
        skip_layers = [None] * num_labels
        with open('pudding/llama_layer_list_6_advanced_tasks.csv', mode="r", newline="", encoding="utf-8") as f:
            # with open('codes/5_adaptive_cluster/clustered_layer_list_10.csv', mode="r", newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            for row in reader:
                row_set_str = row[0]
                row_idx_str = row[1]
                if row_set_str == "Set": continue
                my_tuple = ast.literal_eval(row_set_str)
                skip_layers[int(row_idx_str)] = list(my_tuple)

        print("loading calibdation data")
        dataloader = get_examples("c4", self.tokenizer, n_samples=128, seq_len=2048)
        print("dataset loading complete")
        spsr_layers = [None] * len(skip_layers)
        for i in range(len(spsr_layers)):
            print(i)
            skip_layer = skip_layers[i]
            spsr_layers[i] = spsr_cal(self.model, self.tokenizer, dataloader, skip_layer)
        return spsr_layers


    def get_skip_mask(self, router_logits):

        probabilities = router_logits  # shape: (1, 10)

        # excluded_indices = torch.tensor(self.excluded_indices)
        # probabilities[:, excluded_indices] = float('-inf')

        # print(probabilities)
        _, topk_indices = torch.topk(probabilities, 1, dim=-1, largest=True)
        predicted_label = topk_indices.item()  # (1,1) -> int
        # print(predicted_label)
        # if predicted_label in excluded_indices:
        #     print(probabilities)
        #     print(sdfsf)

        self.add_count(predicted_label)
        skip_layer = self.spsr_layers[predicted_label]
        return skip_layer

    def init_count(self):
        print('initial count!')
        self.count = [0] * 10
        self.total = 0

    def add_count(self, predicted_label):
        self.count[predicted_label] += 1
        self.total += 1

    def print_count(self):
        print('#' * 20)
        print('returning count!')
        print(self.count)
        print(self.total)
        print('#' * 20)

    def forward(self, input_ids=None, attention_mask=None, skip_layer=None, **kwargs):
        seq_len = input_ids.size(1)
        device = input_ids.device

        if attention_mask is None:
            attention_mask = torch.ones((1, seq_len), device=device)

        if skip_layer is None:
            with torch.no_grad():
                input_text = self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)[0]

                answer_index = input_text.find("Answer")

                if answer_index != -1:
                    question_input = input_text[:answer_index]
                else:
                    match = re.search(r":\s*([^\.]*\.)", input_text)
                    if match:
                        question_input = match.group(1).strip()
                    else:
                        words = input_text.split()
                        question_input = " ".join(words[:7])

                if question_input in self.input_set:
                    skip_layer = self.input_set[question_input]

                else:

                    self.input_prompt = question_input
                    bert_inputs = self.bert_tokenizer(
                        input_text,
                        return_tensors='pt',
                        padding=True,
                        truncation=True,
                        max_length=512
                    ).to(device)

                    router_outputs = self.router(
                        input_ids=bert_inputs['input_ids'],
                        attention_mask=bert_inputs['attention_mask']
                    )
                    router_logits = router_outputs.logits  # shape: (batch=1, num_labels=10)

                    skip_layer = self.get_skip_mask(router_logits)
                    self.input_set[question_input] = skip_layer
        else:
            skip_layer = skip_layer#.to(device)

        self.model.model.layers = skip_layer
        outputs = self.model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits

        return CausalLMOutputWithPast(
            loss=None,
            logits=logits,
            hidden_states=None,
            attentions=None,
            # cross_attentions=None
        )


def v23_adaptSLEB(model_name="meta-llama/Meta-Llama-3.1-8B", router_path="result/9_llama/router/10/2_onlylog/MSE/5", origin=False):
    print('loading model v23...')
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        # quantization_config=bnb_config,  # 上面本地模型的配置
        # device_map="cpu" if args.deepsparse else "auto",  # 使用GPU的编号
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    # 冻结
    for param in base_model.named_parameters():
        param[1].requires_grad = False
    base_model.name = model_name
    # base_model = block_replace(base_model)

    config = AutoConfig.from_pretrained(model_name)
    router_model = CustomBertForSequenceClassification.from_pretrained(router_path,
                                                                       device_map="auto", num_labels=10)
    # 冻结
    for param in router_model.named_parameters():
        param[1].requires_grad = False

    model = AdaptiveSLEBForCausalLM(
        config=config,
        base_model=base_model,
        router_model=router_model,
        name='v23_adaptSLEB',
        router_path=router_path,
        origin=origin
    )


    # model.to('cuda')
    model.rank = 0
    model.world_size = 1

    return model, tokenizer


def v23_eval(
        model_name: str = "NousResearch/Meta-Llama-3.1-8B", # 3.1-8B? NousResearch/Meta-Llama-3.1-8B
        router_path: str = "/media/ssd/yzg/LlmPruner2/pudding/result/router/10",
        device: int = 0,
        eval_zeroshot: bool = True,
        origin=False,
        tasks: list = ['arc_challenge', 'arc_easy', 'boolq', 'hellaswag', 'piqa', 'winogrande', 'openbookqa'],
):
    warnings.filterwarnings("ignore", category=UserWarning)

    model, tokenizer = v23_adaptSLEB(model_name, router_path, origin=origin)

    if model is not None:
        print(f"Loaded Model: {model.name if hasattr(model, 'name') else model_name}")
        model.eval()
        # model.to(f'cuda:{device}')
    else:
        raise ValueError("Model could not be loaded.")


    if eval_zeroshot:
        print(f"Starting Zero-shot tasks evaluation...")

        print(tasks)
        print('asdfsadfsadfasdfsadfsad')
        from lm_eval import evaluator
        with torch.no_grad():
            results = evaluator.simple_evaluate(
                model_type="hf-causal-experimental",
                model=(tokenizer, model),
                model_args="pretrained=facebook/opt-125m",
                tasks=tasks,
                num_fewshot=0,
                batch_size=None,
                device=None,
                no_cache=False,
                limit=None,
                description_dict=None,
                decontamination_ngrams_path=None,
                check_integrity=False,
            )

        if results is None:
            return

        dumped = json.dumps(results, indent=2)
        print(dumped)
        print(evaluator.make_table(results))

        task_names = tasks
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


if __name__ == "__main__":
    fire.Fire(v23_eval)