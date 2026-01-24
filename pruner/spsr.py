import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Subset, Dataset
from tqdm import tqdm

from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling

from datas import get_examples


class DiagCalPruner:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    @torch.no_grad()
    def select(self, args, dataloader):
        model, tokenizer = self.model, self.tokenizer
        if not args.fp16 and not args.bf16:
            model.half()

        before_pruning_parameters = sum(p.numel() for p in self.model.parameters())
        print("Before prune, #parameters: {}".format(before_pruning_parameters))

        use_cache = model.config.use_cache
        model.config.use_cache = False

        device = model.device

        if "Llama" in args.base_model or "llama" in args.base_model:
            layers = model.model.layers
        elif "opt" in args.base_model:
            layers = model.model.decoder.layers
        elif "Qwen" in args.base_model:
            layers = model.transformer.h
        else:
            layers = model.model.layers
        num_layers = len(layers)

        pruning_num = int(num_layers * args.final_s)
        # pruning_num = 1
        hidden_dim = model.config.hidden_size
        A_acc = [torch.zeros(hidden_dim, hidden_dim).to(device) for _ in range(num_layers)]
        B_acc = [torch.zeros(hidden_dim, hidden_dim).to(device) for _ in range(num_layers)]
        token_counts = [0] * (num_layers - pruning_num + 1)  # 各层对处理的token总数
        ratios = [0] * (num_layers - pruning_num + 1)  # 存储各层对角化程度
        for j in tqdm(range(len(dataloader)), desc="Collecting hidden states"):
            input_ids = dataloader[j].unsqueeze(0).to(device)
            hidden_states = model(input_ids, output_hidden_states=True).hidden_states
            for i in range(len(token_counts)):
                # 获取第i层和第i+pruning_num层的隐藏状态
                h1 = hidden_states[i][0].to(torch.float32)  # [seq_len, hidden_dim]
                h2 = hidden_states[i + pruning_num][0].to(torch.float32)  # [seq_len, hidden_dim]
                assert len(h2.shape) == 2

                # 中心化处理
                h1_centered = h1 - h1.mean(0, keepdim=True)
                h2_centered = h2 - h2.mean(0, keepdim=True)

                # 在线累加协方差矩阵
                A_acc[i] += h1_centered.T @ h1_centered  # [hidden_dim, hidden_dim]
                B_acc[i] += h1_centered.T @ h2_centered  # [hidden_dim, hidden_dim]
                token_counts[i] += h1.shape[0]  # 累加token数

        # 计算各层对角化程度
        lambda_val = 1e-6
        for i in tqdm(range(num_layers - pruning_num + 1), desc="Computing diagonal ratios"):
            if token_counts[i] == 0: continue

            # 正则化协方差矩阵
            A = A_acc[i] / token_counts[i] + lambda_val * torch.eye(hidden_dim, device=device)
            B = B_acc[i] / token_counts[i]

            # 岭回归求解线性映射 w [hidden_dim, hidden_dim]
            try:
                # Cholesky分解 (高效稳定)
                L = torch.linalg.cholesky(A)
                w = torch.cholesky_solve(B, L)
            except RuntimeError:
                # SVD回退 (应对病态矩阵)
                U, S, Vh = torch.linalg.svd(A, full_matrices=False)
                w = Vh.t() @ ((U.t() @ B) / S.unsqueeze(1))

            # 计算对角化程度 = (对角元素平方和) / (全部元素平方和)
            diag_sq = w.diagonal().pow(2).sum()
            full_sq = w.pow(2).sum()
            ratios[i] = (diag_sq / full_sq).item()

        print(ratios)

        model.config.use_cache = use_cache
        if not args.fp16 and not args.bf16:
            model.float()

    def prune(self, args):
        model, tokenizer = self.model, self.tokenizer

        device = model.device
        print("loading calibdation data")
        dataloader = get_examples(args.dataset, tokenizer, n_samples=args.num_examples, seq_len=2048)
        print("dataset loading complete")

        self.select(args, dataloader)
        torch.cuda.empty_cache()


class SPSRCIPruner:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    @torch.no_grad()
    def select(self, args, dataloader):
        model, tokenizer = self.model, self.tokenizer
        if not args.fp16 and not args.bf16:
            model.half()

        before_pruning_parameters = sum(p.numel() for p in self.model.parameters())
        print("Before prune, #parameters: {}".format(before_pruning_parameters))

        use_cache = model.config.use_cache
        model.config.use_cache = False

        device = model.device

        if "Llama" in args.base_model or "llama" in args.base_model:
            layers = model.model.layers
        elif "opt" in args.base_model:
            layers = model.model.decoder.layers
        elif "Qwen" in args.base_model:
            layers = model.transformer.h
        else:
            layers = model.model.layers
        n = len(layers)

        pruning_num = int(n * args.final_s)
        if len(args.remove_list) == 0:
            score = torch.zeros((n - pruning_num + 1,), device=device)
            for j in tqdm(range(len(dataloader)), desc="Collecting hidden states"):
                input_ids = dataloader[j].unsqueeze(0).to(device)
                hidden_states = model(input_ids, output_hidden_states=True).hidden_states
                for i in range(len(score)):
                    score[i] += (F.normalize(hidden_states[i][0], p=2, dim=1) *
                                 F.normalize(hidden_states[i + pruning_num][0], p=2, dim=1)).sum(dim=1).mean(dim=0)

            for i in range(len(score)):
                score[i] = 1 - score[i] / args.num_examples
            print(score)

            start_index = torch.argmin(score)
            pruning_layers = list(range(start_index, start_index + pruning_num))
            print(pruning_layers)
        else:
            start_index = args.remove_list[0]
            pruning_layers = list(range(start_index, start_index + pruning_num))
            print(pruning_layers)


        hidden_inputs = torch.zeros((len(dataloader), len(dataloader[0]), model.config.hidden_size), device=device,
                                    dtype=torch.bfloat16 if args.bf16 else torch.float16)
        out_norm = torch.zeros((model.config.hidden_size,), device=device, dtype=torch.float32)
        inp_norm = torch.zeros_like(out_norm)
        for j in tqdm(range(len(dataloader)), desc="Collecting hidden states"):
            input_ids = dataloader[j].unsqueeze(0).to(device)
            hidden_states = model(input_ids, output_hidden_states=True).hidden_states
            hidden_inputs[j] = hidden_states[start_index][0]
            inp_norm += torch.mean(hidden_states[start_index][0].to(dtype=torch.float32) ** 2, dim=0).to(
                device) / args.num_examples
            out_norm += torch.mean(hidden_states[start_index + pruning_num][0].to(dtype=torch.float32) ** 2,
                                   dim=0).to(device) / args.num_examples

        s = ((out_norm / inp_norm) ** 0.5)
        b = torch.zeros_like(s)

        if args.bf16:
            s = s.to(torch.bfloat16)
            b = b.to(torch.bfloat16)
        print(torch.mean(s.abs()), torch.mean(b.abs()))

        new_layers = nn.ModuleList(layers[:start_index] +
                                   [IdentityNormLike(s, b)] + layers[start_index + pruning_num:])
        if "Llama" in args.base_model or "llama" in args.base_model:
            model.model.layers = new_layers
        elif "opt" in args.base_model:
            model.model.decoder.layers = new_layers
        del layers[start_index:start_index + pruning_num]

        model.config.use_cache = use_cache
        if not args.fp16 and not args.bf16:
            model.float()

        return start_index, hidden_inputs

    def prune(self, args):
        model, tokenizer = self.model, self.tokenizer

        device = model.device
        print("loading calibdation data")
        dataloader = get_examples(args.dataset, tokenizer, n_samples=args.num_examples, seq_len=2048)
        print("dataset loading complete")

        target_idx, hidden_inputs = self.select(args, dataloader)
        torch.cuda.empty_cache()

        if "Llama" in args.base_model or "llama" in args.base_model:
            layers = model.model.layers
        elif "opt" in args.base_model:
            layers = model.model.decoder.layers
        elif "Qwen" in args.base_model:
            layers = model.transformer.h
        else:
            layers = model.model.layers

        # 冻结
        for param in model.named_parameters():
            param[1].requires_grad = False

        for layer in layers:
            if isinstance(layer, IdentityNormLike):
                layer.s = nn.Parameter(layer.s)
                layer.bias = nn.Parameter(layer.bias)

        dataloader = [(inps, outs) for inps, outs in zip(hidden_inputs.cpu(), dataloader.cpu())]
        total_size = len(dataloader)
        train_size = int(0.9 * total_size)
        train_dataset = Subset(dataloader, range(train_size))
        eval_dataset = Subset(dataloader, range(train_size, total_size))

        data_collator = DataCollator(tokenizer)

        output_dir = f'{args.output_dir}/{args.epochs}'
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=args.epochs,
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            warmup_steps=int(args.epochs * len(train_dataset) * 0.175),
            weight_decay=args.wd,
            logging_dir='./logs',
            eval_strategy='epoch',
            logging_steps=500,
            # lr_scheduler_type="cosine",
            # warmup_ratio=0.175,
            # lr_scheduler_kwargs={"num_cycles": 5},
            learning_rate=args.lr,
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

        for param in model.named_parameters():
            param[1].requires_grad = False
        torch.cuda.empty_cache()

        model.half()
        for layer in layers:
            if isinstance(layer, IdentityNormLike):
                if not isinstance(layer.bias, nn.Parameter):
                    layer.bias = nn.Parameter(layer.bias)
                print(torch.mean(layer.s.data.abs()), torch.mean(layer.bias.data.abs()))
                print(layer.bias)


class SPSRPearsonSinglePruner:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    @torch.no_grad()
    def select(self, args, dataloader):
        model, tokenizer = self.model, self.tokenizer
        if not args.fp16 and not args.bf16:
            model.half()

        before_pruning_parameters = sum(p.numel() for p in self.model.parameters())
        print("Before prune, #parameters: {}".format(before_pruning_parameters))

        use_cache = model.config.use_cache
        model.config.use_cache = False

        device = model.device

        if "Llama" in args.base_model or "llama" in args.base_model:
            layers = model.model.layers
        elif "opt" in args.base_model:
            layers = model.model.decoder.layers
        elif "Qwen" in args.base_model:
            layers = model.transformer.h
        else:
            layers = model.model.layers
        n = len(layers)

        pruning_num = 1
        convs = torch.zeros((n - pruning_num + 1, model.config.hidden_size), device=device)
        std1s = torch.zeros((n - pruning_num + 1, model.config.hidden_size), device=device)
        std2s = torch.zeros((n - pruning_num + 1, model.config.hidden_size), device=device)
        score = torch.zeros((n - pruning_num + 1,), device=device)
        for j in tqdm(range(len(dataloader)), desc="Collecting hidden states"):
            input_ids = dataloader[j].unsqueeze(0).to(device)
            hidden_states = model(input_ids, output_hidden_states=True).hidden_states
            for i in range(len(score)):
                # 获取第i层和第i+pruning_num层的隐藏状态
                h1 = hidden_states[i][0].to(torch.float32)  # [seq_len, hidden_dim]
                h2 = hidden_states[i + pruning_num][0].to(torch.float32)  # [seq_len, hidden_dim]
                assert len(h2.shape) == 2

                # 中心化处理
                h1_centered = h1 - h1.mean(0, keepdim=True)
                h2_centered = h2 - h2.mean(0, keepdim=True)

                # 计算皮尔逊相关系数
                cov = (h1_centered * h2_centered).sum(0)  # 协方差 [hidden_dim]
                std1 = torch.norm(h1_centered, p=2, dim=0)  # 标准差 [hidden_dim]
                std2 = torch.norm(h2_centered, p=2, dim=0)  # 标准差 [hidden_dim]

                convs[i] += cov
                std1s[i] += std1 ** 2
                std2s[i] += std2 ** 2

        corr = convs / ((std1s * std2s) ** 0.5 + 1e-8)
        score = (1 - corr ** 2).max(dim=1)[0]

        start_index = torch.argmin(score)
        pscore = (score - score[start_index]).abs()
        pscore[start_index] = torch.inf
        if (pscore.min()) < 1e-3:
            score = ((1 - corr ** 2).min(dim=1)[0] + (1 - corr ** 2).max(dim=1)[0]) / 2
            start_index = torch.argmin(score)
        print(score)

        score[0] = score[1] = score[-2] = score[-1] = torch.inf  # not remove

        sort_index = torch.argsort(score)
        pruning_layers = sort_index[:int(n * args.final_s)].tolist()
        print(pruning_layers)

        model.config.use_cache = use_cache
        if not args.fp16 and not args.bf16:
            model.float()

        return pruning_layers

    def prune(self, args, dataloader):
        pruning_layers = self.select(args, dataloader)
        torch.cuda.empty_cache()

        return pruning_layers


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
        loss = self.loss_fct(logits.reshape(-1, logits.size(-1)).to(labels.device), labels.reshape(-1))

        return (loss, logits) if return_outputs else loss


class SPSRPlusPruner:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    @torch.no_grad()
    def select(self, args, dataloader, remove_list=None, delect_layers=True):
        model, tokenizer = self.model, self.tokenizer
        if not args.fp16 and not args.bf16:
            model.half()

        before_pruning_parameters = sum(p.numel() for p in self.model.parameters())
        print("Before prune, #parameters: {}".format(before_pruning_parameters))

        use_cache = model.config.use_cache
        model.config.use_cache = False

        device = model.device

        if "Llama" in args.base_model or "llama" in args.base_model:
            layers = model.model.layers
        elif "opt" in args.base_model:
            layers = model.model.decoder.layers
        elif "Qwen" in args.base_model:
            layers = model.transformer.h
        else:
            layers = model.model.layers
        n = len(layers)

        if remove_list is None:
            pruning_num = int(n * args.final_s)
            assert len(args.remove_list) == pruning_num, "removing layers not enough"
            pruning_lst = sorted(args.remove_list)
        else:
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

        norm_ratio = {}
        hidden_inputs = torch.zeros((len(dataloader), len(dataloader[0]), model.config.hidden_size), device=device, dtype=torch.bfloat16 if args.bf16 else torch.float16)
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
                    device) / args.num_examples
                out_norms[idx_lst] += torch.mean(hidden_states[end_index][0].to(dtype=torch.float32) ** 2,
                                       dim=0).to(device) / args.num_examples

        for idx_lst in candidates:
            s = (out_norms[idx_lst] / inp_norms[idx_lst]) ** 0.5
            b = torch.zeros_like(s)

            if args.bf16:
                s = s.to(torch.bfloat16)
                b = b.to(torch.bfloat16)
            norm_ratio[idx_lst] = (s, b)

        model.config.use_cache = use_cache
        if not args.fp16 and not args.bf16:
            model.float()

        if not delect_layers:
            return norm_ratio

        new_layers = layers
        for idx_lst in candidates[::-1]:
            start_index, end_index = idx_lst
            s, b = norm_ratio[idx_lst]
            print(torch.mean(s.abs()), torch.mean(b.abs()))
            new_layers = nn.ModuleList(new_layers[:start_index] +
                                       [IdentityNormLike(s, b)] + new_layers[end_index:])
            del layers[start_index:end_index]
        # model.model.layers = new_layers
        if "Llama" in args.base_model or "llama" in args.base_model:
            model.model.layers = new_layers
        elif "opt" in args.base_model:
            model.model.decoder.layers = new_layers

        return target_idx, hidden_inputs

    def prune(self, args):
        model, tokenizer = self.model, self.tokenizer

        device = model.device
        print("loading calibdation data")
        dataloader = get_examples(args.dataset, tokenizer, n_samples=args.num_examples, seq_len=2048)
        print("dataset loading complete")

        if args.pruner == "spsrs":
            pruner = SPSRPearsonSinglePruner(model, tokenizer)
            args.remove_list = pruner.prune(args, dataloader)

        target_idx, hidden_inputs = self.select(args, dataloader)
        torch.cuda.empty_cache()

        if "Llama" in args.base_model or "llama" in args.base_model:
            layers = model.model.layers
        elif "opt" in args.base_model:
            layers = model.model.decoder.layers
        elif "Qwen" in args.base_model:
            layers = model.transformer.h
        else:
            layers = model.model.layers

        # 冻结
        for param in model.named_parameters():
            param[1].requires_grad = False

        for layer in layers:
            if isinstance(layer, IdentityNormLike):
                layer.s = nn.Parameter(layer.s)
                layer.bias = nn.Parameter(layer.bias)

        dataloader = [(inps, outs) for inps, outs in zip(hidden_inputs.cpu(), dataloader.cpu())]
        total_size = len(dataloader)
        train_size = int(0.9 * total_size)
        train_dataset = Subset(dataloader, range(train_size))
        eval_dataset = Subset(dataloader, range(train_size, total_size))

        data_collator = DataCollator(tokenizer)

        output_dir = f'{args.output_dir}/{args.epochs}'
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=args.epochs,
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            warmup_steps=int(args.epochs * len(train_dataset) * 0.175),
            weight_decay=args.wd,
            logging_dir='./logs',
            eval_strategy='epoch',
            logging_steps=500,
            # lr_scheduler_type="cosine",
            # warmup_ratio=0.175,
            # lr_scheduler_kwargs={"num_cycles": 5},
            learning_rate=args.lr,
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
        print(f"max memory_allocated  {torch.cuda.max_memory_allocated(device) / 1024 ** 2}")

        for param in model.named_parameters():
            param[1].requires_grad = False
        torch.cuda.empty_cache()

        model.half()
        for layer in layers:
            if isinstance(layer, IdentityNormLike):
                if not isinstance(layer.bias, nn.Parameter):
                    layer.bias = nn.Parameter(layer.bias)
                print(torch.mean(layer.s.data.abs()), torch.mean(layer.bias.data.abs()))
                print(layer.bias)


class IdentityNormLike(nn.Module):
    def __init__(self, s, bias, relu=False):
        super().__init__()

        self.s = s
        self.bias = bias
        self.relu = relu

    def forward(self, hidden_states, *args, **kwargs):
        use_cache = kwargs["use_cache"] if "use_cache" in kwargs else False
        output_attentions = kwargs["output_attentions"] if "output_attentions" in kwargs else False
        past_key_value = kwargs["past_key_value"] if "past_key_value" in kwargs else None

        outputs = (
            (hidden_states * self.s.to(device=hidden_states.device) + self.bias.to(device=hidden_states.device)),)

        if self.relu:
            outputs = (F.relu(outputs[0]),)

        if output_attentions:
            outputs += (None,)

        if use_cache:
            outputs += (past_key_value,)
        return outputs


class DataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, examples):
        labels = torch.cat([example[1].unsqueeze(0) for example in examples], dim=0)
        input_ids = torch.cat([example[0].unsqueeze(0) for example in examples], dim=0)
        output_dict = dict(labels=labels, input_ids=input_ids)
        return output_dict