#  SPSR

Official PyTorch implementation of [SPSR: Achieving Superior Large Language Model Layer Pruning Performance by Super-fast Recovery](https://openreview.net/forum?id=zBswcgygZh).


## Results

## Installation 

Step 1: Create a new conda environment:
```
conda create -n spsr python=3.9
conda activate spsr
```
Step 2: Install relevant packages
```
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install transformers==4.51.0 datasets==2.18.0 wandb sentencepiece
pip install accelerate==1.9.0
```


## Usage

We provide a quick overview of the arguments:  
- `--base_model`: The identifier for the LLaMA model on the Hugging Face model hub.
- `--pruner`: We have implemented several layer pruning methods.
- `--final_s`: Denotes the percentage of weights to be pruned.
- `--tasks`: Eval ppl and zero shot tasks.
- `--num_examples`: Calibration dataset size.

### Comparison of Replacement Component Recovery Methods

Below is an example command for pruning LLaMA3-8B with SPSR, to remove 8 layers.

```
python LlmPruner2/main.py \
--base_model meta-llama/Meta-Llama-3-8B \
-p spsrs -s 0.25 \
--num_examples 128 \
--tasks wikitext,ptb,c4,storycloze,rte,openbookqa,arc_easy,winogrande,arc_challenge,piqa,boolq,hellaswag \
--epochs 10 --lr 3e-2 \
--bf16
```    

Below is an example command for pruning LLaMA3-8B with ReplaceMe, to remove 8 layers.

```
python LlmPruner2/main.py \
--base_model meta-llama/Meta-Llama-3-8B \
-p reme -s 0.25 \
--num_examples 8000 \
--tasks wikitext,ptb,c4,storycloze,rte,openbookqa,arc_easy,winogrande,arc_challenge,piqa,boolq,hellaswag \
--epochs 10 --lr 1e-4 \
--dataset Open-Orca/SlimOrca \
--bf16
``` 

Below is an example command for pruning LLaMA3-8B with Linear-Patch, to remove 8 layers.

```
python LlmPruner2/main.py \
--base_model meta-llama/Meta-Llama-3-8B \
-p patch -s 0.25 \
--num_examples 128 \
--tasks wikitext,ptb,c4,storycloze,rte,openbookqa,arc_easy,winogrande,arc_challenge,piqa,boolq,hellaswag \
--epochs 1 --lr 1e-4 \
--dataset wikitext \
--train_size 2048\
--bf16
``` 

Below is an example command for pruning LLaMA3-8B with LLM-Streamline, to remove 8 layers.

```
python LlmPruner2/main.py \
--base_model meta-llama/Meta-Llama-3-8B \
-p stream -s 0.25 \
--num_examples 128 \
--tasks wikitext,ptb,c4,storycloze,rte,openbookqa,arc_easy,winogrande,arc_challenge,piqa,boolq,hellaswag \
--epochs 20 --lr 2e-4 \
--origin \
--bf16
``` 

### Combination with Non-consecutive Layer Pruning

Below is an example command for pruning LLaMA3-8B with SLEB, to remove 8 layers. Removing layers of SLEB can be obtained.

```
python LlmPruner2/main.py \
--base_model meta-llama/Meta-Llama-3-8B \
-p sleb -s 0.25 \
--num_examples 128 \
--tasks wikitext,ptb,c4,storycloze,rte,openbookqa,arc_easy,winogrande,arc_challenge,piqa,boolq,hellaswag \
--fp16
``` 

Below is an example command for pruning LLaMA3-8B with SPSR combined with SLEB, to remove 8 layers.

```
python LlmPruner2/main.py \
--base_model meta-llama/Meta-Llama-3-8B \
-p spsrp -s 0.25 \
--num_examples 128 \
--tasks wikitext,ptb,c4,storycloze,rte,openbookqa,arc_easy,winogrande,arc_challenge,piqa,boolq,hellaswag \
--epochs 10 --lr 3e-2 \
--remove_list 9, 10, 11, 12, 19, 23, 25, 26
--bf16
```    

Below is an example command for pruning LLaMA3-8B with Linear-Patch combined with SLEB, to remove 8 layers.

```
python LlmPruner2/main.py \
--base_model meta-llama/Meta-Llama-3-8B \
-p patchp -s 0.25 \
--num_examples 128 \
--tasks wikitext,ptb,c4,storycloze,rte,openbookqa,arc_easy,winogrande,arc_challenge,piqa,boolq,hellaswag \
--remove_list 9, 10, 11, 12, 19, 23, 25, 26 \
--dataset wikitext \
--bf16
```    

### Acknowledgement
Zero-shot tasks is evaluated on the [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) repositories.


### Citation
```
@inproceedings{yang-etal-2026-spsr,
    title = "{SPSR}: Achieving Superior Large Language Model Layer Pruning Performance by Super-fast Recovery",
    author = "Yang, Zhiguo  and
      Deng, Changjian  and
      Chen, Qinke  and
      Zhou, Zijing  and
      Cheng, Jian",
    editor = "Lee, Hung-Yi  and
      Dev, Sunipa  and
      Li, Jessy  and
      Elliott, Desmond",
    booktitle = "Findings of the {A}ssociation for {C}omputational {L}inguistics: {EMNLP} 2026",
    month = oct,
    year = "2026",
    address = "Budapest, Hungary",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2026.findings-emnlp.1057/",
    pages = "22378--22398",
    ISBN = "979-8-89176-462-0",
    abstract = "Layer pruning aims to accelerate the inference of Large Language Models on general hardware by removing less important Transformer layers. However, naively deleting layers typically degrades model performance. To reduce this degradation, prior work approximates removed layers with replacement components such as linear layers. These approaches often require large datasets and heavy replacement modules, which increase computational and memory costs. Their time complexity also restricts them to static, consecutive-layer pruning. In this work, we introduce a novel method, Superior layer Pruning performance by Super-fast Recovery (SPSR), which uses Pearson correlation coefficients to quantify a layer{'}s recovery ability and applies a super-fast fine-tuning procedure with a lightweight replacement component. SPSR readily extends to non-consecutive pruning and to dynamic pruning. Experimental results show that SPSR achieves performance competitive with state-of-the-art methods while substantially improving recovery efficiency."
}
```
