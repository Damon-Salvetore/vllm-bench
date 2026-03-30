# SlideSparse Accuracy Evaluation — Patched Pruning Code

Code backed up from `rebuttal-main` container (`bcacdwk/vllmbench:universal`) on 2026-03-28.

## Environment

- PyTorch 2.9.0+cu129
- CUDA 12.9
- transformers 4.57.3
- accelerate 1.12.0
- lm-eval 0.4.11
- datasets 4.8.4
- bitsandbytes 0.45.0 (for INT8 evaluation)

## Wanda Patches (6 changes)

### 1. Sparsity type whitelist removed (`main.py`)
```diff
- parser.add_argument("--sparsity_type", type=str, choices=["unstructured", "4:8", "2:4"])
+ parser.add_argument("--sparsity_type", type=str)
```
Allows arbitrary N:M patterns (6:8, 8:10, etc.) instead of only 2:4 and 4:8.

### 2. Assert removed + prune_n/prune_m override (`main.py`)
```diff
- if args.sparsity_type != "unstructured":
-     assert args.sparsity_ratio == 0.5, "sparsity ratio must be 0.5 for structured N:M sparsity"
+ if args.sparsity_type and args.sparsity_type != "unstructured":
      prune_n, prune_m = map(int, args.sparsity_type.split(":"))
+     if args.prune_n > 0:
+         prune_n = args.prune_n
+     if args.prune_m > 0:
+         prune_m = args.prune_m
+     args.sparsity_ratio = prune_n / prune_m
```
Added `--prune_n` and `--prune_m` args for explicit override. Auto-computes sparsity_ratio.

### 3. Seqlen capped to 2048 (`main.py`)
```diff
- model.seqlen = model.config.max_position_embeddings
+ model.seqlen = min(model.config.max_position_embeddings, 2048)
```
Prevents OOM on models with large context windows (Qwen2.5 has 131072).

### 4. Catcher `__getattr__` + position_embeddings (`lib/prune.py`)
```diff
+ def __getattr__(self, name):
+     try:
+         return super().__getattr__(name)
+     except AttributeError:
+         return getattr(self.module, name)
```
Fixes compatibility with newer transformers that access submodule attributes through the Catcher wrapper. Also captures `position_embeddings` kwarg needed by Qwen2.5 rotary embedding.

### 5. Calibration data: c4 → wikitext2 (`lib/prune.py`)
```diff
- dataloader, _ = get_loaders("c4", ...)
+ dataloader, _ = get_loaders("wikitext2", ...)
```
c4 dataset loading was unreliable; wikitext2 is faster and consistent with SparseGPT.

### 6. Early model save (`main.py`)
```diff
+ if args.save_model:
+     model.save_pretrained(args.save_model)
+     tokenizer.save_pretrained(args.save_model)
```
Saves model immediately after pruning, before eval. Prevents losing pruned model if eval OOMs.

## SparseGPT Patches (3 changes)

### 1. Blocksize alignment for n:m pruning (`sparsegpt.py`)
```diff
+ if prunem > 0 and blocksize % prunem != 0:
+     blocksize = blocksize - (blocksize % prunem)
+     if blocksize == 0:
+         blocksize = prunem
```
Fixes `RuntimeError: selected index k out of range` when blocksize (default 128) is not divisible by prunem (e.g. 6, 8, 10).

### 2. Boundary-safe topk in fasterprune (`sparsegpt.py`)
```diff
- tmp = W1[:, i:(i + prunem)] ** 2 / ...
- mask1.scatter_(1, i + torch.topk(tmp, prunen, ...), True)
+ actual_m = min(prunem, count - i)
+ actual_n = min(prunen, actual_m)
+ if actual_n > 0:
+     tmp = W1[:, i:(i + actual_m)] ** 2 / ...
+     mask1.scatter_(1, i + torch.topk(tmp, actual_n, ...), True)
```
Clamps the group size and topk k at block boundaries to prevent index-out-of-range.

### 3. Early save + PTB removal (`llama.py`)
```diff
+ if args.save:
+     model.save_pretrained(args.save)
+     tokenizer = AutoTokenizer.from_pretrained(args.model)
+     tokenizer.save_pretrained(args.save)

- for dataset in ["wikitext2", "ptb", "c4"]:
+ for dataset in ["wikitext2"]:
```
Saves model before eval (same reason as Wanda). Removes PTB (deprecated in newer datasets lib).

## Usage

### Pruning (Wanda)
```bash
cd wanda
# 6:8 sparsity on Qwen2.5-14B (prune 2 out of every 8)
CUDA_VISIBLE_DEVICES=0 python main.py \
  --model Qwen/Qwen2.5-14B \
  --prune_method wanda \
  --sparsity_type 6:8 \
  --prune_n 2 --prune_m 8 \
  --save /path/to/results/14b/wanda_6_8 \
  --save_model /path/to/results/14b/wanda_6_8/model
```

### Pruning (SparseGPT)
```bash
cd sparsegpt
# 6:8 sparsity on Qwen2.5-7B (prune 2 out of every 8)
CUDA_VISIBLE_DEVICES=0 python llama.py Qwen/Qwen2.5-7B wikitext2 \
  --prunen 2 --prunem 8 \
  --save /path/to/results/7b_sparsegpt/sgpt_6_8
```

### N:M Pattern Reference

**Important**: In our notation, N:M means "keep N non-zeros out of every M elements". The number of **pruned (zeroed) weights** per group = M - N = 2 for all patterns below. You must pass `--prune_n 2 --prune_m M` explicitly.

| Pattern | Keep:Total | Pruned | Sparsity | Wanda args | SparseGPT args |
|---------|-----------|--------|----------|------------|----------------|
| 2:4 | 2:4 | 2 | 50% | `--sparsity_type 2:4 --prune_n 2 --prune_m 4` | `--prunen 2 --prunem 4` |
| 4:6 | 4:6 | 2 | 33% | `--sparsity_type 4:6 --prune_n 2 --prune_m 6` | `--prunen 2 --prunem 6` |
| 6:8 | 6:8 | 2 | 25% | `--sparsity_type 6:8 --prune_n 2 --prune_m 8` | `--prunen 2 --prunem 8` |
| 8:10 | 8:10 | 2 | 20% | `--sparsity_type 8:10 --prune_n 2 --prune_m 10` | `--prunen 2 --prunem 10` |

**Key**: For all patterns, we prune 2 weights per group. `prunen`/`prune_n` = weights to zero.

### Batch Evaluation Script
```bash
# All 9 benchmarks in one command
./run_eval.sh /path/to/pruned_model /path/to/results 0 auto

# INT8 mode
INT8=1 ./run_eval.sh Qwen/Qwen2.5-14B /path/to/results/dense_int8 0 4
```

### Evaluation (lm-eval)
```bash
# BF16 pruned model
lm_eval --model hf \
  --model_args pretrained=/path/to/pruned_model,dtype=float16 \
  --tasks piqa,arc_easy,arc_challenge,hellaswag,winogrande,boolq,openbookqa \
  --batch_size 4 --output_path /path/to/results/commonsense

# INT8 quantized (bitsandbytes LLM.int8(), dynamic quantization)
lm_eval --model hf \
  --model_args pretrained=/path/to/pruned_model,load_in_8bit=True \
  --tasks piqa,arc_easy,arc_challenge,hellaswag,winogrande,boolq,openbookqa \
  --batch_size 4 --output_path /path/to/results/commonsense_int8

# MMLU (5-shot)
lm_eval --model hf \
  --model_args pretrained=/path/to/pruned_model,dtype=float16 \
  --tasks mmlu --num_fewshot 5 \
  --batch_size 4 --output_path /path/to/results/mmlu

# GSM8K (5-shot)
lm_eval --model hf \
  --model_args pretrained=/path/to/pruned_model,dtype=float16 \
  --tasks gsm8k --num_fewshot 5 \
  --batch_size 4 --output_path /path/to/results/gsm8k
```

### INT8 Evaluation Notes
- We use **bitsandbytes LLM.int8()** (dynamic mixed INT8+FP16 quantization at inference time)
- NOT W8A8 static quantization — no pre-quantized checkpoints needed
- Dense and sparse models use identical quantization method for fair comparison
- Add `load_in_8bit=True` to `--model_args` to enable INT8 mode

### Docker Environment
```bash
# Base image with all dependencies
docker run -d --name eval --gpus all --shm-size=32g \
  -v /path/to/data:/workspace \
  bcacdwk/vllmbench:universal tail -f /dev/null

# Install additional deps inside container
docker exec eval pip install lm-eval datasets bitsandbytes
```
