# Mini GPT from Scratch (PyTorch, Chinese Corpus, SentencePiece)

一个基于 PyTorch 从零实现的 Mini GPT 项目，包含中文语料预处理、SentencePiece 分词、Transformer 语言模型训练、验证集评估、checkpoint 保存与文本生成。

## 项目简介

这个项目的目标不是调用现成大模型 API，而是从零实现一个可训练的自回归 Transformer 语言模型，并逐步完成以下工作：

- 从最初的字符级原型模型开始
- 升级到基于 SentencePiece 的子词级 tokenizer
- 优化训练数据管线，从全量内存读取改为二进制文件 + memmap
- 引入验证集评估、best checkpoint、梯度裁剪和 AMP 混合精度
- 支持 temperature、top-k、top-p、repetition penalty 等生成控制策略

## 项目亮点

- 从零实现多头自注意力、前馈网络、位置编码和自回归生成流程
- 使用 SentencePiece 构建子词级 tokenizer，支持中文大语料训练
- 将训练数据预处理为 `train.bin / val.bin`，并使用 memmap 提升大语料训练时的数据读取效率
- 引入 train/val loss 评估、best checkpoint 保存、梯度裁剪和混合精度训练
- 对比字符级原型、过渡版和工程版三种实现方式，完成一次完整的项目演进

## 仓库结构

```text
mini-gpt-from-scratch/
├─ README.md
├─ requirements.txt
├─ .gitignore
├─ configs/
│  └─ base_config.json
├─ data/
│  ├─ raw/
│  │  └─ wiki_cleaned_train.txt
│  ├─ processed/
│  │  ├─ train.bin
│  │  ├─ val.bin
│  │  └─ data_meta.json
│  └─ tokenizer/
│     ├─ spm_bpe_12k.model
│     └─ spm_bpe_12k.vocab
├─ docs/
│  ├─ project_notes.md
│  └─ experiment_summary.md
├─ legacy/
│  ├─ v1_char_level_prototype.py
│  └─ v2_sentencepiece_transition.py
├─ outputs/
│  ├─ checkpoints/
│  ├─ figures/
│  ├─ logs/
│  └─ samples/
└─ src/
   ├─ model.py
   ├─ train_tokenizer.py
   ├─ build_bin.py
   ├─ train.py
   ├─ sample.py
   ├─ evaluate.py
   └─ plot_loss.py
```

## 环境依赖

建议环境：

- Python 3.10+
- PyTorch
- sentencepiece
- numpy
- matplotlib
- tqdm
- pandas

安装依赖：

```bash
pip install -r requirements.txt
```

## 使用流程

### 1. 准备原始文本

把你的训练语料放到：

```text
data/raw/wiki_cleaned_train.txt
```

### 2. 训练 SentencePiece tokenizer

```bash
python src/train_tokenizer.py \
  --input data/raw/wiki_cleaned_train.txt \
  --model_prefix data/tokenizer/spm_bpe_12k \
  --vocab_size 12000
```

会生成：

- `data/tokenizer/spm_bpe_12k.model`
- `data/tokenizer/spm_bpe_12k.vocab`

### 3. 生成 `train.bin / val.bin`

```bash
python src/build_bin.py
```

### 4. 开始训练

```bash
python src/train.py
```

训练过程会：

- 定期评估 train/val loss
- 保存 `best_model.pth` 和 `last_model.pth`
- 将日志写到 `outputs/logs/metrics.csv`
- 定期生成样例写入 `outputs/samples/`

### 5. 生成文本

```bash
python src/sample.py \
  --ckpt outputs/checkpoints/best_model.pth \
  --prompt "问题：为什么天空是蓝色的？\n回答：" \
  --max_new_tokens 120 \
  --temperature 0.8 \
  --top_k 50 \
  --top_p 0.9 \
  --repetition_penalty 1.15
```

### 6. 单独评估模型

```bash
python src/evaluate.py
```

### 7. 画 loss 曲线

```bash
python src/plot_loss.py
```

## 最终版本模型配置

默认配置见 `configs/base_config.json`：

- Tokenizer: SentencePiece BPE
- Embedding dimension: 768
- Number of heads: 8
- Number of layers: 8
- Context length (`block_size`): 384
- Dropout: 0.1
- Optimizer: AdamW
- Training tricks:
  - weight decay
  - gradient clipping
  - AMP mixed precision
  - weight tying
  - validation loss evaluation
  - best / last checkpoint saving

## 结果展示建议

你后面训练完以后，把这些补到 README 里：

1. 一张 loss 曲线图：`outputs/figures/loss_curve.png`
2. 一个版本对比表（v1 / v2 / v3）
3. 三组 sample 输出
4. 最终 val loss / perplexity

## 简历描述参考

**Mini GPT 语言模型训练与优化项目（PyTorch）**  
基于 PyTorch 从零实现自回归 Transformer 语言模型，完成中文语料预处理、SentencePiece 分词、模型训练、验证与文本生成流程；将训练数据管线从全量内存读取优化为二进制文件 + memmap 方式，并引入验证集评估、best checkpoint、梯度裁剪、weight decay 与混合精度训练，提升训练稳定性与项目工程完整性。

## 说明

这个仓库已经尽量补成了完整工程。你还需要自己补的通常只有三类东西：

- 真实数据文件
- 真实 tokenizer 文件
- 真实训练结果（ckpt、loss 图、sample 输出）
