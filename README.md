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

## 模型结构

最终版本模型配置如下：

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

## 项目演进

### v1：字符级教学原型
最初版本使用固定中文短文本作为训练语料，采用字符级 tokenizer（`stoi / itos`）和较小模型配置，目的是理解 Transformer 语言模型从编码到训练再到生成的完整闭环。

特点：
- 字符级编码
- 小语料
- 单文件训练脚本
- 基础自回归生成

### v2：SentencePiece 过渡版
第二版引入 SentencePiece BPE，将语料逐行编码，支持更大规模文本训练；同时生成阶段加入了 temperature、top-k、top-p 和 repetition penalty。

特点：
- SentencePiece tokenizer
- 更大的模型规模（768 hidden, 8 layers）
- 逐行编码语料
- AMP 混合精度
- 更可控的生成策略

### v3：工程化最终版
第三版将数据预处理为 `train.bin / val.bin`，训练时通过 `np.memmap` 读取；同时加入验证集评估、best checkpoint、weight decay、梯度裁剪和权重绑定，使训练过程更稳定、更适合作为完整项目展示。

特点：
- `train.bin / val.bin + memmap`
- train/val loss 评估
- best/last checkpoint
- weight decay
- gradient clipping
- weight tying
- 更完整的训练与推理流程

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
├─ src/
│  ├─ model.py
│  ├─ build_bin.py
│  ├─ train.py
│  ├─ sample.py
│  ├─ evaluate.py
│  └─ plot_loss.py
├─ outputs/
│  ├─ checkpoints/
│  ├─ logs/
│  ├─ figures/
│  └─ samples/
└─ docs/
   ├─ project_notes.md
   └─ experiment_summary.md
