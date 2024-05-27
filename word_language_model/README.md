# 使用 RNN 和 Transformer 进行词级别语言建模

这个示例在语言建模任务上训练一个多层 RNN（Elman、GRU 或 LSTM）或 Transformer。默认情况下，训练脚本使用提供的 Wikitext-2 数据集。训练好的模型可以使用 generate 脚本生成新文本。

```bash
python main.py --cuda --epochs 6           # 在 Wikitext-2 数据集上使用 CUDA 训练一个 LSTM。
python main.py --cuda --epochs 6 --tied    # 在 Wikitext-2 数据集上使用 CUDA 训练一个权重共享的 LSTM。
python main.py --cuda --tied               # 在 Wikitext-2 数据集上使用 CUDA 训练一个权重共享的 LSTM，训练 40 个 epoch。
python main.py --cuda --epochs 6 --model Transformer --lr 5
                                           # 在 Wikitext-2 数据集上使用 CUDA 训练一个 Transformer 模型。

python generate.py                         # 从默认的模型检查点生成样本。
```

模型使用 `nn.RNN` 模块（及其姊妹模块 `nn.GRU` 和 `nn.LSTM`）或 Transformer 模块（`nn.TransformerEncoder` 和 `nn.TransformerEncoderLayer`），如果在安装了 cuDNN 的 CUDA 上运行，这些模块将自动使用 cuDNN 后端。

在训练期间，如果收到键盘中断（Ctrl-C），训练将停止，当前模型将对测试数据集进行评估。

`main.py` 脚本接受以下参数：

```bash
可选参数:
  -h, --help            显示此帮助信息并退出
  --data DATA           数据语料库的位置
  --model MODEL         网络类型（RNN_TANH、RNN_RELU、LSTM、GRU、Transformer）
  --emsize EMSIZE       词嵌入的大小
  --nhid NHID           每层的隐藏单元数
  --nlayers NLAYERS     层数
  --lr LR               初始学习率
  --clip CLIP           梯度剪裁
  --epochs EPOCHS       最大 epoch 数
  --batch_size N        批量大小
  --bptt BPTT           序列长度
  --dropout DROPOUT     应用于层的 dropout（0 表示无 dropout）
  --tied                绑定词嵌入和 softmax 权重
  --seed SEED           随机种子
  --cuda                使用 CUDA
  --mps                 在 macOS 上启用 GPU
  --log-interval N      报告间隔
  --save SAVE           保存最终模型的路径
  --onnx-export ONNX_EXPORT
                        导出最终模型为 onnx 格式的路径
  --nhead NHEAD         Transformer 模型编码器/解码器中的头数
  --dry-run             验证代码和模型
```

使用这些参数，可以测试各种模型。
例如，以下参数生成较慢但效果更好的模型：

```bash
python main.py --cuda --emsize 650 --nhid 650 --dropout 0.5 --epochs 40
python main.py --cuda --emsize 650 --nhid 650 --dropout 0.5 --epochs 40 --tied
python main.py --cuda --emsize 1500 --nhid 1500 --dropout 0.65 --epochs 40
python main.py --cuda --emsize 1500 --nhid 1500 --dropout 0.65 --epochs 40 --tied
```