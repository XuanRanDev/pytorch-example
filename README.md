# PyTorch 示例

![运行示例](https://github.com/pytorch/examples/workflows/Run%20Examples/badge.svg)

https://pytorch.org/examples/

`pytorch/examples` 是一个展示如何使用 [PyTorch](https://github.com/pytorch/pytorch) 的示例库。目标是提供经过精心挑选的、简短的、无依赖或少依赖的高质量示例，这些示例彼此之间有显著的区别，可以在您的现有工作中进行模仿。

- 教程请见: https://github.com/pytorch/tutorials
- 更改 pytorch.org 的内容请见: https://github.com/pytorch/pytorch.github.io
- 通用模型库请见: https://pytorch.org/hub/ 或 https://huggingface.co/models
- 运行 PyTorch 生产环境的相关配方请见: https://github.com/facebookresearch/recipes
- 一般的问答和支持请见: https://discuss.pytorch.org/

## 可用的模型

- [使用卷积神经网络进行图像分类（MNIST）](./mnist/README.md)
- [使用 RNN 和 Transformer 的词级语言建模](./word_language_model/README.md)
- [使用流行网络训练 Imagenet 分类器](./imagenet/README.md)
- [生成对抗网络（DCGAN）](./dcgan/README.md)
- [变分自编码器](./vae/README.md)
- [使用高效的子像素卷积神经网络进行超分辨率](./super_resolution/README.md)
- [在多个进程上使用 Hogwild 训练共享的卷积神经网络（MNIST）](mnist_hogwild)
- [在 OpenAI Gym 中使用演员-评论家算法训练平衡 CartPole](./reinforcement_learning/README.md)
- [使用 GloVe 向量、LSTM 和 torchtext 进行自然语言推理（SNLI）](snli)
- [时间序列预测——使用 LSTM 学习正弦波](./time_sequence_prediction/README.md)
- [在图像上实现神经风格迁移算法](./fast_neural_style/README.md)
- [在 OpenAI Gym 上使用演员-评论家和 REINFORCE 算法进行强化学习](./reinforcement_learning/README.md)
- [使用 fx 进行 PyTorch 模块转换](./fx/README.md)
- 使用[分布式数据并行](./distributed/ddp/README.md)和[RPC](./distributed/rpc)的分布式 PyTorch 示例
- [多个展示 C++ 前端的示例](cpp)
- [使用前向-前向进行图像分类](./mnist_forward_forward/README.md)
- [使用 Transformer 进行语言翻译](./language_translation/README.md)

此外，还有一些托管在它们自己仓库中的优秀示例：

- [使用注意力机制的序列到序列 RNN 进行神经机器翻译（OpenNMT）](https://github.com/OpenNMT/OpenNMT-py)

## 贡献

如果您想贡献自己的示例或修复一个错误，请务必查看 [CONTRIBUTING.md](CONTRIBUTING.md)。