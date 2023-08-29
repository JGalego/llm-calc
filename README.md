# LLM Calculator 🧮

Find out how much compute power and storage you need to train and host your model.

<img src="https://media1.giphy.com/media/v1.Y2lkPTc5MGI3NjExaXptNWh3bm54NDZwMDVidXNua2lidGxiZGl5c3VmeWllbmE5NGdueSZlcD12MV9naWZzX3NlYXJjaCZjdD1n/3owzW5c1tPq63MPmWk/200.gif"/>

## References

### Articles 📚

* (Brown *et al.*, 2020) [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165)
* (Hoffmann *et al.*, 2022) [Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556)
* (Ivanov *et al.*, 2020) [Data Movement is All You Need: A Case Study for Optimizing Transformers](https://arxiv.org/abs/2007.00072)
* (Kaplan *et al.*, 2020) [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361)
* (Pope *et al.*, 2022) [Efficiently Scaling Transformer Inference](https://arxiv.org/abs/2211.05102)
* (Sheng *et al.*, 2023) [FlexGen: High-Throughput Generative Inference of Large Language Models with a Single GPU](https://arxiv.org/abs/2303.06865)
* (Shoeybi *et al.*, 2019) [Megratron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism](https://arxiv.org/abs/1909.08053)
* (Tay *et al.*, 2020) [Efficient Transformers: A Survey](https://arxiv.org/abs/2009.06732)

### Blogs ✍️

* [The FLOPS Calculus of Language Model Training](https://medium.com/@dzmitrybahdanau/the-flops-calculus-of-language-model-training-3b19c1f025e4) by Dzmitry Bahdanau
* [LLM Parameter Counting](https://kipp.ly/transformer-param-count/) by Kipply
* [Transformer Inference Arithmetic](https://kipp.ly/transformer-inference-arithmetic/) by Kipply
* [Transformer Math 101](https://blog.eleuther.ai/transformer-math/) by EleutherAI
* [New Scaling Laws for Large Language Models](https://www.lesswrong.com/posts/midXmMb2Xg37F2Kgn/new-scaling-laws-for-large-language-models) by LessWrong
* [How is LLaMa.cpp possible?](https://finbarr.ca/how-is-llama-cpp-possible/) by Finbarr Timbers
* [Decoding Transformers on Edge Devices](https://www.axelera.ai/decoding-transformers-on-edge-devices/) by Axelera
* [Large Transformer Model Inference Optimization](https://lilianweng.github.io/posts/2023-01-10-inference-optimization/) by Lil'Log
* [GPT in 60 Lines of NumPy](https://jaykmody.com/blog/gpt-from-scratch/) by Jay Mody
* [Speeding up the GPT - KV cache](https://www.dipkumar.dev/becoming-the-unbeatable/posts/gpt-kvcache/) by dipkumar
* [Dissecting Batching Effects in GPT Inference](https://le.qun.ch/en/blog/2023/05/13/transformer-batching/) by Lequn Chen
* [Accelerated Inference for Large Transformer Models Using NVIDIA Triton Inference Server](https://developer.nvidia.com/blog/accelerated-inference-for-large-transformer-models-using-nvidia-fastertransformer-and-nvidia-triton-inference-server/) by Nvidia
* [AWS ML Infrastructure](https://aws.amazon.com/machine-learning/infrastructure/)

### Videos 🎥

* [Let's build GPT: from scratch, in code, spelled out](https://www.youtube.com/watch?v=kCc8FmEb1nY) by Andrej Karpathy
	- **Code:** [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT) - The simplest, fastest repository for training/finetuning medium-sized GPTs
	- Check out the [Transformer Sizing](https://github.com/karpathy/nanoGPT/blob/eba36e84649f3c6d840a93092cb779a260544d08/transformer_sizing.ipynb) and [Scaling Laws](https://github.com/karpathy/nanoGPT/blob/eba36e84649f3c6d840a93092cb779a260544d08/scaling_laws.ipynb) notebooks
* [How a Transformer works at inference vs training time](https://www.youtube.com/watch?v=IGu7ivuy1Ag) by Niels Rogge
* [The KV Cache: Memory Usage in Transformers](https://www.youtube.com/watch?v=80bIUggRJf4) by Efficient NLP