r"""
  _____        __                                             _ _   _                    _   _
 |_   _|      / _|                                 /\        (_) | | |                  | | (_)
   | |  _ __ | |_ ___ _ __ ___ _ __   ___ ___     /  \   _ __ _| |_| |__  _ __ ___   ___| |_ _  ___
   | | | '_ \|  _/ _ \ '__/ _ \ '_ \ / __/ _ \   / /\ \ | '__| | __| '_ \| '_ ` _ \ / _ \ __| |/ __|
  _| |_| | | | ||  __/ | |  __/ | | | (_|  __/  / ____ \| |  | | |_| | | | | | | | |  __/ |_| | (__
 |_____|_| |_|_| \___|_|  \___|_| |_|\___\___| /_/    \_\_|  |_|\__|_| |_|_| |_| |_|\___|\__|_|\___|

A simple inference calculator for GPT-based models
"""

import math

from collections import OrderedDict
from typing import Optional

import streamlit as st

st.title("LLM Inference Calculator 🧮")

"""
A simple inference calculator for GPT-based models.
"""

encoding = {
    'FP64': 8,
    'FP32': 4,
    'TF32': 4,
    'FP16': 2,
    'BFLOAT16': 2,
    'INT8': 1,
    'INT4': 0.5,
}

st.sidebar.selectbox(
    label='Encoding Format',
    options=sorted(encoding.keys()),
    key='encoding',
)

st.sidebar.number_input(
    label=r'Vocabulary Size ($n_\texttt{vocab}$)',
    key='n_vocab',
    value=50272,
    min_value=1,
)

st.sidebar.number_input(
    label=r'Model Dimension ($d_\texttt{model}$)',
    key='d_model',
    value=12288,
    min_value=1,
)

st.sidebar.number_input(
    label=r'Number of Layers ($n_\texttt{layer}$)',
    key='n_layer',
    value=96,
    min_value=1,
)

st.sidebar.number_input(
    label='Batch Size ($B$)',
    key='n_batch',
    value=512,
    min_value=1,
)

st.sidebar.number_input(
    label=r'Input Sequence Length ($s_i$)',
    key='s_i',
    value=512,
    min_value=0,
)

st.sidebar.number_input(
    label=r'Output Sequence Length ($s_o$)',
    key='s_o',
    value=32,
    min_value=0,
)

st.sidebar.number_input(
    label=r'Compute Performance ($A_\texttt{flops}$, TFLOPs)',
    key='a_flops',
    value=312,
    min_value=1,
)

st.sidebar.number_input(
    label=r'Memory Bandwidth ($A_\texttt{membw}$, GB/s)',
    key='a_membw',
    value=1500,
    min_value=1,
)

st.sidebar.number_input(
    label=r'GPU Count ($N_\texttt{GPU}$)',
    key='gpu_count',
    value=1,
    min_value=1,
)

r"""
### Parameters 🔢

> **Note:** Sub-leading terms such as nonlinearities, biases and layer normalization are omitted
"""

@st.cache_data
def full_params(
    n_layer: int,
    d_model: int,
    n_vocab: int,
    d_ff: Optional[int] = None,
    d_attn: Optional[int] = None,
    n_ctx: Optional[int] = None) -> OrderedDict:
    """
    Returns the full distribution of parameters in a GPT-style model.

    Parameters:
    n_layer (int): number of layers
    d_model (int): dimension of the residual stream
    n_vocab (int): vocabulary size
    n_ctx (Optional[int]): number of tokens in the input context
    d_ff (Optional[int]): dimension of the intermediate feed-forward layers
    d_attn (Optional[int]): dimension of the attention output
    n_ctx (Optional[int]): number of tokens in the input context

    Notes:
    - The residual stream is simply the sum of the output of all the previous layers and the original embeddings
    - Most Transformer-based architectures assume that d_attn = n_heads x d_head = d_model and d_ff = 4 x d_model
    - Sub-leading terms such as nonlinearities, biases and layer normalization are omitted

    References:
    - (Kaplan et al., 2020) Scaling laws for neural language models - https://arxiv.org/abs/2001.08361
    """

    params = OrderedDict()

    # Embeddings
    if not n_ctx:
        n_ctx = 0
    params['Embedding/Token'] = n_vocab * d_model
    params['Embedding/Position'] = d_model * n_ctx
    params['Embedding'] = params['Embedding/Position'] + params['Embedding/Token']

    # Attention
    if not d_attn:
        d_attn = d_model
    params['Attention/QKV'] = n_layer * d_model * 3*d_attn
    params['Attention/Project'] = n_layer * d_attn * d_model
    params['Attention'] = params['Attention/QKV'] + params['Attention/Project']

    # MLP/FF/Linear
    if not d_ff:
        d_ff = 4*d_model
    params['MLP/FF'] = n_layer * d_model * d_ff
    params['MLP/Project'] = n_layer * d_ff * d_model
    params['MLP'] = params['MLP/FF'] + params['MLP/Project']

    # Transformer
    params['Transformer'] = params['Attention'] + params['MLP']

    # Dense
    params['Dense'] = 0  # Uses the weights from the embedding layer via parameter sharing

    # Total
    params['Total'] = params['Embedding'] + params['Transformer'] + params['Dense']

    return params

def pp_params(p):
    """
    Pretty-prints the parameter distribution

    Parameters:
    p (OrderedDict): parameter distribution
    """
    p_total = p['Total']
    out = """
<table>
  <tr>
    <th>Name</th>
    <th># Params</th>
    <th>Ratio (%)</th>
  </tr>"""
    for k,v in p.items():
        out += f"""
  <tr>
    <td>
       <b {"style='color:red'" if k == 'Total' else None}>{k:20s}</b>
    </td>
    <td>
       <div style='float: right; text-align: right'>
       <code>{v:10d}</code>
       </div>
    </td>
    <td>
       <div style='float: right; text-align: right'>
       <code>{v/p_total*100:10.1f}</code>
       </div>
    </td>
  </tr>
"""
    out += "</table><br>"
    st.markdown(out, unsafe_allow_html=True)

p = full_params(
    st.session_state.n_layer,
    st.session_state.d_model,
    st.session_state.n_vocab,
)

pp_params(p)

r"""
### Memory 💾
"""

# Model Size (# of Parameters)

@st.cache_data
def params(
    n_layer: int,
    d_model: int) -> int:
    """
    Returns an O(d^2_model) estimate of the total number of parameters in a GPT-style model

    Parameters:
    n_layer (int): number of layers
    d_model (int): dimension of the residual stream

    Notes:
    - We are assuming that d_model >> n_ctx (# of tokens in the input context)

    References:
    - (Kaplan et al., 2020) Scaling laws for neural language models - https://arxiv.org/abs/2001.08361
    """
    return 12 * n_layer * d_model**2

# KV Cache (GB)

@st.cache_data
def kv_cache(
    n_bytes: int,
    n_batch: int,
    n_layer: int,
    d_model: int,
    s_i: int,
    s_o: int) -> int:
    """
    Computes the peak size of the KV cache

    Parameters:
    n_bytes (int): number of bytes per parameter
    n_batch (int): batch size
    n_layer (int): number of layers
    d_model (int): dimension of the residual stream
    """
    return n_bytes * n_batch * n_layer * 2 * d_model * (s_i + s_o)


model_size_in_B = params(st.session_state.n_layer, st.session_state.d_model)/1e9

model_size_in_GB = model_size_in_B * encoding[st.session_state.encoding]

kv_c = kv_cache(
    encoding[st.session_state.encoding],
    st.session_state.n_batch,
    st.session_state.n_layer,
    st.session_state.d_model,
    st.session_state.s_i,
    st.session_state.s_o,
)/1e9

with st.expander("Learn more..."):
    r"""
    The memory footprint of LLM inference comes mainly from **model weights** and **KV cache**.

    Assuming $d_\texttt{model} \gg n_\text{ctx}$ and keeping only $\mathcal{O}(d^2_\texttt{model})$ terms, we get a fairly good [estimate of the number of parameters](https://kipp.ly/transformer-param-count/):

    > In most Transformer-based architectures, $n_\texttt{heads} \cdot d_\texttt{head} = d_\texttt{model}$ and $d_\texttt{ff} = 4 \cdot d_\texttt{model}$

    $$N \approx n_\texttt{layer}(\underbrace{3 \cdot d^2_\texttt{model}}_{W_q, W_k, W_v} + \underbrace{d^2_\texttt{model}}_{W_o} + \underbrace{2 \cdot 4 \cdot d^2_\texttt{model}}_{\text{MLP}}) = 12 \cdot n_\texttt{layer} \cdot d^2_\texttt{model}$$

    which we can then use to compute the model size in GB $n_\texttt{bytes} \cdot N$.

    As for the KV cache, this requires holding the KV values for *every* layer, which is equal to storing (Sheng *et al.*, 2023)

    $$\text{Cache}_\texttt{KV} = n_\texttt{bytes} \cdot B \cdot n_\texttt{layer} \cdot 2 \cdot d_\texttt{model} \cdot (s_i + s_o)$$
    """

st.markdown(f"""
<table>
  <tr>
    <td>
       <b>Model Size</b>
    </td>
    <td>
       <code>{model_size_in_B:,.1f} B / {model_size_in_GB:,.1f} GB</code>
    </td>
  </tr>
  <tr>
    <td>
       <b>KV Cache</b>
    </td>
    <td>
       <code>{kv_c:,.1f} GB</code>
    </td>
  </tr>
  <tr>
    <td>
       <b style="color:red">Total</b>
    </td>
    <td>
       <code>{model_size_in_GB + kv_c:,.1f} GB</code>
    </td>
  </tr>
<table>
<br>""", unsafe_allow_html=True)

r"""
### Latency 🐌
"""

with st.expander("Learn more"):
    r"""
    Since reading parameters into on-chip memory happens *asynchronously* in all modern tensor programming libraries, the overall model latency will be the maximum between the compute and memory latencies.

    $$\text{latency}_\texttt{compute} = (2 \cdot N \cdot B)/(N_\texttt{GPU} \cdot A_\texttt{flops})$$

    $$\text{latency}_\texttt{memory} = (2 \cdot N \cdot n_\texttt{bytes})/(N_\texttt{GPU} \cdot A_\texttt{mem\_bw})$$

    $$\text{latency} = \max\left\{\text{latency}_\texttt{compute}, \text{latency}_\texttt{memory}\right\}$$
    """

@st.cache_data
def latency_memory(
    model_size: int,
    n_bytes: int,
    a_membw: int,
    gpu_count: int) -> float:
    """
    Calculates the memory latency

    Parameters:
    model_size (int): number of parameters
    n_bytes (int): number of bytes representing each parameter
    a_membw (int): accelerator memory bandwidth
    gpu_count (int): number of GPUs
    """
    return 2 * model_size * n_bytes * 1e9 * 1e3 / (gpu_count * a_membw * 1e9)

@st.cache_data
def latency_compute(
    model_size: int,
    n_batch: int,
    a_flops: int,
    gpu_count: int) -> float:
    """
    Calculates the compute latency

    Parameters:
    model_size (int): number of parameters
    n_bytes (int): number of bytes representing each parameter
    a_flops (int): FLOPs of the accelerator
    gpu_count (int): number of GPUs
    """
    return 2 * model_size * 1e9 * n_batch * 1e3 / (gpu_count * a_flops * 1e12)

@st.cache_data
def latency(
    mem_lat: float,
    cmp_lat: float) -> float:
    """
    Calculates the overall latency

    Parameters:
    mem_lat (float): memory latency
    cmp_lat (float): compute latency
    """
    return max(mem_lat, cmp_lat)

mem_lat = latency_memory(
    model_size_in_B,
    encoding[st.session_state.encoding],
    st.session_state.a_membw,
    st.session_state.gpu_count
)

cmp_lat = latency_compute(
    model_size_in_B,
    st.session_state.n_batch,
    st.session_state.a_flops,
    st.session_state.gpu_count
)

lat = latency(mem_lat, cmp_lat)

st.markdown(f"""
**Note:** The model is *memory-bound* as long as the batch size $B$ is less than `{math.ceil(1e3 * st.session_state.a_flops/st.session_state.a_membw)}`

<table>
  <tr>
    <td>
       <b>Memory Latency</b>
    </td>
    <td>
       <code>{mem_lat:,.1f} ms</code>
    </td>
  </tr>
  <tr>
    <td>
       <b>Compute Latency</b>
    </td>
    <td>
       <code>{cmp_lat:,.1f} ms</code>
    </td>
  </tr>
  <tr>
    <td>
       <b>Latency</b>
    </td>
    <td>
       <code>{lat:,.1f} ms</code>
    </td>
  </tr>
   <tr>
    <td>
       <b>Generation Throughput</b>
    </td>
    <td>
       <code>{1/lat * 1e3:,.1f} tokens/s</code>
    </td>
  </tr>
<table>
<br>""", unsafe_allow_html=True)

"""
### Want to learn more?

#### Articles 📚

* (Ivanov *et al.*, 2020) [Data Movement is All You Need: A Case Study for Optimizing Transformers](https://arxiv.org/abs/2007.00072)
* (Kaplan *et al.*, 2020) [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361)
* (Ouyang *et al.*, 2023) [Understanding the Performance of Transformer Inference](https://dspace.mit.edu/bitstream/handle/1721.1/151543/ouyang-aouyang-meng-eecs-2023-thesis.pdf?sequence=1&isAllowed=y)
* (Pope *et al.*, 2022) [Efficiently Scaling Transformer Inference](https://arxiv.org/abs/2211.05102)
* (Sheng *et al.*, 2023) [FlexGen: High-Throughput Generative Inference of Large Language Models with a Single GPU](https://arxiv.org/abs/2303.06865)
* (Tay *et al.*, 2020) [Efficient Transformers: A Survey](https://arxiv.org/abs/2009.06732)

#### Blogs ✍️

* [LLM Parameter Counting](https://kipp.ly/transformer-param-count/) by Kipply
* [Transformer Inference Arithmetic](https://kipp.ly/transformer-inference-arithmetic/) by Kipply
* [Making Deep Learning Go Brrrr From First Principles](https://horace.io/brrr_intro.html) by Horace He
* [Large Transformer Model Inference Optimization](https://lilianweng.github.io/posts/2023-01-10-inference-optimization/) by Lil'Log
* [How is LLaMa.cpp possible?](https://finbarr.ca/how-is-llama-cpp-possible/) by Finbarr Timbers
* [Decoding Transformers on Edge Devices](https://www.axelera.ai/decoding-transformers-on-edge-devices/) by Axelera
* [GPT in 60 Lines of NumPy](https://jaykmody.com/blog/gpt-from-scratch/) by Jay Mody
* [Speeding up the GPT - KV cache](https://www.dipkumar.dev/becoming-the-unbeatable/posts/gpt-kvcache/) by dipkumar
* [Dissecting Batching Effects in GPT Inference](https://le.qun.ch/en/blog/2023/05/13/transformer-batching/) by Lequn Chen
* [Accelerated Inference for Large Transformer Models Using NVIDIA Triton Inference Server](https://developer.nvidia.com/blog/accelerated-inference-for-large-transformer-models-using-nvidia-fastertransformer-and-nvidia-triton-inference-server/) by Nvidia
"""
