"""
  _____        __                                             _ _   _                    _   _      
 |_   _|      / _|                                 /\        (_) | | |                  | | (_)     
   | |  _ __ | |_ ___ _ __ ___ _ __   ___ ___     /  \   _ __ _| |_| |__  _ __ ___   ___| |_ _  ___ 
   | | | '_ \|  _/ _ \ '__/ _ \ '_ \ / __/ _ \   / /\ \ | '__| | __| '_ \| '_ ` _ \ / _ \ __| |/ __|
  _| |_| | | | ||  __/ | |  __/ | | | (_|  __/  / ____ \| |  | | |_| | | | | | | | |  __/ |_| | (__ 
 |_____|_| |_|_| \___|_|  \___|_| |_|\___\___| /_/    \_\_|  |_|\__|_| |_|_| |_| |_|\___|\__|_|\___|

A simple inference calculator for GPT-based models
"""

import math

import streamlit as st
import matplotlib.pyplot as plt

st.title("Inference Arithmetic 🔢")

"""
A simple inference calculator for GPT-based models
"""

encoding = {
    'FP64': 64,
    'FP32': 32,
    'TF32': 32,
    'FP16': 16,
    'BFLOAT16': 16,
    'INT8': 8,
    'INT4': 4,
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
    label=r'Number of Blocks ($n_\texttt{blocks}$)',
    key='n_blocks',
    value=96,
    min_value=1,
)

st.sidebar.number_input(
    label='Batch Size ($B$)',
    key='batch_size',
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
    label=r'Number of Accelerators ($N_\texttt{accel}$, GB/s)',
    key='n_accel',
    value=1,
    min_value=1,
)

r"""
### Memory

The memory footprint of LLM inference comes mainly from model weights and the KV cache.

Under the assumption that $n_\texttt{heads} \cdot d_\texttt{head} = d_\texttt{model}$, we can [estimate model size](https://kipp.ly/transformer-param-count/)

$$N = 12 \cdot n_\texttt{blocks} \cdot d^2_\texttt{model} + n_\texttt{vocab} \cdot d_\texttt{model}$$

The KV cache requires holding the KV values for every layer, which is equal to storing

$$\text{Cache}_\texttt{KV} = 2 \cdot n_\texttt{bytes} \cdot B \cdot n_\texttt{blocks} \cdot d_\texttt{model} \cdot (s_\texttt{input} + s_\texttt{output})$$
"""

@st.cache_data
def num_params(n_blocks, d_model, n_vocab):
    """
    Computes the number of parameters in a GPT-style model
    """
    return (12 * n_blocks * d_model**2 + n_vocab * d_model)/1e9

@st.cache_data
def kv_cache(n_bytes, batch_size, n_blocks, d_model, s_i, s_o):
    """
    Computes the peak size of the KV cache
    """
    return 2 * (n_bytes/8) * batch_size * n_blocks * d_model * (s_i + s_o) / 1e9


n_params = num_params(
    st.session_state.n_blocks,
    st.session_state.d_model,
    st.session_state.n_vocab
)
kv_c = kv_cache(
    encoding[st.session_state.encoding],
    st.session_state.batch_size,
    st.session_state.n_blocks,
    st.session_state.d_model,
    st.session_state.s_i,
    st.session_state.s_o
)

st.markdown(f"""
#### Outputs

<table>
  <tr>
    <td>
       <b>Model Size</b>
    </td>
    <td>
       <code>{n_params:,.1f} B / {n_params * encoding[st.session_state.encoding]/8:,.1f} GB</code>
    </td>
  </tr>
  <tr>
    <td>
       <b>KV Cache</b>
    </td>
    <td>
       <code>{kv_c:,.3f} GB</code>
    </td>
  </tr>
<table>
<br>""", unsafe_allow_html=True)

r"""
### Latency 🐌

Since reading parameters into on-chip memory happens *asynchronously* in all modern tensor programming libraries, the overall model latency will be the maximum between the compute and memory latencies.

$$\text{latency}_\texttt{compute} = (2 \cdot N \cdot B)/(N_\texttt{accel} \cdot A_\texttt{flops})$$

$$\text{latency}_\texttt{memory} = (2 \cdot N)/(N_\texttt{accel} \cdot A_\texttt{mem\_bw})$$

$$\text{latency} = \max\left\{\text{latency}_\texttt{compute}, \text{latency}_\texttt{memory}\right\}$$
"""

@st.cache_data
def latency_memory(model_size, n_bytes, a_membw, n_accel=1):
    """
    Calculates the memory latency
    """
    return 2 * model_size * (n_bytes/8) * 1e9 * 1e3 / (n_accel * a_membw * 1e9)

@st.cache_data
def latency_compute(model_size, batch_size, a_flops, n_accel=1):
    """
    Calculates the compute latency
    """
    return 2 * model_size * 1e9 * batch_size * 1e3 / (n_accel * a_flops * 1e12)

@st.cache_data
def latency(mem_lat, cmp_lat):
    """
    Calculates the overall latency
    """
    return max(mem_lat, cmp_lat)

mem_lat = latency_memory(
    n_params,
    encoding[st.session_state.encoding],
    st.session_state.a_membw,
    st.session_state.n_accel
)

cmp_lat = latency_compute(
    n_params,
    st.session_state.batch_size,
    st.session_state.a_flops,
    st.session_state.n_accel
)

lat = latency(mem_lat, cmp_lat)

st.markdown(f"""
#### Outputs

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
"""