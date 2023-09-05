r"""
  _      _      __  __    _____      _            _       _
 | |    | |    |  \/  |  / ____|    | |          | |     | |
 | |    | |    | \  / | | |     __ _| | ___ _   _| | __ _| |_ ___  _ __
 | |    | |    | |\/| | | |    / _` | |/ __| | | | |/ _` | __/ _ \| '__|
 | |____| |____| |  | | | |___| (_| | | (__| |_| | | (_| | || (_) | |
 |______|______|_|  |_|  \_____\__,_|_|\___|\__,_|_|\__,_|\__\___/|_|

Learn how to size LLM workloads - find out how much compute power and storage you need to train your model!
"""

import math

import streamlit as st

st.title("LLM Pre-Training Calculator 🧮")
st.subheader("How much compute and storage do I need?")

"""
The purpose of this calculator is to size LLM pre-training workloads by predicting training duration, compute and storage costs. It works with **any** accelerator, provided you have accurate information on hourly price and peak performance, and **any** Transformer-based model whose compute requirements are well-approximated by the [FLOPs equation](https://medium.com/@dzmitrybahdanau/the-flops-calculus-of-language-model-training-3b19c1f025e4).
"""

# By default, we'll assume that the checkpoint size follows the BLOOM ratio
# 2.3TB / 176B ~ 13GB per billion parameters
# https://huggingface.co/bigscience/tr11-176B-logs
# https://huggingface.co/blog/bloom-megatron-deepspeed
BLOOM_RATIO = 2.3e3/176

# and that the training dataset contains approximately
# 350B / 1.6TB ~ 0.2 tokens-per-byte
# https://accubits.com/large-language-models-leaderboard/bloom/
TOKENS_PER_BYTE = 350e9/1.6e12

# Pick a currency
currencies = {
    'USD': '$',
    'EUR': '€'
}
st.sidebar.selectbox(
    label='Currency',
    options=currencies.keys(),
    format_func=lambda name: f"{name} ({currencies[name]})",
    key='currency'
)

# How much does it cost per hour to run a single GPU?
st.sidebar.number_input(
    label='GPU Price (per hour)',
    key='gpu_price_per_hour',
    value=4.09 if st.session_state.currency == 'USD' else 3.77,  # p4d OD pricing for us-east-1
    step=0.01,
    format='%.2f',
)

# How much does it cost to store 1GB for 1 month?
st.sidebar.number_input(
    label='Storage Cost (per GB-month)',
    key='sto_cost_per_GB_month',
    value=0.022 if st.session_state.currency == 'USD' else 0.020,  # S3 standard storage (50-500TB) pricing for us-east-1
    step=0.001,
    format='%.3f',
)

# How many tokens do we have for each byte in our dataset?
st.sidebar.number_input(
    label='Tokens-per-Byte Ratio',
    key='tokens_per_byte_ratio',
    value=TOKENS_PER_BYTE,
    step=0.001,
    format='%.2f',
)

# How much does the checkpoint size increase for each 1B parameters?
st.sidebar.number_input(
    label='Checkpoint Size Ratio',
    key='chkpt_size_ratio',
    value=BLOOM_RATIO,
    step=0.1,
    format='%.1f',
)

with st.expander("Disclaimer ⚠️"):
    """
    Please beware, this application is *highly* experimental! ⚡

    As with all experiments, we had to make a few assumptions that are only valid under *strict* conditions.

    If you don't trust the numbers, feel free to run your own experiments.

    Quoting from [Isaiah Berlin](https://plato.stanford.edu/entries/berlin/):

    > *"No amount of calculation can save us from painful choices and imperfect solutions"*
    """

"""
### Inputs ➡️
"""

st.number_input(
    label='Peak Theoretical Performance (TFLOP/s/GPU)',
    key='peak_perf_in_TFLOP_s_per_gpu',
    value=312,  # peak float16 FLOP throughput of A100
    min_value=0,
    # A100: https://www.nvidia.com/en-us/data-center/a100/
    # Trn1: https://aws.amazon.com/machine-learning/trainium/
    # H100: https://www.nvidia.com/en-us/data-center/h100/
)

st.number_input(
    label='GPU Utilization (%)',
    key='gpu_utilization',
    value=30,
    min_value=0,
    max_value=100,
    step=1,
    help=r'30% is a good default, +50% is a well optimized stack',
)

st.number_input(
    label='GPU Count',
    key='gpu_count',
    value=300,
    min_value=0,
    help=r'Usually between 5 and 20 GPUs per Billion parameters',
)

st.number_input(
    label='GPUs per Instance',
    key='gpus_per_instance',
    value=8,
    min_value=1,
)

st.number_input(
    label='Instance Failure Rate (per Instance-day)',
    key='inst_fail_rate_per_inst_day',
    min_value=0.01,
    max_value=1.0,
    step=0.01,
)

st.number_input(
    label='Failure Recovery Time (hours)',
    key='fail_rcv_t_in_hours',
    value=1.0,
    format='%.1f',
    help=r'This value varies depending on the checkpoint speed and failure recovery automation',
)

st.number_input(
    label='Model Size (# Parameters/Billion)',
    key='model_size_in_B',
    value=40,
)

st.number_input(
    label='Training Data Size (Gtokens)',
    key='data_size_in_Gtokens',
    value=1000,
)

st.number_input(
    label='Experiments Budget (GPU-hours)',
    key='expt_budget_in_gpu_hours',
    value=0.0,
    help=r'This is usually between 1-10% of the total budget'
)

st.number_input(
    label='Checkpoint Frequency (per day)',
    key='chkpt_freq_per_day',
    value=24,    # hourly
    min_value=1, # daily
)

"""
### Outputs 📋
"""

with st.expander("Learn more about the math 🔢"):
    r"""
    #### Units & Conventions

    ##### Compute

    When discussing compute requirements, we follow the convention of using [`FLOP` for quantity and `FLOP/s` for performance](https://blog.heim.xyz/flop-for-quantity-flop-s-for-performance/):

    * **Quantity:** `FLOP` is short for *floating-point operation*. FLOPs (lowercase *s*, no apostrophe) is just the plural of `FLOP`, as in "1 MAC = 2 FLOPs". If we are consistent, the [whole case against `FLOPs`](https://www.lesswrong.com/posts/XiKidK9kNvJHX9Yte/avoid-the-abbreviation-flops-use-flop-or-flop-s-instead) is nonsensical. Just to be on the safe side though and to avoid confusion, we will only use FLOPs in written text.

    * **Performance:** `FLOP/s` (*preferred*) and `FLOPS` (uppercase *s*) can be used interchangeably.

    Some papers will report compute values in `PetaFLOP/s-days`, often abbreviated as `PF-days`, see e.g. Kaplan *et al.* (2020). Simple dimensional analysis tells us that `[Performance] x [Time]` where `[Performance] = [Compute] x [Time]^-1` is just `[Compute]`.

    ##### Storage

    For the most part, we'll use `GB` (not to be confused with `GiB`).

    There are two main reasons for this:

    * It's a good compromise to talk about the size of our dataset and the (cumulative) checkpoint size without adding too many significant digits
    * Storage is usually billed by the gigabyte-month or `GB-month` (`# GBs x # months`, similar to the concept of kiloWatt-hour or kWh)

    ##### Time

    Depending on the context, we will work at different timescales:

    * Training takes a couple of `days`, sometimes even `months`.

    * Performance is usually measured by the `second`.

    * Failures usually last a couple of `hours`.

    #### GPUs vs GPU Instances

    Training large models requires massive amounts of compute.

    At present, GPUs are the most cost-effective hardware accelerators for the job.

    However, choosing the right one can be challenging cf. [Selecting Servers and GPUs](https://d2l.ai/chapter_appendix-tools-for-deep-learning/selecting-servers-gpus.html) from d2l.ai.

    The golden rule is to [choose the right **GPU instance**, not just the right GPU](https://towardsdatascience.com/choosing-the-right-gpu-for-deep-learning-on-aws-d69c157d8c86).

    Once you choose an instance, keeping in mind that you can choose multiple ones, the question is how many you'll need.

    Now, assuming that all instances have the same number of GPUs ($n_\texttt{GPU}$), we can easily calculate the number of GPU instances ($N_\texttt{inst}$) based on the total number of GPUs ($N_\texttt{GPU}$):

    $$~N_\texttt{inst} = \lceil N_\texttt{GPU} / n_\texttt{GPU} \rceil$$

    The `ceil`ing function ensures that the formula returns the right number of instances even when $n_\texttt{GPU}$ does **not** factor $N_\texttt{GPU}$.

    #### Estimating Storage

    While training, we'll have to hold on to our datasets as well as multiple snapshots of our model.

    These snapshots, commonly known as **checkpoints**, represent the model's *entire internal state* and can be used to resume training or to load a model that has already been trained.

    *Ceteris paribus*, the ratio $r_\texttt{chkpt}$ between the size of a single checkpoint ($s_\texttt{chkpt}$) and the size of our model (represented by the number of parameters $N$) will be approximately the same for models of similar size.

    We can use this fact to estimate the size of a single checkpoint (in `GB`):

    $$~s_\texttt{chkpt} = r_\texttt{chkpt} \cdot N$$

    As a default, we use the checkpoint size ratio for the [BLOOM-176B](https://huggingface.co/blog/bloom-megatron-deepspeed) model

    $$~r_\texttt{chkpt}(\text{BLOOM-176B}) = \underbrace{2.3\text{TB}}_\text{Checkpoint Size} / \underbrace{176\text{B}}_\text{Model Size} \approx 13.1 \text{GB/Billion parameters}$$

    Equivalently, we can convert our dataset size $D$ (usually reported in `Gtokens`) to `GB` based on available information for similar datasets:

    $$~D_\ast = r^{-1}_{\texttt{t2b}} \cdot D$$

    The tokens-per-byte ratio $r_{\texttt{t2b}}$ for [The BigScience ROOTS Corpus](https://arxiv.org/abs/2303.03915) dataset is approximately

    $$~r_\texttt{t2b}(\text{The BigScience ROOTS Corpus}) = \underbrace{350\text{B}}_\text{Token Count} / \underbrace{1.6\text{TB}}_\text{Dataset Size} \approx 0.22 \text{tokens/B}$$

    #### Estimating Compute

    ##### Throughput and GPU Utilization

    GPU datasheets usually advertise their *theoretical* throughput values $\tau_\texttt{peak}$.

    These are rarely if ever met in practice.

    One way to correct for this discrepancy is to factor in **GPU utilization** ($r_\texttt{GPU}$)

    $$~\tau_\texttt{actual} = r_\texttt{GPU} \cdot \tau_\texttt{peak}$$

    As a rule of thumb, a well-optimized stack should be somewhere around `50%`.

    ##### Training Requirements

    The amount of compute we'll need for training is tied to the number of parameters in our model and the number of elementary operations needed to compute them.

    In the regime where the **weight** FLOPs contribution dominates everything else, including layer renormalization, attention, residual connections, &c., we can use the [Transformer FLOPs equation](https://medium.com/@dzmitrybahdanau/the-flops-calculus-of-language-model-training-3b19c1f025e4) (Kaplan *et al.*, 2020; Brown *et al.*, 2020) to estimate compute requirements ($C$) cf. [The FLOPs Calculus of Language Model Training](https://medium.com/@dzmitrybahdanau/the-flops-calculus-of-language-model-training-3b19c1f025e4) for additional details.

    Evaluating a forward pass involves roughly

    $$C_\texttt{forward} \approx 2 \cdot N \cdot D$$

    add-multiply operations, where the factor of 2 comes from the multiply-accumulate operation used in matrix multiplication.

    Accounting for the backward pass (which is approximately 2x the compute of the forward pass)

    $$C_\texttt{backward} \approx 4 \cdot N \cdot D$$

    we finally get

    $$C = C_\texttt{forward} + C_\texttt{backward} \approx 6 \cdot N \cdot D$$

    While training, the typical bottleneck is GPU memory not compute. A popular method for exchanging memory for compute is to recompute activations for certain layers instead of storing them in GPU memory (Korthikanti *et al.*, 2022).

    The upper bound on this recomputation is a full additional forward pass ($2 \cdot N \cdot D$) so that

    $$C_\texttt{forward} \leq 4 \cdot N \cdot D$$

    #### Estimating Time

    ##### Training Time

    Once we have the compute requirements for training our model we can start estimating how long it will take to do it.

    We define the **theoretical training time** ($T_\texttt{theory}$) as the ratio between our training compute needs ($C$) and the *real* throughput calculated in the last section ($\tau_\texttt{actual}$):

    $$T_\texttt{theory} = C / \tau_\texttt{actual}$$

    Dividing by the GPU count, we get the **cluster theoretical training time** ($T^\texttt{c}_\texttt{theory}$):

    $$T^\texttt{c}_\texttt{theory} = T_\texttt{theory} / N_\texttt{GPU}$$

    ##### Accounting for Failures

    What if one of the instances in our cluster fails? How will it affect the duration of our training run?

    Well, for the most part, failures are unavoidable. As Werner Vogels (Amazon CTO) is fond of saying:

    > *"Everything fails, all the time"*

    Each time an instance fails, the cluster will have to work *extra time* to recover from downtime and recompute uncheckpointed work:

    $$T^\texttt{c}_\texttt{total} = T^\texttt{c}_\texttt{theory} + T_\texttt{downtime} + T_\texttt{recomp} = T^\texttt{c}_\texttt{theory} + F \cdot (t_\texttt{fail} + t_\texttt{recomp})$$

    where $F$ is the number of expected failures within our cluster, $t_\texttt{fail}$ is the failure recovery time, $t_\texttt{recomp} \approx 1 / (2 \cdot f_\texttt{chkpt})$ is the time needed to recompute after each failure event and $f_\texttt{chkpt}$ is the checkpoint frequency.

    $F$ will vary depending on the number of instances in our cluster $N_\texttt{inst}$, the total cluster uptime $T^\texttt{c}_\texttt{total}$ and, of course, the failure rate $f$

    $$F = f \cdot T^\texttt{c}_\texttt{total} \cdot N_\texttt{inst}$$

    We can use this estimate to compute the **expected GPU time in a failed state**

    $$F \cdot N_\texttt{GPU} \cdot t_\texttt{fail}$$

    and the **expected GPU time recomputing uncheckpointed work**

    $$F \cdot N_\texttt{GPU} \cdot t_\texttt{recomp}$$

    By combining everything, we get a formula for the *real* cluster training time

    $$T^\texttt{c}_\texttt{total} = T^\texttt{c}_\texttt{theory} / [1 - f \cdot N_\texttt{inst} \cdot (t_\texttt{fail} + t_\texttt{recomp})]$$

    #### References

    * (Brown *et al.*, 2020) [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165)
    * (Hoffmann *et al.*, 2022) [Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556)
    * (Kaplan *et al.*, 2020) [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361)
    * (Korthikanti *et al.*, 2022) [Reducing Activation Recomputation in Large Transformer Models](https://arxiv.org/abs/2205.05198)
    * (Shoeybi *et al.*, 2019) [Megratron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism](https://arxiv.org/abs/1909.08053)"""

# Instance Count

@st.cache_data
def instance_count(
    gpu_count: int,
    gpus_per_instance: int) -> int:
    """
    Returns the number of GPU instances necessary to fulfill a given GPU count

    Parameters:
    gpu_count (int): total number of GPUs
    gpus_per_instance (int): number of GPUs per instance
    """
    return math.ceil(gpu_count / gpus_per_instance)

inst_count = instance_count(
    st.session_state.gpu_count,
    st.session_state.gpus_per_instance,
)

# Checkpoint Size (GB)

@st.cache_data
def checkpoint_size(
    model_size_in_B: int,
    chkpt_size_ratio: float) -> float:
    """
    Estimates checkpoint size in GB based on the model size (# parameters)

    Parameters:
    model_size_in_B (int): number of parameters measured by the Billions
    chkpt_size_ratio (float): checkpoint size ratio reported by a similar model
    """
    return chkpt_size_ratio * model_size_in_B

chkpt_size_in_GB = checkpoint_size(
    st.session_state.model_size_in_B,
    st.session_state.chkpt_size_ratio,
)

# Dataset Size (GB)

@st.cache_data
def data_size(
    data_size_in_Gtokens: int,
    tokens_per_byte_ratio: float) -> float:
    """
    Estimates training data size in GB from the number of tokens

    Parameters:
    data_size_in_Gtokens (int): training dataset size in Gtokens
    tokens_per_byte_ratio (float): tokens-per-byte ratio reported by a similar dataset
    """
    return data_size_in_Gtokens / tokens_per_byte_ratio

data_size_in_GB = data_size(
    st.session_state.data_size_in_Gtokens,
    st.session_state.tokens_per_byte_ratio,
)

# Actual Performance (TFLOP/s per GPU)

@st.cache_data
def actual_performance(
    peak_perf_in_TFLOP_s_per_gpu: int,
    gpu_utilization: int) -> float:
    """
    Returns actual TFLOP/s per GPU based on peak performance and GPU utilization

    Parameters:
    peak_perf_in_TFLOP_s_per_gpu (int): peak theoretical performance measured in TFLOP/s
    gpu_utilization (int): GPU utilization in %
    """
    return peak_perf_in_TFLOP_s_per_gpu * gpu_utilization / 100

actual_perf_in_TFLOP_s_per_gpu = actual_performance(
    st.session_state.peak_perf_in_TFLOP_s_per_gpu,
    st.session_state.gpu_utilization
)

# Training Compute Requirements (TFLOP)

@st.cache_data
def train_compute_requirements(
    model_size_in_B: int,
    data_size_in_Gtokens: int) -> int:
    """
    Returns training TFLOP requirements based on the size of the model and training data

    Parameters:
    model_size_in_B (int): number of parameters measured by the Billions
    data_size_in_Gtokens (int): training dataset size measured in Gtokens
    """
    return 6 * model_size_in_B * 1e9 * data_size_in_Gtokens * 1e9 / 1e12

train_comp_reqs_in_TFLOP = train_compute_requirements(
    st.session_state.model_size_in_B,
    st.session_state.data_size_in_Gtokens,
)

# Theoretical Training Time (GPU-hours)

@st.cache_data
def theoretical_training_time(
    train_comp_reqs_in_TFLOP: int,
    actual_perf_in_TFLOP_s_per_gpu: float) -> float:
    """
    Returns the *theoretical* training time in GPU-hours

    Parameters:
    train_comp_reqs_in_TFLOP (int): training compute requirements in TFLOP
    actual_perf_in_TFLOP_s_per_gpu: actual throughput for each GPU measured in TFLOP/s
    """
    return train_comp_reqs_in_TFLOP / actual_perf_in_TFLOP_s_per_gpu / 3600

t_train_time_in_gpu_hours = theoretical_training_time(
    train_comp_reqs_in_TFLOP,
    actual_perf_in_TFLOP_s_per_gpu
)

# Cluster Theoretical Training Time (days)

@st.cache_data
def cluster_theoretical_training_time(
    t_train_time_in_gpu_hours: float,
    gpu_count: int) -> float:
    """
    Returns the cluster *theoretical* training time in days

    Parameters:
    t_train_time_in_gpu_hours (float): theoretical training time in GPU-hours
    gpu_count (int): total number of GPUs
    """
    return t_train_time_in_gpu_hours / gpu_count / 24

cluster_t_train_time_in_days = cluster_theoretical_training_time(
    t_train_time_in_gpu_hours,
    st.session_state.gpu_count,
)

# Cluster Actual Training Time (days)

@st.cache_data
def cluster_actual_training_time(
    cluster_t_train_time_in_days: float,
    inst_count: int,
    inst_fail_rate_per_inst_day: float,
    fail_rcv_t_in_hours: float,
    chkpt_freq_per_day: int) -> float:
    """
    Returns the cluster *actual* training time in days

    Parameters:
    cluster_t_train_time_in_days (float): cluster theoretical training time in days
    inst_count (int): total number of GPU instances
    inst_fail_rate_per_inst_day (float): instance failure rate per Instance-day
    fail_rcv_t_in_hours (float): failure recovery time in hours
    chkpt_freq_per_day (int): checkpoint frequency per day
    """
    return cluster_t_train_time_in_days / (1 - (inst_fail_rate_per_inst_day / 100) * inst_count * (fail_rcv_t_in_hours/24 + 1/(2 * chkpt_freq_per_day)))

cluster_a_train_time_in_days = cluster_actual_training_time(
    cluster_t_train_time_in_days,
    inst_count,
    st.session_state.inst_fail_rate_per_inst_day,
    st.session_state.fail_rcv_t_in_hours,
    st.session_state.chkpt_freq_per_day,
)

# Expected Failures

@st.cache_data
def expected_failures(
    inst_fail_rate_per_inst_day: float,
    cluster_a_train_time_in_days: float,
    inst_count: int) -> float:
    """
    Returns the expected number of failures during the training cycle

    Parameters:
    inst_fail_rate_per_inst_day (float): instance failure rate per Instance-day
    cluster_a_train_time_in_days (float): cluster actual training time in days
    inst_count (int): total number of GPU instances
    """
    return (inst_fail_rate_per_inst_day / 100) * cluster_a_train_time_in_days * inst_count

exp_fail = expected_failures(
    st.session_state.inst_fail_rate_per_inst_day,
    cluster_a_train_time_in_days,
    inst_count,
)

# Expected GPU Time in Failed State (GPU-hours)

@st.cache_data
def expected_gpu_time_in_failed_state(
    exp_fail: float,
    gpu_count: int,
    fail_rcv_t_in_hours: float) -> float:
    """
    Returns the expected number of GPU-hours spent in a failed state

    Parameters:
    exp_fail (float): expected number of failure events
    gpu_count (int): total number of GPUs
    fail_rcv_t_in_hours (float): failure recovery time in hours
    """
    return exp_fail * gpu_count * fail_rcv_t_in_hours

exp_gpu_time_fail_in_gpu_hours = expected_gpu_time_in_failed_state(
    exp_fail,
    st.session_state.gpu_count,
    st.session_state.fail_rcv_t_in_hours,
)

# Expected GPU Time recomputing uncheckpointed work (GPU-hours)

@st.cache_data
def expected_gpu_time_recomputing(
    chkpt_freq_per_day: float,
    exp_fail: float,
    gpu_count: int) -> float:
    """
    Returns the expected number of GPU-hours to recover uncheckpointed work

    Parameters:
    chckpt_freq_per_day (float): checkpoint frequency per day
    exp_fail (float): expected number of failure events
    gpu_count (int): total number of GPUs
    """
    return 24 / chkpt_freq_per_day / 2 * exp_fail * gpu_count

exp_gpu_time_recomp_in_gpu_hours = expected_gpu_time_recomputing(
    st.session_state.chkpt_freq_per_day,
    exp_fail,
    st.session_state.gpu_count,
)

# Number of Checkpoints

@st.cache_data
def checkpoint_count(
    cluster_a_train_time_in_days: float,
    chkpt_freq_per_day: float) -> float:
    """
    Returns the number of checkpoints based on cluster actual training time

    Parameters:
    cluster_a_train_time_in_days (float): cluster actual training time in days
    chckpt_freq_per_day (float): checkpoint frequency per day
    """
    return math.ceil(cluster_a_train_time_in_days * chkpt_freq_per_day)

chkpt_count = checkpoint_count(
    cluster_a_train_time_in_days,
    st.session_state.chkpt_freq_per_day,
)

# Cumulative Checkpoint Size (GB)

@st.cache_data
def cumulative_checkpoint_size(
    chkpt_count: int,
    chkpt_size_in_GB: float) -> float:
    """
    Returns the cumulative checkpoint storage size in GB

    Parameters:
    chkpt_count (int): number of checkpoints
    chkpt_size_in_GB (float): size of a single checkpoint in GB
    """
    return chkpt_count * chkpt_size_in_GB

cum_chkpt_size_in_GB = cumulative_checkpoint_size(
    chkpt_count,
    chkpt_size_in_GB,
)

st.markdown(f"""
<table>
  <tr>
    <td>
       <b>Instance Count</b>
    </td>
    <td>
       <code>{inst_count:,}</code>
    </td>
  </tr>
  <tr>
    <td>
       <b>Checkpoint Size</b>
    </td>
    <td>
       <code>{chkpt_size_in_GB:,.1f} GB</code>
    </td>
  </tr>
  <tr>
    <td>
       <b>Dataset Size</b>
    </td>
    <td>
       <code>{data_size_in_GB:,.1f} GB</code>
    </td>
  </tr>
  <tr>
    <td>
       <b>Actual Performance</b>
    </td>
    <td>
       <code>{actual_perf_in_TFLOP_s_per_gpu:,} TFLOP/s/GPU</code>
    </td>
  </tr>
  <tr>
    <td>
       <b>Training Compute Requirements</b>
    </td>
    <td>
       <code>{train_comp_reqs_in_TFLOP:.2E} TFLOP</code>
    </td>
  </tr>
  <tr>
    <td>
       <b>Theoretical Training Time</b>
    </td>
    <td>
       <code>{t_train_time_in_gpu_hours:,.1f} GPU-hours</code>
    </td>
  </tr>
  <tr>
    <td>
       <b>Cluster Theoretical Training Time</b>
    </td>
    <td>
       <code>{cluster_t_train_time_in_days:,.1f} days</code>
    </td>
  </tr>
  <tr>
    <td>
       <b>Cluster Actual Training Time</b>
    </td>
    <td>
       <code>{cluster_a_train_time_in_days:,.1f} days</code>
    </td>
  </tr>
  <tr>
    <td>
       <b>Expected Failures</b>
    </td>
    <td>
       <code>{exp_fail:,.0f}</code>
    </td>
  </tr>
  <tr>
    <td>
       <b>Expected GPU Time in Failed State</b>
    </td>
    <td>
       <code>{exp_gpu_time_fail_in_gpu_hours:,.0f} GPU-hours</code>
    </td>
  </tr>
  <tr>
    <td>
       <b>Expected GPU Time Recomputing</b>
    </td>
    <td>
       <code>{exp_gpu_time_recomp_in_gpu_hours:,.0f} GPU-hours</code>
    </td>
  </tr>
  <tr>
    <td>
       <b># Checkpoints</b>
    </td>
    <td>
       <code>{chkpt_count:,}</code>
    </td>
  </tr>
  <tr>
    <td>
       <b>Cumulative Checkpoint Size</b>
    </td>
    <td>
       <code>{cum_chkpt_size_in_GB:,.1f} GB</code>
    </td>
  </tr>
</table>
<br>
""", unsafe_allow_html=True)

"""
### Costs 💰
"""

# Total Storage Cost

@st.cache_data
def storage_cost(
    sto_cost_per_GB_month: float,
    data_size_in_GB: float,
    cum_chkpt_size_in_GB: float,
    cluster_t_train_time_in_days: float) -> float:
    """
    Returns the total storage cost

    Parameters:
    sto_cost_per_GB_month (float): storage cost per GB-month
    data_size_in_GB (float): training dataset size in GB
    cum_chkpt_size_in_GB (float): cumulative checkpoint size in GB
    cluster_t_train_time_in_days (float): cluster theoretical training time in days
    """
    return sto_cost_per_GB_month * (data_size_in_GB + cum_chkpt_size_in_GB/2) * (cluster_t_train_time_in_days/30.25)

sto_cost = storage_cost(
    st.session_state.sto_cost_per_GB_month,
    data_size_in_GB,
    cum_chkpt_size_in_GB,
    cluster_t_train_time_in_days,
)

# Experiments Cost

@st.cache_data
def experiments_cost(
    gpu_price_per_hour: float,
    expt_budget_in_gpu_hours: float) -> float:
    """
    Returns the total cost of experimentation

    Parameters:
    gpu_price_per_hour (float): hourly price for a single GPU
    expt_budget_in_gpu_hours (float): budget for experimentation in GPU-hours
    """
    return gpu_price_per_hour * expt_budget_in_gpu_hours

expt_cost = experiments_cost(
    st.session_state.gpu_price_per_hour,
    st.session_state.expt_budget_in_gpu_hours,
)

# Downtime Cost

@st.cache_data
def failures_downtime_cost(
    gpu_price_per_hour: float,
    exp_gpu_time_fail_in_gpu_hours: float) -> float:
    """
    Returns the cost associated with downtime due to failures

    Parameters:
    gpu_price_per_hour (float): hourly price for a single GPU
    exp_gpu_time_fail_in_gpu_hours (float): expected GPU time in a failed state expressed in GPU-hours
    """
    return gpu_price_per_hour * exp_gpu_time_fail_in_gpu_hours


fail_downt_cost = failures_downtime_cost(
    st.session_state.gpu_price_per_hour,
    exp_gpu_time_fail_in_gpu_hours,
)

# Recomputation Cost

@st.cache_data
def failures_recomputation_cost(
    gpu_price_per_hour: float,
    exp_gpu_time_recomp_in_gpu_hours: float) -> float:
    """
    Returns the cost associated with recomputation due to failures

    Parameters:
    gpu_price_per_hour (float): hourly price for a single GPU
    exp_gpu_time_recomp_in_gpu_hours (float): expected GPU time spent recomputing uncheckpointed work expressed in GPU-hours
    """
    return gpu_price_per_hour * exp_gpu_time_recomp_in_gpu_hours

fail_recomp_cost = failures_downtime_cost(
    st.session_state.gpu_price_per_hour,
    exp_gpu_time_recomp_in_gpu_hours,
)

# Theoretical Training Cost

@st.cache_data
def theoretical_training_cost(
    gpu_price_per_hour: float,
    t_train_time_in_gpu_hours: float) -> float:
    """
    Returns the theoretical training cost

    Parameters:
    gpu_price_per_hour (float): hourly price for a single GPU
    t_train_time_in_gpu_hours (float): theoretical training time in GPU-hours
    """
    return gpu_price_per_hour * t_train_time_in_gpu_hours

t_train_cost = theoretical_training_cost(
    st.session_state.gpu_price_per_hour,
    t_train_time_in_gpu_hours,
)

# Total Training Cost

@st.cache_data
def total_training_cost(sto_cost, expt_cost, downt_cost, recomp_cost, t_train_cost):
    """
    Returns the total training cost
    """
    return sto_cost + expt_cost + downt_cost + recomp_cost + t_train_cost

total_train_cost = total_training_cost(
    sto_cost,
    expt_cost,
    fail_downt_cost,
    fail_recomp_cost,
    t_train_cost,
)

###########
# Outputs #
###########

st.markdown(f"""
<table>
  <tr>
    <td>
       <b>Storage</b>
    </td>
    <td>
       <div style='float: right; text-align: right'>
       <code>{sto_cost:,.2f}{currencies[st.session_state.currency]}</code>
       </div>
    </td>
  </tr>
  <tr>
    <td>
       <b>Experiments</b>
    </td>
    <td>
       <div style='float: right; text-align: right'>
       <code>{expt_cost:,.2f}{currencies[st.session_state.currency]}</code>
       </div>
    </td>
  </tr>
  <tr>
    <td>
       <b>Failures (Downtime)</b>
    </td>
    <td>
       <div style='float: right; text-align: right'>
       <code>{fail_downt_cost:,.2f}{currencies[st.session_state.currency]}</code>
       </div>
    </td>
  </tr>
  <tr>
    <td>
       <b>Failures (Recomputation)</b>
    </td>
    <td>
       <div style='float: right; text-align: right'>
       <code>{fail_recomp_cost:,.2f}{currencies[st.session_state.currency]}</code>
       </div>
    </td>
  </tr>
  <tr>
    <td>
       <b>Theoretical Training Cost</b>
    </td>
    <td>
       <div style='float: right; text-align: right'>
       <code>{t_train_cost:,.2f}{currencies[st.session_state.currency]}</code>
       </div>
    </td>
  </tr>
  <tr>
    <td>
       <b style="color:red">Total Training Cost</b>
    </td>
    <td>
       <div style='float: right; text-align: right'>
       <code>{total_train_cost:,.2f}{currencies[st.session_state.currency]}</code>
       </div>
    </td>
  </tr>
</table>
<br>
""", unsafe_allow_html=True)

"""
### Want to learn more?

#### Articles 📚

* (Brown *et al.*, 2020) [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165)
* (Hoffmann *et al.*, 2022) [Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556)
* (Kaplan *et al.*, 2020) [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361)
* (Shoeybi *et al.*, 2019) [Megratron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism](https://arxiv.org/abs/1909.08053)

#### Blogs ✍️

* [The FLOPs Calculus of Language Model Training](https://medium.com/@dzmitrybahdanau/the-flops-calculus-of-language-model-training-3b19c1f025e4) by Dzmitry Bahdanau
* [Transformer Math 101](https://blog.eleuther.ai/transformer-math/) by EleutherAI
* [Transformer Inference Arithmetic](https://kipp.ly/transformer-inference-arithmetic/) by Kipply
* [New Scaling Laws for Large Language Models](https://www.lesswrong.com/posts/midXmMb2Xg37F2Kgn/new-scaling-laws-for-large-language-models) by LessWrong
* [AWS ML Infrastructure](https://aws.amazon.com/machine-learning/infrastructure/)
"""
