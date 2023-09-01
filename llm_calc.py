r"""
  _      _      __  __    _____      _            _       _
 | |    | |    |  \/  |  / ____|    | |          | |     | |
 | |    | |    | \  / | | |     __ _| | ___ _   _| | __ _| |_ ___  _ __
 | |    | |    | |\/| | | |    / _` | |/ __| | | | |/ _` | __/ _ \| '__|
 | |____| |____| |  | | | |___| (_| | | (__| |_| | | (_| | || (_) | |
 |______|______|_|  |_|  \_____\__,_|_|\___|\__,_|_|\__,_|\__\___/|_|

Find out how much compute power and storage you need to train your model!
"""

import math

import streamlit as st

st.title("LLM Calculator 🧮")
st.subheader("How much compute and storage do I need?")

"""
The purpose of this calculator is to size LLM pre-training workloads by predicting training duration, compute and storage costs. It works with **any** accelerator, provided you have accurate information on hourly price and peak performance, and **any** Transformer-based model whose compute requirements are well-approximated by the [FLOPs equation](https://medium.com/@dzmitrybahdanau/the-flops-calculus-of-language-model-training-3b19c1f025e4).
"""

# By default, we'll assume that the checkpoint size follows the BLOOM ratio
# 2.3TB / 176B ~ 13GB per billion parameters
# https://huggingface.co/bigscience/tr11-176B-logs
# https://huggingface.co/blog/bloom-megatron-deepspeed
BLOOM_RATIO = 2.3e3/176

# and that the training data contains approximately
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

# How much does it cost to run a single GPU per hour?
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
    key='storage_cost_per_GB_month',
    value=0.022 if st.session_state.currency == 'USD' else 0.020,  # S3 standard storage (50-500TB) pricing for us-east-1
    step=0.001,
    format='%.3f',
)

# In our dataset, how many tokens do we have for each byte?
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
    f"""
    This application is *highly* experimental! ⚡

    As with all experiments, we had to make a few assumptions which are only valid under *strict* conditions.

    If you don't trust the numbers, please consider running your own experiments.

    Quoting from [Isaiah Berlin](https://plato.stanford.edu/entries/berlin/):

    > *"No amount of calculation can save us from painful choices and imperfect solutions"*
    """

"""
### Inputs ➡️
"""

st.number_input(
    label='Peak Theoretical Performance (TFLOP/s/GPU)',
    key='peak_perf_in_TFLOPs_per_gpu',
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
    help=r'30% is a good default, 50% if well optimized stack',
)

st.number_input(
    label='GPU Count',
    key='gpu_count',
    value=300,
    min_value=0,
    help=r'Usually between 5 and 20 GPUs per billion parameters',
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
    max_value=1.00,
    step=0.01,
)

st.number_input(
    label='Failure Recovery Time (hours)',
    key='fail_rcv_t_in_h',
    value=1.0,
    help=r'This value varies depending on the checkpoint speed and failure recovery automation',
)

st.number_input(
    label='Model Size (Billion parameters)',
    key='model_size_in_B',
    value=40,
)

st.number_input(
    label='Training Data Size (Gtokens)',
    key='data_size_in_Gtokens',
    value=1000.0,
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
    #### GPU Instance Count

    Assuming that all instances have the same number of GPUs ($n_\texttt{GPU}$), we can easily calculate the total number of GPU instances required

    $$~N_\texttt{instances} = \lceil N_\texttt{GPU} / n_\texttt{GPU} \rceil$$

    The `ceil`ing function ensures that the formula returns the right number of instances even when the total number of GPUs ($N_\texttt{GPU}$) is **not** a factor of the number of GPUs *per instance*.

    #### Estimating Storage

    Checkpoints are intermediate dumps of a model's entire internal state (weights, learning rate, &c.).

    They can be used to resume training or to load a model that has already been trained.

    Based on reported values for similar models and datasets, we can estimate **Checkpoint Size** given the size of the model ($N$)

    $$~r_\texttt{checkpoint} \times N$$

    and translate the **Training Dataset Size** ($D$) from `Gtokens` to `GB`

    $$~D / r_{\texttt{tokens} \rightarrow \texttt{byte}}$$

    As a default, we use the **Checkpoint Size Ratio** ($r_\texttt{checkpoint}$) for the [BLOOM-176B](https://huggingface.co/blog/bloom-megatron-deepspeed) model

    `2.3TB (checkpoint size) / 176B (# parameters) ~ 13.1 GB/Billion parameters`

    and **Tokens per Byte Ratio** ($r_{\texttt{tokens} \rightarrow \texttt{byte}}$) for [The BigScience ROOTS Corpus](https://arxiv.org/abs/2303.03915) dataset

    `350B (# tokens) / 1.6TB (dataset size) ~ 0.22 Tokens/Byte`

    #### Estimating Compute

    GPU datasheets usually advertise their *theoretical* throughput values $\tau_\texttt{peak}$, which are rarely if ever met in practice. One way to correct for this discrepancy is to factor in **GPU Utilization** ($r_\texttt{GPU}$)

    $$~\tau_\texttt{actual} = r_\texttt{GPU} \times \tau_\texttt{peak}$$

    In the regime where the **weight FLOPs** contribution dominates everything else (incl. layer renormalization, attention, residual connections, &c.), we can use the [Transformer FLOPs equation](https://medium.com/@dzmitrybahdanau/the-flops-calculus-of-language-model-training-3b19c1f025e4) (Kaplan *et al.*, 2020; Brown *et al.*, 2020) to estimate compute requirements ($C$) cf. [The FLOPs Calculus of Language Model Training](https://medium.com/@dzmitrybahdanau/the-flops-calculus-of-language-model-training-3b19c1f025e4) for additional details

    $$C \approx 6 \times N \times D$$

    #### Estimating Time

    The **Theoretical Training Time** ($T_\texttt{theory}$) is just the ratio of the training compute requirements ($C$) and the *real* throughput calculated in the last section ($\tau_\texttt{actual}$):

    $$T_\texttt{theory} = C / \tau_\texttt{actual}$$

    Dividing by the **GPU Count**, we get the **Cluster Theoretical Training Time** ($T^\texttt{c}_\texttt{theory}$):

    $$T^\texttt{c}_\texttt{theory} = T_\texttt{theory} / N_\texttt{GPU}$$

    #### Accounting for Failures

    Failures are inevitable. As Werner Vogels (Amazon CTO) is fond of saying:

    > *"Everything fails, all the time"*

    Each time an instance fails, the cluster will have to work *extra time* to recover from downtime and recompute uncheckpointed work:

    $$T^\texttt{c}_\texttt{total} = T^\texttt{c}_\texttt{theory} + T_\texttt{downtime} + T_\texttt{recompute}$$

    The expected number of failures will depend on the number of instance in our cluster, the cluster running time ($T^\texttt{c}_\texttt{total}$) and, of course, the failure rate ($f$)

    $$F = f \times T^\texttt{c}_\texttt{total} \times N_\texttt{instances}$$

    We can use this estimate to compute the **Expected GPU Time in Failed State**

    $$F \times N_\texttt{GPU} \times t_\texttt{failure}$$

    and the **Expected GPU Time Recomputing Uncheckpointed Work**

    $$F \times N_\texttt{GPU} / (2 \times f_\texttt{checkpoint})$$

    where $t_\texttt{failure}$ is the **Failure Recovery Time** and $f_\texttt{checkpoint}$ is the **Checkpoint Frequency**.

    Combining everything, we get a formula for the *real* cluster training time ($T^\texttt{c}_\texttt{total}$)

    $$T^\texttt{c}_\texttt{total} = T^\texttt{c}_\texttt{theory} / [1 - f \times N_\texttt{instances} \times (t_\texttt{failure} + t_\texttt{checkpoint})]$$

    #### References

    * (Shoeybi *et al.*, 2019) [Megratron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism](https://arxiv.org/abs/1909.08053)
    * (Kaplan *et al.*, 2020) [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361)
    * (Brown *et al.*, 2020) [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165)
    * (Hoffmann *et al.*, 2022) [Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556)
    """

# Instance Count

@st.cache_data
def instance_count(gpu_count, gpus_per_instance):
    """
    Returns the number of GPU instances necessary to fulfill a given GPU count
    """
    return math.ceil(gpu_count / gpus_per_instance)

inst_count = instance_count(
    st.session_state.gpu_count,
    st.session_state.gpus_per_instance,
)

# Checkpoint Size (GB)

@st.cache_data
def checkpoint_size(model_size_in_B, chkpt_size_ratio):
    """
    Estimates checkpoint size in GB based on the model size (# parameters)
    """
    return chkpt_size_ratio * model_size_in_B

chkpt_size_in_GB = checkpoint_size(
    st.session_state.model_size_in_B,
    st.session_state.chkpt_size_ratio,
)

# Dataset Size (GB)

@st.cache_data
def data_size(data_size_in_Gtokens, tokens_per_byte_ratio):
    """
    Estimates training data size in GB from the number of tokens
    """
    return data_size_in_Gtokens / tokens_per_byte_ratio

data_size_in_GB = data_size(
    st.session_state.data_size_in_Gtokens,
    st.session_state.tokens_per_byte_ratio,
)

# Actual Performance (TFLOPs/GPU)

@st.cache_data
def actual_performance(peak_perf_in_TFLOPs_per_gpu, gpu_utilization):
    """
    Returns actual TFLOPs per GPU based on peak performance and GPU utilization
    """
    return peak_perf_in_TFLOPs_per_gpu * gpu_utilization / 100

actual_perf_in_TFLOPs_per_gpu = actual_performance(
    st.session_state.peak_perf_in_TFLOPs_per_gpu,
    st.session_state.gpu_utilization
)

# Training Compute Requirements (TFLOPs)

@st.cache_data
def train_compute_requirements(model_size_in_B, data_size_in_Gtokens):
    """
    Returns training TFLOPs requirements based on the size of the model and training data
    """
    return 6 * model_size_in_B * 1e9 * data_size_in_Gtokens * 1e9 / 1e12

train_comp_reqs_in_TFLOPs = train_compute_requirements(
    st.session_state.model_size_in_B,
    st.session_state.data_size_in_Gtokens,
)

# Theoretical Training Time (GPU-hours)

@st.cache_data
def theoretical_training_time(train_comp_reqs_in_TFLOPs, actual_perf_in_TFLOPs_per_gpu):
    """
    Returns the *theoretical* training time in GPU-hours
    """
    return train_comp_reqs_in_TFLOPs / actual_perf_in_TFLOPs_per_gpu / 3600

t_train_time_in_gpu_hours = theoretical_training_time(
    train_comp_reqs_in_TFLOPs,
    actual_perf_in_TFLOPs_per_gpu
)

# Cluster Theoretical Training Time (days)

@st.cache_data
def cluster_theoretical_training_time(t_train_time_in_gpu_hours, gpu_count):
    """
    Returns the cluster *theoretical* training time in days
    """
    return t_train_time_in_gpu_hours / gpu_count / 24

cluster_t_train_time_in_days = cluster_theoretical_training_time(
    t_train_time_in_gpu_hours,
    st.session_state.gpu_count,
)

# Cluster Actual Training Time (days)

@st.cache_data
def cluster_actual_training_time(cluster_t_train_time_in_days, inst_count, inst_fail_rate_per_inst_day, fail_rcv_t_in_h, chkpt_freq_per_day):
    """
    Returns the cluster *actual* training time in days
    """
    return cluster_t_train_time_in_days / (1 - (inst_fail_rate_per_inst_day / 100) * inst_count * (fail_rcv_t_in_h/24 + 1/(2 * chkpt_freq_per_day)))

cluster_a_train_time_in_days = cluster_actual_training_time(
    cluster_t_train_time_in_days,
    inst_count,
    st.session_state.inst_fail_rate_per_inst_day,
    st.session_state.fail_rcv_t_in_h,
    st.session_state.chkpt_freq_per_day,
)

# Expected Failures

@st.cache_data
def expected_failures(inst_fail_rate_per_inst_day, cluster_actual_training_time, inst_count):
    """
    Returns the expected number of failures during the training cycle
    """
    return (inst_fail_rate_per_inst_day / 100) * cluster_actual_training_time * inst_count

exp_fail = expected_failures(
    st.session_state.inst_fail_rate_per_inst_day,
    cluster_a_train_time_in_days,
    inst_count,
)

# Expected GPU Time in Failed State (GPU-hours)

@st.cache_data
def expected_gpu_time_in_failed_state(exp_fail, gpu_count, fail_rcv_t_in_h):
    """
    Returns the expected number of GPU-hours spent in a failed state
    """
    return exp_fail * gpu_count * fail_rcv_t_in_h

exp_gpu_time_fail_in_gpu_hours = expected_gpu_time_in_failed_state(
    exp_fail,
    st.session_state.gpu_count,
    st.session_state.fail_rcv_t_in_h,
)

# Expected GPU Time recomputing uncheckpointed work (GPU-hours)

@st.cache_data
def expected_gpu_time_recomputing(chkpt_freq_per_day, exp_fail, gpu_count):
    """
    Returns the expected number of GPU-hours to recover uncheckpointed work
    """
    return 24 / chkpt_freq_per_day / 2 * exp_fail * gpu_count

exp_gpu_time_recomp_in_gpu_hours = expected_gpu_time_recomputing(
    st.session_state.chkpt_freq_per_day,
    exp_fail,
    st.session_state.gpu_count,
)

# Number of Checkpoints

@st.cache_data
def checkpoint_count(cluster_a_train_time_in_days, chkpt_freq_per_day):
    """
    Returns the number of checkpoints based on cluster actual training time
    """
    return math.ceil(cluster_a_train_time_in_days * chkpt_freq_per_day)

chkpt_count = checkpoint_count(
    cluster_a_train_time_in_days,
    st.session_state.chkpt_freq_per_day,
)

# Cumulative Checkpoint Size (GB)

@st.cache_data
def cumulative_checkpoint_size(chkpt_count, chkpt_size_in_GB):
    """
    Returns the cumulative checkpoint storage size in GB
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
       <code>{actual_perf_in_TFLOPs_per_gpu:,} TFLOPs/s/GPU</code>
    </td>
  </tr>
  <tr>
    <td>
       <b>Training Compute Requirements</b>
    </td>
    <td>
       <code>{train_comp_reqs_in_TFLOPs:.2E} TFLOPs</code>
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
def storage_cost(storage_cost_per_GB_month, data_size_in_GB, cum_chkpt_size_in_GB, cluster_t_train_time_in_days):
    """
    Returns the total storage cost
    """
    return storage_cost_per_GB_month * (data_size_in_GB + cum_chkpt_size_in_GB/2) * (cluster_t_train_time_in_days / 30.25)

sto_cost = storage_cost(
    st.session_state.storage_cost_per_GB_month,
    data_size_in_GB,
    cum_chkpt_size_in_GB,
    cluster_t_train_time_in_days,
)

# Experiments Cost

@st.cache_data
def experiments_cost(gpu_price_per_hour, expt_budget_in_gpu_hours):
    """
    Returns the total cost of experimentation
    """
    return gpu_price_per_hour * expt_budget_in_gpu_hours

expt_cost = experiments_cost(
    st.session_state.gpu_price_per_hour,
    st.session_state.expt_budget_in_gpu_hours,
)

# Downtime Cost

@st.cache_data
def failures_downtime_cost(gpu_price_per_hour, exp_gpu_time_fail_in_gpu_hours):
    """
    Returns the cost associated with downtime due to failures
    """
    return gpu_price_per_hour * exp_gpu_time_fail_in_gpu_hours


fail_downtime_cost = failures_downtime_cost(
    st.session_state.gpu_price_per_hour,
    exp_gpu_time_fail_in_gpu_hours,
)

# Recomputation Cost

@st.cache_data
def failures_recomputation_cost(gpu_price_per_hour, exp_gpu_time_recomp_in_gpu_hours):
    """
    Returns the cost associated with recomputation due to failures
    """
    return gpu_price_per_hour * exp_gpu_time_recomp_in_gpu_hours

fail_recomp_cost = failures_downtime_cost(
    st.session_state.gpu_price_per_hour,
    exp_gpu_time_recomp_in_gpu_hours,
)

# Theoretical Training Cost

@st.cache_data
def theoretical_training_cost(gpu_price_per_hour, t_train_time_in_gpu_hours):
    """
    Returns the theoretical training cost
    """
    return gpu_price_per_hour * t_train_time_in_gpu_hours

t_train_cost = theoretical_training_cost(
    st.session_state.gpu_price_per_hour,
    t_train_time_in_gpu_hours,
)

# Total Training Cost

@st.cache_data
def total_training_cost(storage_cost, experiments_cost, downtime_cost, recomputation_cost, theoretical_training_cost):
    """
    Returns the total training cost
    """
    return storage_cost + experiments_cost + downtime_cost + recomputation_cost + theoretical_training_cost

total_train_cost = total_training_cost(
    sto_cost,
    expt_cost,
    fail_downtime_cost,
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
       <code>{fail_downtime_cost:,.2f}{currencies[st.session_state.currency]}</code>
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

* [The FLOPS Calculus of Language Model Training](https://medium.com/@dzmitrybahdanau/the-flops-calculus-of-language-model-training-3b19c1f025e4) by Dzmitry Bahdanau
* [Transformer Math 101](https://blog.eleuther.ai/transformer-math/) by EleutherAI
* [Transformer Inference Arithmetic](https://kipp.ly/transformer-inference-arithmetic/) by Kipply
* [New Scaling Laws for Large Language Models](https://www.lesswrong.com/posts/midXmMb2Xg37F2Kgn/new-scaling-laws-for-large-language-models) by LessWrong
* [AWS ML Infrastructure](https://aws.amazon.com/machine-learning/infrastructure/)
"""
