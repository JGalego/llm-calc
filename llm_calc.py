#pylint: disable=line-too-long,redefined-outer-name,pointless-statement,pointless-string-statement
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
The purpose of the LLM Calculator is to size an LLM pre-training workload by predicting training duration, compute and storage costs.

It should work with **any** accelerator, provided you have accurate information on hourly price and peak TFLOPs, and **any** Transformer-based model whose compute requirements are well-approximated by the [FLOPs equation](https://medium.com/@dzmitrybahdanau/the-flops-calculus-of-language-model-training-3b19c1f025e4).
"""

# Let's assume that the checkpoint size follows the BLOOM ratio
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
    label='GPU hourly price',
    key='gpu_hourly_price',
    value=4.09 if st.session_state.currency == 'USD' else 3.77,  # p4d on-demand pricing for us-east-1
    step=0.01,
    format='%.2f',
)

# How much does it cost to store 1GB for 1 month?
st.sidebar.number_input(
    label='Storage Cost (per GB-month)',
    key='storage_cost',
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
    key='checkpoint_size_ratio',
    value=BLOOM_RATIO,
    step=0.1,
    format='%.1f',
)

with st.expander("Disclaimer ⚠️"):
    f"""
    This application is *highly* experimental! ⚡

    As with all experiments, we had to make a few assumptions:

    * Hardware failures impact only training, not experimentation
    * Training datasets contain `{st.session_state.tokens_per_byte_ratio:.2f}` tokens per byte
    * Checkpoint size is around `{st.session_state.checkpoint_size_ratio:.1f}GB` per billion parameters
    * ... *and many, many more*!

    These are only valid under *strict* conditions.

    If you don't trust the numbers, please consider running your own experiments.

    Quoting from [Isaiah Berlin](https://plato.stanford.edu/entries/berlin/):

    > *"No amount of calculation can save us from painful choices and imperfect solutions"*
    """

"""
### Inputs ➡️
"""

st.number_input(
    label='Peak Theoretical Performance (TFLOP/s/GPU)',
    key='peak_tflops_per_gpu',
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
    key='instance_failure_rate',
    min_value=0.01,
    max_value=1.00,
    step=0.01,
)

st.number_input(
    label='Failure Recovery Time (hours)',
    key='failure_recovery_time',
    value=1.0,
    help=r'This value varies depending on the checkpoint speed and failure recovery automation',
)

st.number_input(
    label='Model Size (Billion parameters)',
    key='model_size',
    value=40,
)

st.number_input(
    label='Training Data Size (Gigatokens)',
    key='training_data_size',
    value=1000.0,
)

st.number_input(
    label='Experiments Budget (GPU-hours)',
    key='experiments_budget',
    value=0.0,
    help=r'This is usually between 1-10% of the total budget'
)

st.number_input(
    label='Checkpoint Frequency (per day)',
    key='checkpoint_frequency',
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

    and translate the **Training Dataset Size** ($D$) from `GTokens` to `GB`

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

@st.cache_data
def instance_count(gpu_count, gpus_per_instance):
    """
    Computes the number of GPU instances necessary to fulfill the GPU count
    """
    return math.ceil(gpu_count / gpus_per_instance)

@st.cache_data
def checkpoint_size(model_size, checkpoint_size_ratio):
    """
    Estimates checkpoint size based on model size
    """
    return checkpoint_size_ratio * model_size

@st.cache_data
def training_data_size(training_data_size, tokens_per_byte_ratio):
    """
    Estimates training data size in bytes
    """
    return training_data_size / tokens_per_byte_ratio

@st.cache_data
def actual_tflops_per_gpu(peak_tflops_per_gpu, gpu_utilization):
    """
    Computes actual TFLOPs per GPU based on peak performance and GPU utilization
    """
    return peak_tflops_per_gpu * gpu_utilization / 100

@st.cache_data
def training_tflops_requirements(model_size, training_data_size):
    """
    Computes training TFLOPs requirements based on the size of the model and training data
    """
    return 6 * model_size * 1e9 * training_data_size * 1e9 / 1e12

@st.cache_data
def theoretical_training_time(training_tflops_requirements, actual_tflops_per_gpu):
    """
    Computes theoretical training time in GPU-hours
    """
    return training_tflops_requirements / actual_tflops_per_gpu / 3600

@st.cache_data
def cluster_theoretical_training_time(theoretical_training_time, gpu_count):
    """
    Computes cluster theoretical training time in days
    """
    return theoretical_training_time / gpu_count / 24

@st.cache_data
def cluster_actual_training_time(cluster_theoretical_training_time, instance_count, instance_failure_rate, failure_recovery_time, checkpoint_frequency):
    """
    Computes cluster actual training time in days
    """
    return cluster_theoretical_training_time / (1 - (instance_failure_rate / 100) * instance_count * (failure_recovery_time/24 + 1/(2 * checkpoint_frequency)))

@st.cache_data
def expected_failures(instance_failure_rate, cluster_actual_training_time, instance_count):
    """
    Computes the number of expected failures
    """
    return (instance_failure_rate / 100.) * cluster_actual_training_time * instance_count

@st.cache_data
def expected_gpu_time_in_failed_state(expected_failures, gpu_count, failure_recovery_time):
    """
    Computes expected GPU-hours spent in failed state
    """
    return expected_failures * gpu_count * failure_recovery_time

@st.cache_data
def expected_gpu_time_recomputing_uncheckpointed_work(checkpoint_frequency, expected_failures, gpu_count):
    """
    Computes expected GPU-hours to recover uncheckpointed work
    """
    return 24 / checkpoint_frequency / 2 * expected_failures * gpu_count

@st.cache_data
def number_of_checkpoints(cluster_actual_training_time, checkpoint_frequency):
    """
    Computes the number of checkpoints based on cluster actual training time
    """
    return math.ceil(cluster_actual_training_time * checkpoint_frequency)

@st.cache_data
def cumulative_checkpoint_size(number_of_checkpoints, checkpoint_size):
    """
    Computes the cumulative checkpoint storage size in GB
    """
    return number_of_checkpoints * checkpoint_size

inst_count = instance_count(st.session_state.gpu_count, st.session_state.gpus_per_instance)

cpoint_size = checkpoint_size(st.session_state.model_size, st.session_state.checkpoint_size_ratio)

train_data_size = training_data_size(st.session_state.training_data_size, st.session_state.tokens_per_byte_ratio)

a_tflops_per_gpu = actual_tflops_per_gpu(st.session_state.peak_tflops_per_gpu, st.session_state.gpu_utilization)

training_tflops_reqs = training_tflops_requirements(st.session_state.model_size, st.session_state.training_data_size)

t_training_time = theoretical_training_time(training_tflops_reqs, a_tflops_per_gpu)

cluster_t_training_time = cluster_theoretical_training_time(t_training_time, st.session_state.gpu_count)

cluster_a_training_time = cluster_actual_training_time(cluster_t_training_time, inst_count, st.session_state.instance_failure_rate, st.session_state.failure_recovery_time, st.session_state.checkpoint_frequency)

exp_failures = expected_failures(st.session_state.instance_failure_rate, cluster_a_training_time, inst_count)

exp_gpu_time_in_failed_state = expected_gpu_time_in_failed_state(exp_failures, st.session_state.gpu_count, st.session_state.failure_recovery_time)

exp_gpu_time_recomputing_uncheckpointed_work = expected_gpu_time_recomputing_uncheckpointed_work(st.session_state.checkpoint_frequency, exp_failures, st.session_state.gpu_count)

num_checkpoints = number_of_checkpoints(cluster_a_training_time, st.session_state.checkpoint_frequency)

cum_cpoint_size = cumulative_checkpoint_size(num_checkpoints, cpoint_size)

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
       <b>Checkpoint Size (GB)</b>
    </td>
    <td>
       <code>{cpoint_size:,.1f}</code>
    </td>
  </tr>
  <tr>
    <td>
       <b>Training Data Size (GB)</b>
    </td>
    <td>
       <code>{train_data_size:,.1f}</code>
    </td>
  </tr>
  <tr>
    <td>
       <b>Actual TFLOPs/s/GPU</b>
    </td>
    <td>
       <code>{a_tflops_per_gpu:,}</code>
    </td>
  </tr>
  <tr>
    <td>
       <b>Training TFLOPs Requirements</b>
    </td>
    <td>
       <code>{training_tflops_reqs:.2E}</code>
    </td>
  </tr>
  <tr>
    <td>
       <b>Theoretical Training Time (GPU-hours)</b>
    </td>
    <td>
       <code>{t_training_time:,.1f}</code>
    </td>
  </tr>
  <tr>
    <td>
       <b>Cluster Theoretical Training Time (days)</b>
    </td>
    <td>
       <code>{cluster_t_training_time:,.1f}</code>
    </td>
  </tr>
  <tr>
    <td>
       <b>Cluster Actual Training Time (days)</b>
    </td>
    <td>
       <code>{cluster_a_training_time:,.1f}</code>
    </td>
  </tr>
  <tr>
    <td>
       <b>Expected Failures</b>
    </td>
    <td>
       <code>{exp_failures:,.1f}</code>
    </td>
  </tr>
  <tr>
    <td>
       <b>Expected GPU Time in Failed State (GPU-hours)</b>
    </td>
    <td>
       <code>{exp_gpu_time_in_failed_state:,.0f}</code>
    </td>
  </tr>
  <tr>
    <td>
       <b>Expected GPU Time Recomputing Uncheckpointed Work (GPU-hours)</b>
    </td>
    <td>
       <code>{exp_gpu_time_recomputing_uncheckpointed_work:,.0f}</code>
    </td>
  </tr>
  <tr>
    <td>
       <b># Checkpoints</b>
    </td>
    <td>
       <code>{num_checkpoints:,}</code>
    </td>
  </tr>
  <tr>
    <td>
       <b>Cumulative Checkpoint Size (GB)</b>
    </td>
    <td>
       <code>{cum_cpoint_size:,.1f}</code>
    </td>
  </tr>
</table>
<br>
""", unsafe_allow_html=True)

"""
### Costs 💰
"""

@st.cache_data
def total_storage_cost(training_data_size, storage_cost, cluster_theoretical_training_time, cumulative_checkpoint_size):
    """
    Computes the total storage cost
    """
    return training_data_size * storage_cost * (cluster_theoretical_training_time / 30.25) + cumulative_checkpoint_size * storage_cost * (cluster_theoretical_training_time / 30.25) / 2

@st.cache_data
def total_experiments_cost(gpu_hourly_price, experiments_budget):
    """
    Computes the total cost of experimentation
    """
    return gpu_hourly_price * experiments_budget

@st.cache_data
def failures_downtime_cost(gpu_hourly_price, expected_gpu_time_in_failed_state):
    """
    Computes the cost associated with downtime due to failures
    """
    return gpu_hourly_price * expected_gpu_time_in_failed_state

@st.cache_data
def failures_recomputation_cost(gpu_hourly_price, expected_gpu_time_recomputing_uncheckpointed_work):
    """
    Computes the cost associated with recomputation due to failures
    """
    return gpu_hourly_price * expected_gpu_time_recomputing_uncheckpointed_work

@st.cache_data
def theoretical_training_cost(gpu_hourly_price, theoretical_training_time):
    """
    Computes the theoretical training cost
    """
    return gpu_hourly_price * theoretical_training_time

@st.cache_data
def total_training_cost(storage_cost, experiments_cost, downtime_cost, recomputation_cost, theoretical_training_cost):
    """
    Computes the total training cost
    """
    return storage_cost + experiments_cost + downtime_cost + recomputation_cost + theoretical_training_cost

tot_storage_cost = total_storage_cost(train_data_size, st.session_state.storage_cost, cluster_t_training_time, cum_cpoint_size)

tot_experiments_cost = total_experiments_cost(st.session_state.gpu_hourly_price, st.session_state.experiments_budget)

fail_downtime_cost = failures_downtime_cost(st.session_state.gpu_hourly_price, exp_gpu_time_in_failed_state)

fail_recomputation_cost = failures_downtime_cost(st.session_state.gpu_hourly_price, exp_gpu_time_recomputing_uncheckpointed_work)

t_training_cost = theoretical_training_cost(st.session_state.gpu_hourly_price, t_training_time)

tot_training_cost = total_training_cost(tot_storage_cost, tot_experiments_cost, fail_downtime_cost, fail_recomputation_cost, t_training_cost)

st.markdown(f"""
<table>
  <tr>
    <td>
       <b>Storage</b>
    </td>
    <td>
       <code>{tot_storage_cost:,.2f}{currencies[st.session_state.currency]}</code>
    </td>
  </tr>
  <tr>
    <td>
       <b>Experiments</b>
    </td>
    <td>
       <code>{tot_experiments_cost:,.2f}{currencies[st.session_state.currency]}</code>
    </td>
  </tr>
  <tr>
    <td>
       <b>Failures (Downtime)</b>
    </td>
    <td>
       <code>{fail_downtime_cost:,.2f}{currencies[st.session_state.currency]}</code>
    </td>
  </tr>
  <tr>
    <td>
       <b>Failures (Recomputation)</b>
    </td>
    <td>
       <code>{fail_recomputation_cost:,.2f}{currencies[st.session_state.currency]}</code>
    </td>
  </tr>
  <tr>
    <td>
       <b>Theoretical Training Cost</b>
    </td>
    <td>
       <code>{t_training_cost:,.2f}{currencies[st.session_state.currency]}</code>
    </td>
  </tr>
  <tr>
    <td>
       <b style="color:red">Total Training Cost</b>
    </td>
    <td>
       <code>{tot_training_cost:,.2f}{currencies[st.session_state.currency]}</code>
    </td>
  </tr>
</table>
<br>
""", unsafe_allow_html=True)

"""
### Want to learn more? 📚

* [The FLOPS Calculus of Language Model Training](https://medium.com/@dzmitrybahdanau/the-flops-calculus-of-language-model-training-3b19c1f025e4) by Dzmitry Bahdanau
* [Transformer Math 101](https://blog.eleuther.ai/transformer-math/) by EleutherAI
* [Transformer Inference Arithmetic](https://kipp.ly/transformer-inference-arithmetic/) by Kipply
* [New Scaling Laws for Large Language Models](https://www.lesswrong.com/posts/midXmMb2Xg37F2Kgn/new-scaling-laws-for-large-language-models) by LessWrong
* [AWS ML Infrastructure](https://aws.amazon.com/machine-learning/infrastructure/)
"""
