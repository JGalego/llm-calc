#pylint: disable=line-too-long,redefined-outer-name,pointless-statement,pointless-string-statement
r"""
  _      _      __  __    _____      _            _       _
 | |    | |    |  \/  |  / ____|    | |          | |     | |
 | |    | |    | \  / | | |     __ _| | ___ _   _| | __ _| |_ ___  _ __
 | |    | |    | |\/| | | |    / _` | |/ __| | | | |/ _` | __/ _ \| '__|
 | |____| |____| |  | | | |___| (_| | | (__| |_| | | (_| | || (_) | |
 |______|______|_|  |_|  \_____\__,_|_|\___|\__,_|_|\__,_|\__\___/|_|

Find out how much compute power and storage you need to train your model
"""

import math

import streamlit as st

st.title("LLM Calculator 🧮")
st.subheader("How much compute and storage do I need?")

# Let's assume that the checkpoint size follows the BLOOM ratio
# 2.3TB / 176 ~ 13GB per billion parameters
# https://huggingface.co/bigscience/tr11-176B-logs
BLOOM_RATIO = 2.3e3/176

# and that the training data contains ~0.2 tokens per byte
# https://accubits.com/large-language-models-leaderboard/bloom/
TOKENS_PER_BYTE = 350e9/1.6e12

"""
The purpose of the LLM Calculator is to size an LLM pre-training workload by predicting training duration, storage and compute costs. It works with **any** accelerator, provided you have information on hourly price and peak TFLOPs, and **any** model that adheres to the [Transformer FLOPs equation](https://medium.com/@dzmitrybahdanau/the-flops-calculus-of-language-model-training-3b19c1f025e4).
"""

with st.expander("Disclaimer ⚠️"):
    f"""
    This application is *highly* experimental! ⚡

    As with all experiments, we had to make a few assumptions:

    * Hardware failures impact only training, not experimentation
    * Training datasets contain `{TOKENS_PER_BYTE:.2f}` tokens-per-byte
    * Checkpoint size is around `{BLOOM_RATIO:.1f}GB` per billion parameters
    * ... *and many, many more*!

    These are only valid under strict conditions.

    If you don't trust the numbers, consider running your own experiments.

    Quoting from [Isaiah Berlin](https://plato.stanford.edu/entries/berlin/):

    > "No amount of calculation can save us from painful choices and imperfect solutions"
    """

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

st.sidebar.number_input(
    label='GPU hourly price',
    key='gpu_hourly_price',
    value=2.50 if st.session_state.currency == 'USD' else 2.30,
    step=0.01,
    format='%.2f',
    help="Use a precise PPA rate if one is available",
)

st.sidebar.number_input(
    label='Storage Cost per GB-month',
    key='storage_cost',
    value=0.022 if st.session_state.currency == 'USD' else 0.020,
    step=0.001,
    format='%.3f',
    help="Default value assumes S3 Standard tier",
)

"""
### Inputs ➡️
"""

st.number_input(
    label='Peak TFLOP/s/GPU',
    key='peak_tflops_per_gpu',
    value=312,
    min_value=0,
    # A100: https://www.nvidia.com/en-us/data-center/a100/
    # Trn1: https://aws.amazon.com/machine-learning/trainium/
    # H100: https://www.nvidia.com/en-us/data-center/h100/
)

st.number_input(
    label='GPU Utilization',
    key='gpu_utilization',
    value=0.30,
    min_value=0.01,
    max_value=1.00,
    step=0.01,
    help=r'30% is a good default, 50% if well optimized stack',
)

st.number_input(
    label='GPU Count',
    key='gpu_count',
    value=300,
    min_value=0,
    help='Usually between 5 and 20 GPUs per billion parameters',
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
    min_value=1e-2,
    max_value=1.0,
    step=1e-4,
)

st.number_input(
    label='Failure Recovery Time (hours)',
    key='failure_recovery_time',
    value=1.0,
    help='Depends on failure recovery automation and checkpoint speed',
)

st.number_input(
    label='Model Size (Billion parameters)',
    key='model_size',
    value=40,
)

st.number_input(
    label='Training Data Size (Gigatokens)',
    key='training_data_size',
    value=1000,
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

@st.cache_data
def instance_count(gpu_count, gpus_per_instance):
    """
    Computes the number of GPU instances necessary to fulfill the GPU count
    """
    return math.ceil(gpu_count / gpus_per_instance)

@st.cache_data
def real_tflops_per_gpu(peak_tflops_per_gpu, gpu_utilization):
    """
    Computes real TFLOPs per GPU based on peak usage and GPU utilization
    """
    return math.ceil(peak_tflops_per_gpu * gpu_utilization)

@st.cache_data
def training_tflops_requirements(model_size, training_data_size):
    """
    Computes training TFLOPs requirements based on the size of the model and training data
    """
    return 6 * model_size * 1e9 * training_data_size * 1e9 / 1e12

@st.cache_data
def theoretical_training_time(training_tflops_requirements, real_tflops_per_gpu):
    """
    Computes theoretical training time in GPU-hours
    """
    return math.ceil(training_tflops_requirements / real_tflops_per_gpu / 3600)

@st.cache_data
def cluster_theoretical_training_time(theoretical_training_time, gpu_count):
    """
    Computes cluster theoretical training time in days
    """
    return math.ceil(theoretical_training_time / gpu_count / 24)

@st.cache_data
def cluster_real_training_time(cluster_theoretical_training_time, instance_count, instance_failure_rate, failure_recovery_time, checkpoint_frequency):
    """
    Computes cluster real training time in days
    """
    return math.ceil(cluster_theoretical_training_time / (1 - instance_failure_rate * instance_count * (failure_recovery_time/24 + 1/(2 * checkpoint_frequency))))

@st.cache_data
def expected_failures(instance_failure_rate, cluster_real_training_time, instance_count):
    """
    Computes the number of expected failures
    """
    return math.ceil(instance_failure_rate * cluster_real_training_time * instance_count)

@st.cache_data
def expected_gpu_time_in_failed_state(expected_failures, gpu_count, failure_recovery_time):
    """
    Computes expected GPU-hours spent in failed state
    """
    return math.ceil(expected_failures * gpu_count * failure_recovery_time)

@st.cache_data
def expected_gpu_time_recomputing_uncheckpointed_work(checkpoint_frequency, expected_failures, gpu_count):
    """
    Computes expected GPU-hours to recover uncheckpointed work
    """
    return math.ceil(24 / (checkpoint_frequency / (2 * expected_failures * gpu_count)))

@st.cache_data
def number_of_checkpoints(cluster_real_training_time, checkpoint_frequency):
    """
    Computes the number of checkpoints based on cluster real training time
    """
    return cluster_real_training_time * checkpoint_frequency

@st.cache_data
def cumulative_checkpoint_size(number_of_checkpoints, checkpoint_size):
    """
    Computes the cumulative checkpoint storage size in GB
    """
    return number_of_checkpoints * checkpoint_size

inst_count = instance_count(st.session_state.gpu_count, st.session_state.gpus_per_instance)

checkpoint_size = int(BLOOM_RATIO * st.session_state.model_size)

training_data_size = int(st.session_state.training_data_size / TOKENS_PER_BYTE)

r_tflops_per_gpu = real_tflops_per_gpu(st.session_state.peak_tflops_per_gpu, st.session_state.gpu_utilization)

training_tflops_reqs = training_tflops_requirements(st.session_state.model_size, st.session_state.training_data_size)

t_training_time = theoretical_training_time(training_tflops_reqs, r_tflops_per_gpu)

cluster_t_training_time = cluster_theoretical_training_time(t_training_time, st.session_state.gpu_count)

cluster_r_training_time = cluster_real_training_time(cluster_t_training_time, inst_count, st.session_state.instance_failure_rate, st.session_state.failure_recovery_time, st.session_state.checkpoint_frequency)

exp_failures = expected_failures(st.session_state.instance_failure_rate, cluster_r_training_time, inst_count)

exp_gpu_time_in_failed_state = expected_gpu_time_in_failed_state(exp_failures, st.session_state.gpu_count, st.session_state.failure_recovery_time)

exp_gpu_time_recomputing_uncheckpointed_work = expected_gpu_time_recomputing_uncheckpointed_work(st.session_state.checkpoint_frequency, exp_failures, st.session_state.gpu_count)

num_checkpoints = number_of_checkpoints(cluster_r_training_time, st.session_state.checkpoint_frequency)

cum_checkpoint_size = cumulative_checkpoint_size(num_checkpoints, checkpoint_size)

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
       <code>{checkpoint_size:,}</code>
    </td>
  </tr>
  <tr>
    <td>
       <b>Training Data Size (GB)</b>
    </td>
    <td>
       <code>{training_data_size:,}</code>
    </td>
  </tr>
  <tr>
    <td>
       <b>Real TFLOPs/s/GPU</b>
    </td>
    <td>
       <code>{r_tflops_per_gpu:,}</code>
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
       <code>{t_training_time:,}</code>
    </td>
  </tr>
  <tr>
    <td>
       <b>Cluster Theoretical Training Time (days)</b>
    </td>
    <td>
       <code>{cluster_t_training_time:,}</code>
    </td>
  </tr>
  <tr>
    <td>
       <b>Cluster Real Training Time (days)</b>
    </td>
    <td>
       <code>{cluster_r_training_time:,}</code>
    </td>
  </tr>
  <tr>
    <td>
       <b>Expected Failures</b>
    </td>
    <td>
       <code>{exp_failures:,}</code>
    </td>
  </tr>
  <tr>
    <td>
       <b>Expected GPU Time in Failed State (GPU-hours)</b>
    </td>
    <td>
       <code>{exp_gpu_time_in_failed_state:,}</code>
    </td>
  </tr>
  <tr>
    <td>
       <b>Expected GPU Time Recomputing Uncheckpointed Work (GPU-hours)</b>
    </td>
    <td>
       <code>{exp_gpu_time_recomputing_uncheckpointed_work:,}</code>
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
       <code>{cum_checkpoint_size:,}</code>
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

tot_storage_cost = total_storage_cost(training_data_size, st.session_state.storage_cost, cluster_t_training_time, cum_checkpoint_size)

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
