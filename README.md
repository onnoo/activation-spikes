# Mitigating Quantization Errors Due to Activation Spikes in GLU-Based LLMs [[pdf](https://arxiv.org/abs/2405.14428)]

## Installation

**Requirements)**
- pytorch==2.2.0
- transformers==4.38.1

```bash
pip install transformers==4.38.1 accelerate bitsandbytes easydict matplotlib datasets scipy seaborn sentencepiece protobuf

# install lm-eval
git clone https://github.com/EleutherAI/lm-evaluation-harness.git
cd lm-evaluation-harness
git checkout 6a1c19
pip install -e .
cd ..

# install act_spike
pip install -e .
```
## Run Code

### 1) Prepare misc (calibration results)

```
cd exp
python extract_misc.py {hf_model_name}
```

### 2) Evaluation

```
cd exp
python eval.py {hf_model_name} {--flags}
```

**Spported Flags)**
- `--use_cache` : enable QFeP
- `--except_layer` : enable QFeM
- `--sq` : enable smooth quant
- `--osp` : enable outlier suppression plus
- `--weight_quant` : weight quantization scheme
- `--act_granul` : activation quantization scheme
- `--bmm` : enable BMM quantization
- `--fp16` : enable FP16

### 3) Benchmark Computational Cost

```
cd exp
python bench.py {hf_model_name} {--flags}
```

**Spported Flags)**
- `--use_cache` : enable QFeP
- `--except_layer` : enable QFeM
- `--seqlen` : set sequence length
- `--n_samples` : set number of samples
- `--act_granul` : activation quantization scheme
- `--fp16` : enable FP16
