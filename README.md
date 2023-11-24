
# TEA: Test-time Energy Adaptation

![Our Proposed TEA](./pic/tea.jpg)

### Main Usage

The default model using trained WRN-28-10 from [RobustBench](https://github.com/RobustBench/robustbench).

```python
CUDA_VISIBLE_DEVICES=0 python main.py --cfg cfgs/cifar10/energy.yaml
```

### Baseline Support

Our code supports running other baselines with a one-line script, the supported baselines include:
- **Source:** model without any adaptation
- **PL:** Pseudo-Label-The Simple and Efficient Semi-Supervised Learning Method for Deep Neural Networks (ICMLW 2013)
- **SHOT:** Do We Really Need to Access the Source Data? Source Hypothesis Transfer for Unsupervised Domain Adaptation (ICML 2020)
- **BN**: Improving robustness against common corruptions by covariate shift adaptation (NeurIPS 2020)
- **TENT:** Tent: Fully Test-Time Adaptation by Entropy Minimization (ICLR 2021)
- **ETA:** Efficient Test-Time Model Adaptation without Forgetting (ICML 2022)
- **EATA:** Efficient Test-Time Model Adaptation without Forgetting (ICML 2022)
- **SAR:** Towards Stable Test-time Adaptation in Dynamic Wild World (ICLR 2023)

```python
# Baselines
CUDA_VISIBLE_DEVICES=0 python main.py --cfg cfgs/cifar10/source.yaml
CUDA_VISIBLE_DEVICES=0 python main.py --cfg cfgs/cifar10/norm.yaml
CUDA_VISIBLE_DEVICES=0 python main.py --cfg cfgs/cifar10/tent.yaml
CUDA_VISIBLE_DEVICES=0 python main.py --cfg cfgs/cifar10/eta.yaml
CUDA_VISIBLE_DEVICES=0 python main.py --cfg cfgs/cifar10/eata.yaml
CUDA_VISIBLE_DEVICES=0 python main.py --cfg cfgs/cifar10/sar.yaml
CUDA_VISIBLE_DEVICES=0 python main.py --cfg cfgs/cifar10/pl.yaml
CUDA_VISIBLE_DEVICES=0 python main.py --cfg cfgs/cifar10/shot.yaml
```