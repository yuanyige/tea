
# TEA: Test-time Energy Adaptation

The default model using trained WRN-28-10 from [RobustBench](https://github.com/RobustBench/robustbench).


**Usage**:

```python
CUDA_VISIBLE_DEVICES=1 python main.py --cfg cfgs/cifar10/energy.yaml
```

**other baseline support**
- Source: model without any adaptation
- BN
- TENT
- ETA
- EATA
- SAR
- SHOT
- PL