
# TEA: Test-time Energy Adaptation

the default model using WRN-28-10 for [RobustBench](https://github.com/RobustBench/robustbench).

**Usage**:

```python
CUDA_VISIBLE_DEVICES=1 python main.py --cfg cfgs/cifar10/source.yaml
CUDA_VISIBLE_DEVICES=1 python main.py --cfg cfgs/cifar10/energy.yaml
```
