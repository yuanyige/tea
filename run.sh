# CUDA_VISIBLE_DEVICES=1 python main.py --cfg cfgs/bn/pacs/sketch/energy_10_0003_20_01.yaml
# CUDA_VISIBLE_DEVICES=1 python main.py --cfg cfgs/bn/pacs/sketch/energy_10_0002_20_01.yaml
# CUDA_VISIBLE_DEVICES=1 python main.py --cfg cfgs/bn/pacs/sketch/energy_10_0001_20_01.yaml

# CUDA_VISIBLE_DEVICES=1 python main.py --cfg cfgs/gn/cifar10/energy_2_00005.yaml
# CUDA_VISIBLE_DEVICES=1 python main.py --cfg cfgs/gn/cifar10/energy_3_00001.yaml
# CUDA_VISIBLE_DEVICES=1 python main.py --cfg cfgs/gn/cifar100/energy_2_00005.yaml
# CUDA_VISIBLE_DEVICES=1 python main.py --cfg cfgs/gn/cifar100/energy_4_00001.yaml

#
# CUDA_VISIBLE_DEVICES=3 python main.py --cfg cfgs/bn/pacs/art/art2sk_energy-vis.yaml
CUDA_VISIBLE_DEVICES=2 python main.py --cfg cfgs/bn/pacs/sketch/sk2sk_energy-vis.yaml
#CUDA_VISIBLE_DEVICES=1 python main.py --cfg cfgs/bn/pacs/sketch/sk2sk_energy-vis.yaml