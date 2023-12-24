# loss_with_noise_rate

# Stage0: preprocess

change the code in `data/data_dir` to your own datasets path.

# Stage1: Select noise samples

## Cifar10

### select with topk rate

```shell
python sele.py --conf_path=./configs/select/resnet18_cifar10.yaml
 ```

### select with threshold

```shell
python sele.py --conf_path=./configs/select/resnet18_cifar10_thre.yaml
 ```

you can change the settings for different experiments in the `configs/` folder.

# Stage2: Generate samplewise perturbation(ue)

```shell
python ue.py --conf_path=./configs/ue/resnet18_cifar10_thre.yaml
 ```

# Stage3: Train with ue

```shell
python train_with_ue.py --conf_path=./configs/train_with_ue/resnet18_cifar10_thre.yaml
 ```

# Results

| stage                        | settings/results | NoiseCIFAR10    | NoiseSVHN  |
|------------------------------|------------------|-----------------|------------|
| **stage1<br />select**       | noise_rate       | 0.2             | 0.2        |
| -                            | threshold        | 2.5             | 2.5        |
| -                            | recall           | 0.85            | 0.88       |
| **stage2<br/>ue**            | attack           | PGD             | PGD        |
| -                            | -                | alpha=0.8       | alpha=0.8  |
| -                            | -                | steps=20        | steps=20   |
| -                            | -                | eps=8           | eps=8      |
| -                            | ue_attack        | samplewise      | samplewise |
| **stage3<br/>train_with_ue** | accuracy         | 80% / 30% / 40% | about 12%  |

# common problem

1. sometimes ue need much time to generate ,you can change the random seed to have a try.

