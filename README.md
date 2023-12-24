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

| settings/datasets   |   | PoisonNoiseCIFAR10                |         |
|---------------------|---|-----------------------------------|---------|
| /                   | / | stage2(loops)      \|  stage3 acc |
| noise_rate=0<br/>   |   | 4          \|                     |         |
| noise_rate=0.2<br/> |   | 4\|14                             |
|                     |   |                                   |         |
|                     |   |                                   |         |
|                     |   |                                   |         |

# common problem

1. sometimes ue need much time to generate ,you can change the random seed to have a try.

