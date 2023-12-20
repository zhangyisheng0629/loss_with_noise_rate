# loss_with_noise_rate
# preprocess
change the code in `data/data_dir` to your own dataset path.
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
you can change the settings in the `configs` folder.

# Stage2: Generate samplewise perturbation(ue) and
# Stage3: Train with ue

```shell
python ue.py --conf_path=./configs/ue/resnet18_cifar10_thre.yaml
 ```



# 


