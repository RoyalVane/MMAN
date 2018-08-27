
## MMAN
This is the code for "Macro-Micro Adversarial Network for Human Parsing" in ECCV2018. [Paper link](https://arxiv.org/abs/1807.08260)

By Yawei Luo, Zhedong Zheng, Liang Zheng, Tao Guan, Junqing Yu<sup>*</sup> and Yi Yang. 
###### <sup>*</sup> Corresponding Author: <yjqing@hust.edu.cn>

The proposed framework is capable of producing competitive parsing performance compared with the state-of-the-art methods, i.e., mIoU=46.81% and 59.91% on LIP and
PASCAL-Person-Part, respectively. On a relatively small dataset PPSS, our pre-trained model demonstrates impressive generalization ability.


## Prerequisites
- Python 3.6
- GPU Memory >= 4G
- Pytorch 0.3.1
- Visdom


## Getting started
Clone MMAN source code

Download [The LIP Dataset]( https://drive.google.com/open?id=1SlvucF37ApWCQjmdCYQ8i9yoUHNvFMiC )

The folder is structured as follows:
```
├── MMAN/
│   ├── data/                 	/* Files for data processing  		*/
│   ├── model/                 	/* Files for model    			*/
│   ├── options/          	/* Files for options    		*/
│   ├── ...			/* Other dirs & files 			*/
└── Human/
    ├── train_LIP_A/		/* Training set: RGB images		*/
    ├── train_LIP_B/		/* Training set: GT labels		*/
    ├── test_LIP_A/		/* Testing set: RGB images		*/
    └── test_LIP_B/		/* Testing set: GT labels		*/
```


## Train
### Open a visdom server
```bash
python -m visdom.server
```

### Train a model
```bash
python train.py --dataroot ../Human --dataset LIP --name Exp_0 --output_nc 20 --gpu_ids 0 --pre_trained --loadSize 286 --fineSize 256
```
`--dataroot` The root of the training set.

`--dataset` The name of the training set.

`--name` The name of output dir. 

`--output_nc` The number of classes. For LIP, it equals to 20. 

`--gpu_ids` Which gpu to run.

`--pre_trained` Using ResNet101 model pretrained on Imagenet.

`--loadSize` Resize training images into 286 * 286.

`--fineSize` Randomly crop 256 * 256 patch from a 286 * 286 image.

Enjoy the training process in http://XXX.XXX.XXX.XXX:8097/ , where XXX is your server IP address.


## Test
### Use trained model to parse human images
```bash
python test.py --dataroot ../Human --dataset LIP --name Exp_0 --gpu_ids 0 --which_epoch 30 --how_many 10000 --output_nc 20 --loadSize 256
```
`--dataroot` The root of the testing set.

`--dataset` The name of the testing set.

`--name` The dir name of trained model.

`--gpu_ids` Which gpu to run.

`--which_epoch` Select the i-th model.

`--how_many` Total number of test images.

`--output_nc` The number of classes. For LIP, it equals to 20. 

`--loadSize` Resize testing images into 256 * 256.

### New! Pretrained models are available via this link:
[Google Drive](https://drive.google.com/open?id=1pLFXIf8o3Jpq-w4_D4l8yWE9Q-1-TVLh)

## Qualitative results
Trained on ``LIP train_set`` -> Tested on ``LIP val_set``
![](https://github.com/RoyalVane/MMAN/blob/master/jpg/LIP.JPG)

Trained on ``LIP train_set`` -> Tested on ``Market1501``
![](https://github.com/RoyalVane/MMAN/blob/master/jpg/Market1501.JPG)

## Citation
If you find MMAN useful in your research, please consider citing:
```
@inproceedings{luo2018macro,
	title={Macro-Micro Adversarial Network for Human Parsing},
	author={Luo, Yawei and 
		Zheng, Zhedong and 
		Zheng, Liang and 
		Guan, Tao and 
		Yu, Junqing and 
		Yang, Yi},
	booktitle ={ECCV},
	year={2018}
}

```

## Related Repos
1. [Pedestrian Alignment Network](https://github.com/layumi/Pedestrian_Alignment)
2. [pix2pix](https://github.com/phillipi/pix2pix)
3. [Market-1501](http://www.liangzheng.org/Project/project_reid.html)

