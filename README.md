
<font size='5'>**TransLandSeg: A Transfer Learning Approach for
Landslide Semantic Segmentation Based on Vision
Foundation Model**</font>

<!-- [Yuan Hu](https://scholar.google.com.sg/citations?user=NFRuz4kAAAAJ&hl=zh-CN), Jianlong Yuan, Congcong Wen, Xiaonan Lu, [Xiang Li☨](https://xiangli.ac.cn) -->

Changhong Hou, [Junchuan Yu☨](https://github.com/JunchuanYu), Daqing Ge, Liu Yang, Laidian Xi, Yunxuan Pang, and Yi Wen

☨corresponding author

Refer to our [paper](http://arxiv.org/abs/2403.10127) for more details.

<!-- <a href='https://rsgpt.github.io'><img src='https://img.shields.io/badge/Project-Page-Green'></a>  -->
<!-- <a href='https://arxiv.org/abs/2307.15266'><img src='https://img.shields.io/badge/Paper-Arxiv-red'></a> -->

## Updates
* **[2024.3.3]** Paper submission. 
* **[2024.3.15]** Dataset uploaded.
* **[2024.4.20]** Code realease.

## Dataset
* [Landslide4Sense](https://github.com/iarai/Landslide4Sense-2022): contains 3799 training samples.
* [Bijie Landslide dataset](http://gpcv.whu.edu.cn/data/Bijie_pages.html): contains 770 landslide images within Bijie City in northwestern Guizhou Province, China. 
  + Save the file in your download directory:
    + `/data/{Bijie,Landslide4Sense}/{image,label}`

## Code
![](https://dunazo.oss-cn-beijing.aliyuncs.com/blog/pic1.jpg)

Structure of the proposed TransLandSeg and Segment Anything Model (SAM)

Click the links below to download the checkpoint for the corresponding model type.

- `ViT-L SAM model`: [ViT-L SAM model.](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth)
  + Save the file in your download directory:
    + `/pretrained/sam_vit_l_0b3195.pth`

- `TransLandSeg model`: [TransLandSeg model.](https://pan.baidu.com/s/1ipFqbnh1VqkAqZaGY9v80A?pwd=93g6)
  + Save the file in your download directory:
    + `/checkpoint/{Bijie.pth.tar,Landslide4Sense.pth.tar}`

+ The supporting library information of the code is shown below:
<center>

|Package                    |Version|
|:----:  |:----: |
| GDAL                      |3.6.2|
| h5py                      |3.9.0|
| matplotlib                |3.7.2|
| numpy                     |1.24.1|
| opencv-python             |4.8.0.74|
| scipy                     |1.10.1|
| tensorboard               |2.10.1|
| tensorboardX              |2.6.2.2|
| torch                     |1.12.1|
| torchsummary              |1.5.1|
| torchvision               |0.13.1|
| tqdm                      |4.65.0|

</center>

## Acknowledgement
+ [SAM](https://segment-anything.com). A new vision foundation model from Meta AI.
+ [Heywhale](https://www.heywhale.com/home). Provided the arithmetic platform for this work.

If you're using TransLandSeg in your research or applications, please cite using this BibTeX:

```bibtex
@article{
  title={TransLandSeg: A Transfer Learning Approach for
Landslide Semantic Segmentation Based on Vision
Foundation Model},
  author={Changhong Hou, Junchuan Yu*, Daqing Ge, Liu Yang, Laidian Xi, Yunxuan Pang, and Yi Wen}
  year={2024}
}
```

