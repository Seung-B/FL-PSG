# FL-PSG

<p align="center">
  <img src="https://github.com/Seung-B/FL-OpenPSG/assets/14955366/cdc892e9-9c9c-451c-a86f-53af9a8f81af" align="center" width="80%">

  <p align="center">
  <a href="https://arxiv.org/abs/2412.10436" target='_blank'>
    <img src="https://img.shields.io/badge/Paper-arXiv%2024-b31b1b?style=flat-square">
  </a>


</p>


  <p align="center">
  <font size=5><strong>Benchmarking Federated Learning for Semantic Datasets: Federated Scene Graph Generation</strong></font>
    <br>
      <a href="https://github.com/Seung-B" target='_blank'>SeungBum Ha</a>,&nbsp;
      <a href="https://github.com/hwan-sig" target='_blank'>Taehwan Lee</a>,&nbsp;
      <a href="https://www.etri.re.kr/intro.html" target='_blank'>Jiyoun Lim</a>,&nbsp;
      <a href="https://sites.google.com/view/swyoon89" target='_blank'>Sung Whan Yoon</a>
    <br>
  MIIL Lab, Ulsan National Institute of Science & Technology | Electronics and Telecommunications Research Institute
  </p>
</p>

## Environment Setting


### Recommendatiaon - Docker Image
You can download this [Docker Image](https://hub.docker.com/r/sleepope/openpsg) to build the environment for this implementation.

### Manual Setting - [OpenPSG](https://github.com/Jingkang50/OpenPSG).

```
conda env create -f environment.yml
```
You shall manually install the following dependencies.
```
pip install mmcv-full==1.4.3 -f https://download.openmmlab.com/mmcv/dist/cu101/torch1.7.0/index.html
pip install openmim
mim install mmdet==2.20.0
pip install git+https://github.com/cocodataset/panopticapi.git
```

## Preparing Dataset - [OpenPSG](https://github.com/Jingkang50/OpenPSG)
[Datasets](https://entuedu-my.sharepoint.com/personal/jingkang001_e_ntu_edu_sg/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fjingkang001%5Fe%5Fntu%5Fedu%5Fsg%2FDocuments%2Fopenpsg%2Fdata&ga=1)  are provided. Please unzip the files if necessary.
```
├── data
│   ├── coco
│   │   ├── panoptic_train2017
│   │   ├── panoptic_val2017
│   │   ├── train2017
│   │   └── val2017
│   └── psg
│       ├── psg_train_val.json
│       ├── psg_val_test.json
│       └── ...
├── ...
```

### Federated Learning Benchmarks used in paper
This is a category (presented below) file used for split of user datasets.
```
├── data
│   ├── word_categorized.json
│   └── FL_DATA_SPLIT
```

.json files containing information on the data used by each user are included for each scenario.
```
├── ...
├── data/FL_DATA_SPLIT
│   ├── mini_Random
│   │   ├── User0.json
│   │   ├── User1.json
│   │   └── ...
│   ├── mini_Super_cluster_5_Diri_02
│   │   └── ...
│   ├── mini_Super_cluster_5_Diri_1
│   │   └── ...
│   ├── mini_Super_cluster_5_Diri_10
│   │   └── ...
│   ├── mini_Super_cluster_5_IID
│   │   └── ...
│   ├── mini_Super_cluster_5_nonIID
│   │   └── ...
│   └── mini_Super_cluster_5_Full.json
├── ...
```

### Benchmark Generation and PCA Visualization
You can generated benchmark using our method from ```data/data_split.ipynb```.

2D, 3D PCA Visualizations are supported!
(You need to install ```plotly``` package for 3D visualization.)

## Model Zoo (R/mR@100)
| Algorithms | CL           | Random         | Shard - IID    | Shard - non-IID  | Model Link |
|:----------:|--------------|----------------|----------------|------------------|------------|
|     IMP    | 18.37 / 7.11 | 14.46 / 3.56   | 14.45 / 3.65   | 13.06 / 2.68     | [link](https://drive.google.com/drive/folders/1cA7xH_14gxhLU9s_uCv5euTRbL8b4lVG?usp=sharing) |
|   MOTIFS   | 19.15 / 8.14 | 15.64 / 5.16   | 15.38 / 5.20   | 15.43 / 4.65     | [link](https://drive.google.com/drive/folders/1DR48e7wpavSJGP-eIwKcrydUAv83Wema?usp=sharing) |
|   VCTree   | 19.02 / 7.82 | 14.69 / 4.87   | 14.97 / 5.05   | 14.62 / 4.54     | [link](https://drive.google.com/drive/folders/1OA3ULh_I8q5L9p9T-YpU_kE-HS2UIFKQ?usp=sharing) |
|   GPS-Net  | 20.28 / 8.47 | 16.34 / 6.66   | 17.08 / 7.55   | 16.91 / 6.49     | [link](https://drive.google.com/drive/folders/1-khoMgN5Iuwt_YiJNnrS3hKOAriT93kG?usp=sharing) |

## Fedearted Learning - Train
When you perform training, you can edit or choose the parameters as follows.

- MODEL-NAME: [`imp, motifs, vctree, gps-net`]
- CONFIG-PATH: ```configs/#MODEL-NAME/panoptic_fpn_r50_fpn_1x_sgdet_psg.py```
- CLUSTER-TYPE: [`Super`, `Random`] 
- DISTRIBUTION: [`IID`, `nonIID`, `Diri_02`, `Diri_1`, `Diri_10`]

### Split
```PYTHONPATH=‘.’:$PYTHONPATH python3 tools/fl_train_mini.py {CONFIG-PATH} --model_name {MODEL-NAME} --job_name sgdet --n_rounds 100 --num_client 100 --selected_client 5 --cluster_type {CLUSTER-TYPE} --num_cluster 5 --distribution {DISTRIBUTION} ```

### Random
```PYTHONPATH=‘.’:$PYTHONPATH python3 tools/fl_train_mini.py {CONFIG-PATH} --model_name {MODEL-NAME} --job_name sgdet --n_rounds 100 --num_client 100 --selected_client 5 --cluster_type Random --num_cluster 5 ```


## Test
You have to change "MODEL_NAME" and "path/to/checkpoint.pth"
```
PYTHONPATH='.':$PYTHONPATH \
python tools/test.py \
  configs/"MODEL_NAME"/panoptic_fpn_r50_fpn_1x_sgdet_psg.py \
  "path/to/checkpoint.pth" \
  --eval sgdet
```


## Categories
### Object categories
|        Categories        |                           Subject &amp; Object                           |
|:------------------------:|:------------------------------------------------------------------------:|
|          Person          |                                  person                                  |
|         Vehicles         |        bicycle, car, motorcycle, airplane, bus, train, truck, boat       |
|       Road Objects       |       banner, traffic light, fire hydrant, stop sign, parking meter      |
|         Furniture        |     bench, chair, couch, potted plant, bed, dining table, rug-merged     |
|          Animals         |     bird, cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe    |
| Clothing and Accessories |                backpack, umbrella, handbag, tie, suitcase                |
|    Outdoor Activities    |        frisbee, skis, snowboard, sports ball, kite, baseball bat, baseball glove, skateboard, surfboard, tennis racket, playingfield, net |
|    Kitchen and Dining    |             bottle, wine glass, cup, fork, knife, spoon, bowl            |
|           Food           |            banana, apple, sandwich, orange, broccoli, carrot, hot dog, pizza, donut, cake, food-other-merged              |
|      Household Items     |      curtain, blanket, toilet, tv, laptop, mouse, remote, keyboard, cell phone, microwave, oven, toaster, sink, refrigerator, book, pillow, towel, clock, vase, scissors, teddy bear, hair drier, toothbrush |
|        Structures        |           bridge, house, tent, door-stuff, wall-other-merged, building-other-merged, pavement-merged, ceiling-merged, wall-brick, wall-stone, wall-tile, wall-wood, stairs, railroad, road, roof, floor-wood, platform, floor-other-merged, fence-merged    |
|          Nature          |        flower, fruit, gravel, river, sea, tree-merged, snow, sand, water-other, mountain-merged, grass-merged, dirt-merged, rock-merged, sky-other-merged                      |
|           Misc.          |       cardboard, counter, light, mirror-stuff, shelf, window-blind, window-other, cabinet-merged, table-merged, paper-merged         |

### Predicate categories (https://psgdataset.org/data_stats.html - Clear Predicate Definition)
|             Categories             | Predicate                                                                                                                                                                                                                                                                                                                                                       |
|:----------------------------------:|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|      Positional Relations (6)      | over, in front of, beside, on, in, attached to                                                                                                                                                                                                                                                                                                                  |
| Common Object-Object Relations (5) | hanging from, on the back of,falling off, going down, painted on                                                                                                                                                                                                                                                                                                |
|         Common Actions (31)        | walking on, running on, crossing, standing on, lying on, sitting on, leaning on, flying over, jumping over, jumping from, wearing, holding, carrying, looking at, guiding, kissing, eating, drinking, feeding, biting, catching, picking (grabbing), playing with, chasing, climbing, cleaning(washing, brushing), playing, touching, pushing, pulling, opening |
|          Human Actions (4)         | cooking, talking to, throwing (tossing), slicing                                                                                                                                                                                                                                                                                                                |
| Actions in Traffic Scene (4)       | driving, riding, parked on, driving on                                                                                                                                                                                                                                                                                                                          |
| Actions in Sports Scene (3)        | About to hit, kicking, swinging                                                                                                                                                                                                                                                                                                                                 |
| Interaction between Background (3) | entering, exiting, enclosing (surrounding, warping in)                                                                                                                                                                                                                                                                                                          |


## Thanks
Our implementation is based on https://github.com/Jingkang50/OpenPSG and https://github.com/open-mmlab/mmdetection.

|Algorithm	|# of model parameters|
|:-:|:-:|
|IMP|	32M|
|MOTIFS| 63M|
|VCTree|	59M|
|GPS-Net|	37M|

|Algorithm	|# of model parameters|
|:-:|:-:|
|IMP|	32M|
|MOTIFS| 63M|
|VCTree|	59M|
|GPS-Net|	37M|

|R/mR@100|	Shard IID|	Shard non-IID|
|:-:|:-:|:-:|
|IMP|	64(x 1)	|64(x 1)|
|MOTIFS|	63 (x 0.98)	|63 (x 0.98)|
|VCTree|	59 (x 0.92)|	59 (x 0.92)|
|GPS-Net|	37 (x 0.57)|	37 (x 0.57)|

||participation rate 5 (paper)||	participation rate 20 ||
|:-:|:-:|:-:|:-:|:-:|
| |Shard IID|	Shard non-IID|	Shard IID|	Shard non-IID|
|IMP|	14.45 / 3.65| 	13.06 / 2.68|	8.69 / 1.62|	8.22 / 1.66|
|MOTIFS|	15.38 / 5.20 |	15.43 / 4.65|	17.3 / 6.39	|16.83 / 6.23|
|VCTree|	14.97 / 5.05 |	14.62 / 4.54|	15.03 / 4.83|	14.92 / 4.73|
|GPS-Net|	17.08 / 7.55 |	16.91 / 6.49|	16.81 / 6.36	|16.66 / 6.04|



Full Table 3) Performance of FedAvgM

||FedAvg (paper)| | FedAvgM| |
|:-:| -| -| -|-| 
|Method|Shard IID|	Shard non-IID|	Shard IID	|Shard non-IID|
|IMP| 14.45 / 3.65| 	13.06 / 2.68|	13.61 / 3.52|	15.31 / 4.43|
|MOTIFS|	15.38 / 5.20| 	15.43 / 4.65|	17.65 / 6.55|	17.83 / 6.38|
|VCTree|	14.97 / 5.05| 	14.62 / 4.54|	17.75 / 6.74|	17.58 / 6.3|
|GPS-Net|	17.08 / 7.55| 	16.91 / 6.49|	18.94 / 7.9 |	18.68 / 6.52|

Full Table 2) Participation Ratio

||participation rate 5 (paper)||	participation rate 20 ||
|:-:|:-:|:-:|:-:|:-:|
| |Shard IID|	Shard non-IID|	Shard IID|	Shard non-IID|
|IMP|	14.45 / 3.65| 	13.06 / 2.68|	8.69 / 1.62|	8.22 / 1.66|
|MOTIFS|	15.38 / 5.20 |	15.43 / 4.65|	17.3 / 6.39	|16.83 / 6.23|
|VCTree|	14.97 / 5.05 |	14.62 / 4.54|	15.03 / 4.83|	14.92 / 4.73|
|GPS-Net|	17.08 / 7.55 |	16.91 / 6.49|	16.81 / 6.36	|16.66 / 6.04|
