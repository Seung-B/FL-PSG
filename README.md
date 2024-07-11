# FL-OpenPSG

![clustering](https://github.com/Seung-B/FL-OpenPSG/assets/14955366/cdc892e9-9c9c-451c-a86f-53af9a8f81af)

## Environment Setting
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


## Dataset
### Benchmarks used in paper

### Benchmark Generation
You can generated benchmark using our method from ```Data_split.ipynb```.

There are detailed comments in notebook.

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
