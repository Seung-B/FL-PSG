import os

rounds = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 99]
model = 'motifs'   # motifs, vctree, imp, gpsnet
cluster_type = 'Super' # Random, Super
ratio = 5
distributions = ['Diri_10', 'Diri_1', 'Diri_02']#'nonIID' , 'Diri_1', 'Diri_10'
cluster_num = 5
TOTAL_CLIENT = 100

for r in rounds:
    for distribution in distributions:
        print(f"PYTHONPATH='.':$PYTHONPATH \
          python3 tools/fl_test.py \
          configs/{model}/panoptic_fpn_r50_fpn_1x_sgdet_psg.py \
          /home/hwan/sgg/data/FL_TRAINED_MODEL/mini_{model}/{cluster_type}-{distribution}-Cluster{cluster_num}_Ratio{ratio}_User{TOTAL_CLIENT}/Round-{r}.pth \
          --work-dir /home/hwan/sgg_result_Diri/mini_{model}/{cluster_type}-{distribution}-Cluster{cluster_num}_Ratio{ratio}_User{TOTAL_CLIENT} \
          --eval sgdet\n")

# for r in rounds:
#     print(f"PYTHONPATH='.':$PYTHONPATH \
#       python3 tools/fl_test.py \
#       configs/{model}/panoptic_fpn_r50_fpn_1x_sgdet_psg.py \
#       /home/hwan/sgg/FL_TRAINED_MODEL/{model}/{cluster_type}_Ratio{ratio}_User{TOTAL_CLIENT}/Round-{r}.pth \
#       --work-dir /home/hwan/sgg_result/{model}/{cluster_type}_Ratio{ratio}_User{TOTAL_CLIENT} \
#       --eval sgdet\n")
