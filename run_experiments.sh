python main.py --start_stage 1 --checkpoint_path /home/erlandbo/repo/SCECLR/logs/2024_03_01_12_30_23_sce_heavy-tailed/checkpoint_stage_1.pth --batchsize 1024 --rho 0.9999 --epochs 1000 500 100 --outfeatures 2 --criterion sce --alpha 0.25 --numworkers 20 --metric heavy-tailed --s_init 2.0 --lr_anneal linear_anneal --weight_decay 5.0e-4
