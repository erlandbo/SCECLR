python main.py --outfeatures 128 --mlp_hidden_features 1024 --eval_epoch 10 --batchsize 512 --rho 0.99999 --epochs 501 50 250 --criterion sceclrv1 --alpha 0.25 --numworkers 10 --metric cauchy --s_init 2.0 --lr_anneal cosine_anneal --weight_decay 5.0e-4
python main.py --outfeatures 128 --mlp_hidden_features 1024 --eval_epoch 10 --batchsize 512 --rho 0.99999 --epochs 501 50 250 --criterion sceclrv1 --alpha 0.1 --numworkers 10 --metric cauchy --s_init 2.0 --lr_anneal cosine_anneal --weight_decay 5.0e-4
