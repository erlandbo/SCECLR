python main.py --metric cosine --criterion infonce --no-norm_mlp_layer --use_ffcv --use_fp16 --basedataset cifar100 --no-first_conv --no-first_maxpool --mlp_outfeatures 128 --mlp_hidden_features 1024 --batchsize 1024 --rho 0.99999 --epochs 1000 50 250 --alpha 0.25 --numworkers 20 --s_init 2.0
#python main.py --metric cauchy --criterion infonce --no-norm_mlp_layer --no-use_ffcv --use_fp16 --basedataset stl10_unlabeled --mlp_outfeatures 128 --mlp_hidden_features 1024 --batchsize 512 --rho 0.99999 --epochs 1000 50 250 --alpha 0.25 --numworkers 20 --s_init 2.0
