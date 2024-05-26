cd ..

python run_parallel.py \
    --config_name lsun \
    --log_path lsun_bedroom/Adam/alexgda \
    --seeds 0 1 2 3 4 \
    --devices 1 2 3 4 5 6 7 \
    --lrs_G 0.0001 0.0003 0.0005 \
    --lrs_D 0.0001 0.0003 0.0005 \
    --gammas 0 1 1.2 1.5 2 4 \
    --deltas 1 1.2 1.5 2 4 \
    --num_exp_per_device 5