


export CUDA_VISIBLE_DEVICES="1"



# cifar100
python main.py --method cclis --mem_type ring --dataset cifar100 --batch_size 10 --temp 0.5 --seed 0\
               --learning_rate 0.03 --mem_size 2000 --epochs 50 --start_epoch 100 --log_name erring --date 2025_04_16
