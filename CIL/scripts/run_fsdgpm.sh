
export CUDA_VISIBLE_DEVICES="3"

# cifar100
# python main.py --method er --mem_type ring --dataset cifar100 \
#                --batch_size 10 --learning_rate 0.03 --mem_size 0 --epochs 100


# cifar100
# python main.py --method fs-dgpm --mem_type ring --dataset cifar100 --batch_size 64 --seed 0\
#                --learning_rate 0.03 --mem_size 2000 --epochs 1 --log_name practice --date 2025_04_12\
#                 --sharpness --threshold 0.97 --thres_add 0.003 --mem_batch_size 125 --second_order --fsdgpm_method xdgpm
python main.py --method fs-dgpm --mem_type ring --dataset cifar100 --batch_size 64 --seed 0\
               --learning_rate 0.03 --mem_size 2000 --epochs 1 --log_name practice --date 2025_04_12\
                --sharpness --threshold 0.97 --thres_add 0.003 --mem_batch_size 10 --second_order --fsdgpm_method xdgpm