

export CUDA_VISIBLE_DEVICES="2"






python main.py --method cclis --mem_type ring --dataset cifar100 --batch_size 512 --seed 0 \
               --temp 0.5 --current_temp 0.2 --past_temp 0.1 --distill_type PRD --distill_power 0 \
               --learning_rate 1.0 --learning_rate_prototypes 0.01 --mem_size 2000 --epochs 100 --start_epoch 500 --log_name cclis_w/o_distill --date 2025_04_26

python main.py --method cclis --mem_type ring --dataset cifar100 --batch_size 512 --seed 1 \
               --temp 0.5 --current_temp 0.2 --past_temp 0.1 --distill_type PRD --distill_power 0 \
               --learning_rate 1.0 --learning_rate_prototypes 0.01 --mem_size 2000 --epochs 100 --start_epoch 500 --log_name cclis_w/o_distill --date 2025_04_26

python main.py --method cclis --mem_type ring --dataset cifar100 --batch_size 512 --seed 2 \
               --temp 0.5 --current_temp 0.2 --past_temp 0.1 --distill_type PRD --distill_power 0 \
               --learning_rate 1.0 --learning_rate_prototypes 0.01 --mem_size 2000 --epochs 100 --start_epoch 500 --log_name cclis_w/o_distill --date 2025_04_26

python main.py --method cclis --mem_type ring --dataset cifar100 --batch_size 512 --seed 3 \
               --temp 0.5 --current_temp 0.2 --past_temp 0.1 --distill_type PRD --distill_power 0 \
               --learning_rate 1.0 --learning_rate_prototypes 0.01 --mem_size 2000 --epochs 100 --start_epoch 500 --log_name cclis_w/o_distill --date 2025_04_26

python main.py --method cclis --mem_type ring --dataset cifar100 --batch_size 512 --seed 4 \
               --temp 0.5 --current_temp 0.2 --past_temp 0.1 --distill_type PRD --distill_power 0 \
               --learning_rate 1.0 --learning_rate_prototypes 0.01 --mem_size 2000 --epochs 100 --start_epoch 500 --log_name cclis_w/o_distill --date 2025_04_26





python main.py --method cclis --mem_type ring --dataset tiny-imagenet --batch_size 512 --seed 0 \
               --temp 0.5 --current_temp 0.2 --past_temp 0.1 --distill_type PRD --distill_power 0 \
               --learning_rate 1.0 --learning_rate_prototypes 0.01 --mem_size 2000 --epochs 50 --start_epoch 500 --log_name cclis_w/o_distill --date 2025_04_26

python main.py --method cclis --mem_type ring --dataset tiny-imagenet --batch_size 512 --seed 1 \
               --temp 0.5 --current_temp 0.2 --past_temp 0.1 --distill_type PRD --distill_power 0 \
               --learning_rate 1.0 --learning_rate_prototypes 0.01 --mem_size 2000 --epochs 50 --start_epoch 500 --log_name cclis_w/o_distill --date 2025_04_26

python main.py --method cclis --mem_type ring --dataset tiny-imagenet --batch_size 512 --seed 2 \
               --temp 0.5 --current_temp 0.2 --past_temp 0.1 --distill_type PRD --distill_power 0 \
               --learning_rate 1.0 --learning_rate_prototypes 0.01 --mem_size 2000 --epochs 50 --start_epoch 500 --log_name cclis_w/o_distill --date 2025_04_26

python main.py --method cclis --mem_type ring --dataset tiny-imagenet --batch_size 512 --seed 3 \
               --temp 0.5 --current_temp 0.2 --past_temp 0.1 --distill_type PRD --distill_power 0 \
               --learning_rate 1.0 --learning_rate_prototypes 0.01 --mem_size 2000 --epochs 50 --start_epoch 500 --log_name cclis_w/o_distill --date 2025_04_26

python main.py --method cclis --mem_type ring --dataset tiny-imagenet --batch_size 512 --seed 4 \
               --temp 0.5 --current_temp 0.2 --past_temp 0.1 --distill_type PRD --distill_power 0 \
               --learning_rate 1.0 --learning_rate_prototypes 0.01 --mem_size 2000 --epochs 50 --start_epoch 500 --log_name cclis_w/o_distill --date 2025_04_26





