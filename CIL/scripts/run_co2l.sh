# cifar10
# python main.py --method co2l --mem_type ring --dataset cifar10 --batch_size 512 \
#                --current_temp 0.2 --past_temp 0.01 --distill_power 1.0

# cifar100
python main.py --method co2l --mem_type ring --dataset cifar100 --batch_size 512 --epochs 100 --start_epoch 500 \
               --current_temp 0.2 --past_temp 0.01 --distill_power 1.0