python main.py --dataset film --epoch_num 50 --lr 0.0001 --moving_average_decay 0.95 --dimension 4096 --sample_size 5 --wd2 1e-05 --num_MLP 2 --tau 1
python main.py --dataset texas --epoch_num 50 --lr 0.0001 --moving_average_decay 0.95 --dimension 2048 --sample_size 5 --wd2 1e-05 --num_MLP 1 --tau 0.75
python main.py --dataset squirrel  --epoch_num 40  --lr 0.0001 --moving_average_decay 0.97 --dimension 8192 --sample_size 10 --wd2 1e-05 --num_MLP 1 --tau 0.25 
python main.py --dataset wisconsin --epoch_num 50 --lr 0.0001 --moving_average_decay 0.97 --dimension 4096 --sample_size 5 --wd2 1e-05 --num_MLP 1 --tau 0.5
python main.py --dataset crocodile --epoch_num 50 --lr 0.0001 --moving_average_decay 0.97 --dimension 128 --sample_size 5 --wd2 1e-05 --num_MLP 1 --tau 0.5
python main.py --dataset chameleon --epoch_num 30 --lr 0.0001 --moving_average_decay 0.999 --dimension 4096 --sample_size 5 --wd2 1e-05 --num_MLP 1 --tau 0.5
python main.py --dataset cornell --epoch_num 50 --lr 0.0001 --moving_average_decay 0.95 --dimension 4096 --sample_size 5 --wd2 1e-05 --num_MLP 1 --tau 0.5


python ds.py --dataset texas
python ds.py --dataset wisconsin --dimension 4096 --num_MLP 1
python ds.py --dataset cornell --dimension 4096 --num_MLP 1
python ds.py --dataset chameleon --dimension 2048 --num_MLP 1
python ds.py --dataset squirrel --dimension 512 --num_MLP 1
python ds.py --dataset crocodile --dimension 128 --num_MLP 1
python ds.py --dataset film --dimension 4096 --num_MLP 2


python main.py --dataset chameleon --epoch_num 30 --lr 0.0001 --moving_average_decay 0.999 --dimension 2048 --sample_size 5 --wd2 1e-05 --num_MLP 1 --tau 0.5
python ds.py --dataset chameleon --dimension 1024 --num_MLP 1

python main.py --dataset squirrel  --epoch_num 40  --lr 0.0001 --moving_average_decay 0.999 --dimension 2048 --sample_size 5 --wd2 1e-05 --num_MLP 1 --tau 0.25 
python ds.py --dataset squirrel --dimension 2048 --num_MLP 1