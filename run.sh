# python mymain.py --dataset MUTAG --subgraph_size 4 --epoch 200 --num_layer 4 --num_mlp_layer 2 --lr 0.00001 --batch_size 64 --device 1 --multi_head
# python mymain.py --dataset NCI1 --subgraph_size 5 --epoch 300 --num_layer 4 --num_mlp_layer 2 --lr 0.0001 --batch_size 64 --multi_head
python mymain.py --dataset PROTEINS --subgraph_size 5 --epoch 300 --num_layer 5 --num_mlp_layer 2 --lr 0.0001 --batch_size 64 --multi_head --device 1