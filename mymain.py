import os
import time
import argparse

from numpy.lib.function_base import append
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from models.graphcnn import GraphCNN
import matplotlib.pyplot as plt
from datetime import datetime
import pytz
import copy

from myutil import load_data, load_hyper_data, separate_data, separate_data_no_val

criterion = nn.CrossEntropyLoss()


def train(args, model, device, train_graphs, hyper_train_graphs, train_motif2A, optimizer, epoch):
    model.train()

    total_iters = args.iters_per_epoch
    # pbar = tqdm(range(total_iters), unit='batch')

    loss_accum = 0
    for i in range(total_iters):
        selected_idx = np.random.permutation(len(train_graphs))[:args.batch_size]

        batch_graph = [train_graphs[idx] for idx in selected_idx]
        hyper_batch_graph = [hyper_train_graphs[idx] for idx in selected_idx]
        batch_motif2A = [train_motif2A[idx] for idx in selected_idx]

        output = model(batch_graph, hyper_batch_graph, batch_motif2A)

        labels = torch.LongTensor([graph.label for graph in batch_graph]).to(device)

        # compute loss
        loss = criterion(output, labels)

        # backprop
        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        loss = loss.detach().cpu().numpy()
        loss_accum += loss

        # report
        # pbar.set_description('epoch: %d' % (epoch))

    average_loss = loss_accum / total_iters
    print("epoch: %d \t loss training: %f" % (epoch,average_loss))

    return average_loss


###pass data to model with minibatch during testing to avoid memory overflow (does not perform backpropagation)
def pass_data_iteratively(model, graphs, hyper_graphs, test_motif2A, minibatch_size=64):
    model.eval()
    output = []
    idx = np.arange(len(graphs))
    for i in range(0, len(graphs), minibatch_size):
        sampled_idx = idx[i:i + minibatch_size]
        if len(sampled_idx) == 0:
            continue
        output.append(model([graphs[j] for j in sampled_idx], [hyper_graphs[j] for j in sampled_idx],
                            [test_motif2A[j] for j in sampled_idx]).detach())
    return torch.cat(output, 0)


def test(model, device, graphs, hyper_graph,motif2A):
    model.eval()
    output = pass_data_iteratively(model, graphs, hyper_graph, motif2A)
    pred = output.max(1, keepdim=True)[1]
    labels = torch.LongTensor([graph.label for graph in graphs]).to(device)
    correct = pred.eq(labels.view_as(pred)).sum().cpu().item()
    acc_train = correct / float(len(graphs))

    # output = pass_data_iteratively(model, test_graphs, hyper_test_graphs, test_motif2A)
    # pred = output.max(1, keepdim=True)[1]
    # labels = torch.LongTensor([graph.label for graph in test_graphs]).to(device)
    # correct = pred.eq(labels.view_as(pred)).sum().cpu().item()
    # acc_test = correct / float(len(test_graphs))

    return acc_train


def main():
    # Training settings

    # Note: Hyper-parameters need to be tuned in order to obtain results reported in the paper.
    parser = argparse.ArgumentParser(
        description='PyTorch graph convolutional neural net for whole-graph classification')
    parser.add_argument('--dataset', type=str, default='MUTAG',
                        help='name of dataset (default: MUTAG)')
    parser.add_argument('--subgraph_size', type=int, default=5,
                        help='number of nodes in each subgraph including centor one.(default: 5)')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate (default: 0.01)')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='input batch size for training (default: 32)')
    parser.add_argument('--iters_per_epoch', type=int, default=50,
                        help='number of iterations per each epoch (default: 50)')
    parser.add_argument('--epochs', type=int, default=350,
                        help='number of epochs to train (default: 350)')
    parser.add_argument('--multi_head', action="store_true",default=False,
                        help='use independent attention parameters in each layer')
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed for splitting the dataset into 10 (default: 0)')
    parser.add_argument('--fold_idx', type=int, default=10,
                        help='the index of fold in 10-fold validation. Should be less then 10.')
    parser.add_argument('--num_layers', type=int, default=5,
                        help='number of layers INCLUDING the input one (default: 5)')
    parser.add_argument('--num_mlp_layers', type=int, default=2,
                        help='number of layers for MLP EXCLUDING the input one (default: 2). 1 means linear model.')
    parser.add_argument('--hidden_dim', type=int, default=64,
                        help='number of hidden units (default: 64)')
    parser.add_argument('--final_dropout', type=float, default=0.5,
                        help='final layer dropout (default: 0.5)')
    parser.add_argument('--graph_pooling_type', type=str, default="sum", choices=["sum", "average"],
                        help='Pooling for over nodes in a graph: sum or average')
    parser.add_argument('--neighbor_pooling_type', type=str, default="sum", choices=["sum", "average", "max"],
                        help='Pooling for over neighboring nodes: sum, average or max')
    parser.add_argument('--learn_eps', action="store_true",
                        help='Whether to learn the epsilon weighting for the center nodes. Does not affect training accuracy though.')
    parser.add_argument('--degree_as_tag', action="store_true",default=False,
                        help='let the input node features be the degree of nodes (heuristics for unlabeled graph)')
    parser.add_argument('--filename', type=str, default="",
                        help='output file')
    
    args = parser.parse_args()

    check_path = os.path.join('checkpoints', args.dataset)
    if not os.path.exists(check_path):
        os.mkdir(check_path)
    if args.filename=='':
        filename=os.path.join(check_path,time.strftime("%m%d_%H%M"))

    filepath_paras = os.path.join(filename, 'params.txt')
    if not os.path.exists(filename):
        os.mkdir(filename)
    with open(filepath_paras, 'w') as f:
        f.write(str(vars(args)))
        f.close()

    # set up seeds and gpu device
    torch.manual_seed(0)
    np.random.seed(0)
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    # print(device)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    graphs, num_classes, motf2As, graph_labels = load_data(args.dataset,args.subgraph_size)
    hyper_graphs = load_hyper_data(args.dataset,args.subgraph_size,graph_labels)

    best_model_test_accs = []
    best_epochs=[]
    most_frequency_tests=[]
    val_flag=False
    for i in range(args.fold_idx):
        ##10-fold cross validation. Conduct an experiment on the fold specified by args.fold_idx.
        if val_flag:
            train_graphs, hyper_train_graphs, train_motif2A, val_graphs, hyper_val_graphs, val_motif2A, test_graphs, hyper_test_graphs, test_motif2A = separate_data(graphs, hyper_graphs, motf2As, args.seed, i)
            print('train:'+str(len(train_graphs))+'\n val:'+str(len(val_graphs))+'\n test:'+str(len(test_graphs))+'\n')
        
        else:
            train_graphs, hyper_train_graphs, train_motif2A, test_graphs, hyper_test_graphs, test_motif2A \
                = separate_data_no_val(graphs, hyper_graphs, motf2As, args.seed, i)
            

        model = GraphCNN(args.num_layers, args.num_mlp_layers, train_graphs[0].node_features.shape[1],
                        hyper_train_graphs[0].node_features.shape[1],
                         args.hidden_dim, num_classes, args.final_dropout, args.learn_eps,
                         args.graph_pooling_type, args.neighbor_pooling_type, device,args.multi_head).to(device)

        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

        max_acc_val = 0
        best_epoc=0
        model_best=copy.deepcopy(model)
        avg_losses,acc_trains,acc_vals,acc_tests=[],[],[],[]
        for epoch in range(1, args.epochs + 1):
            scheduler.step()
    
            avg_loss = train(args, model, device, train_graphs, hyper_train_graphs, train_motif2A, optimizer, epoch)
            acc_train= test(model, device, train_graphs, hyper_train_graphs, train_motif2A)
            avg_losses.append(avg_loss)
            acc_trains.append(acc_train)
            
            if val_flag:
                acc_val=test(model,device, val_graphs, hyper_val_graphs, val_motif2A)
                acc_vals.append(acc_val)
                if acc_val > max_acc_val:
                    max_acc_val = acc_val
                    model_best=copy.deepcopy(model)
                    best_epoc=epoch
                print("%s fold: %d//%d  accuracy train: %.3f val: %.3f max_val: %.3f" % 
                      (args.dataset,args.epoch, i, acc_train, acc_val,max_acc_val))
                
            else:
                acc_test=test(model,device, test_graphs, hyper_test_graphs, test_motif2A)
                acc_tests.append(acc_test)
                print("%s fold: %d accuracy train: %.3f test: %.3f " % (args.dataset, i, acc_train, acc_test))


        epoches=range(0,args.epochs)
        plt.figure(i*2)
        plt.plot(epoches, acc_trains)
        
        acc_in_file=acc_tests
        if val_flag:
            acc_test_with_best_model=test(model_best,device, test_graphs, hyper_test_graphs, test_motif2A)
            acc_in_file=acc_vals           
            plt.plot(epoches, [acc_test_with_best_model] * args.epochs)
            best_model_test_accs.append(acc_test_with_best_model)
            best_epochs.append(best_epoc)
        else:
            most_frequency=max(set(acc_tests), key=acc_tests.count)
            plt.plot(epoches, [most_frequency] * args.epochs)
            most_frequency_tests.append(most_frequency)
            
        plt.plot(epoches, acc_in_file)
        plt.savefig(os.path.join(filename,str(i) + '_acc.jpg'))
        # plt.figure(i*2+1)
        # plt.plot(epoches, avg_losses)
        # plt.savefig(os.path.join(filename,str(i) + '_loss.jpg'))
        
        filepath_process=os.path.join(filename, str(i) + '.txt')
        with open(filepath_process, 'w+') as f:
            for epoch in range(1, args.epochs + 1):
                f.write("%f %f %f %f" % (epoch,avg_losses[epoch-1],
                                            acc_trains[epoch-1], acc_in_file[epoch-1]))
                f.write("\n")

    filepath_acc=os.path.join(filename, 'final_accs.txt')
    f=open(filepath_acc,'w')
    acc_in_final_file=np.array(most_frequency_tests)
    if val_flag:   
        acc_in_final_file=np.array(best_model_test_accs)
        f.write('best epochs:'+str(best_epochs)+'\n')
        
    f.write('accs:'+str(acc_in_final_file)+'\n')
    f.write('mean:'+str(np.mean(acc_in_final_file))+'\n')
    f.write('std:'+str(np.std(acc_in_final_file))+'\n')
    f.write(str(args.dataset)+'\t batch_size:'+str(args.batch_size)+
            '\t epochs:'+str(args.epochs)+'\t num_layer:'+str(args.num_layers)+
            '\t num_mlp:'+str(args.num_mlp_layers)+'\t hidden_dim:'+str(args.hidden_dim)+
            '\t subgraph_size:'+str(args.subgraph_size)+'\t multi_head:'+str(args.multi_head)+'\n')
 
    t = datetime.fromtimestamp(int(time.time()),
    pytz.timezone('Asia/Shanghai')).strftime('%Y-%m-%d %H:%M:%S %Z%z')
    f.write(str(t))
    # f.write('train:'+str(len(train_graphs))+'\t val:'+str(len(val_graphs))+'\t test:'+str(len(test_graphs))+'\n')
    f.close()
    return


if __name__ == '__main__':
    main()

    # datasets = ['MUTAG']
    # param_dicts = {
    #     'MUTAG':{'epoch':300,'num_layer':3,'num_mlp_layer':1,'lr':0.01,'batch_size':32,'degree_as_tag':False,'subgraph_size':3},
    #     'PTC_MR':{'epoch':400,'num_layer':5,'num_mlp_layer':3,'lr':0.01,'batch_size':32,'degree_as_tag':False},
    #     'PTC_MM':{'epoch':400,'num_layer':5,'num_mlp_layer':3,'lr':0.01,'batch_size':32,'degree_as_tag':False},
    #     'PTC_FR':{'epoch':400,'num_layer':5,'num_mlp_layer':3,'lr':0.01,'batch_size':32,'degree_as_tag':False},
    #     'PTC_FM':{'epoch':400,'num_layer':5,'num_mlp_layer':3,'lr':0.01,'batch_size':32,'degree_as_tag':False},
    #     'COX2': {'epoch': 400, 'num_layer': 3, 'num_mlp_layer':1, 'lr': 1e-2,'batch_size':8,'degree_as_tag':False},
    #     'NCI1':{'epoch':800,'num_layer':5,'num_mlp_layer':3,'lr':0.01,'batch_size':32,'degree_as_tag':False},
    #     'PROTEINS': {'epoch': 600, 'num_layer': 5, 'num_mlp_layer': 1, 'lr': 0.01, 'batch_size': 8,'degree_as_tag':False},
    #     'IMDB-MULTI': {'epoch': 600, 'num_layer': 3, 'num_mlp_layer': 3, 'lr': 0.01,'batch_size':8,'degree_as_tag':True},
    #     'IMDB-BINARY':{'epoch':800,'num_layer':5,'num_mlp_layer':3,'lr':0.01,'batch_size':32,'degree_as_tag':True},
    #     'Synthetic': {'epoch': 300, 'num_layer': 3, 'num_mlp_layer': 1, 'lr': 0.01, 'batch_size': 32, 'degree_as_tag': False}
    # }

    # accs, mean_accs, std_accs = [], [], []

    # for dataset in datasets:
    #     acc, mean_acc, std_acc = main(dataset,param_dicts[dataset]['epoch'],
    #                                   param_dicts[dataset]['num_layer'],
    #                                   param_dicts[dataset]['num_mlp_layer'],
    #                                   param_dicts[dataset]['lr'],
    #                                   param_dicts[dataset]['batch_size'],
    #                                   filename='./res/{}_',
    #                                   degree_as_tag=param_dicts[dataset]['degree_as_tag'],
    #                                   subgraph_size=param_dicts[dataset]['subgraph_size']
    #                                   )
    #     accs.append(acc)
    #     mean_accs.append(mean_acc)
    #     std_accs.append(std_acc)
    #     plt.close('all')



