import argparse
import datetime
import itertools
import subprocess
import time
import torch
import numpy as np
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch
from typing import Iterable, Optional
from timm.utils import ModelEmaV2
from engine import AverageMeter, dispatch_clip_grad
torch.set_default_dtype(torch.float32)

import os
from logger import FileLogger
from pathlib import Path

from datasets.SuperCon import SC
from features.process_data import get_Path,splitdata
from features.identity_disorder import classify

# AMP
from contextlib import suppress
from timm.utils import NativeScaler

import nets
from nets import model_entrypoint

from timm.utils import ModelEmaV2
from timm.scheduler import create_scheduler
from optim_factory import create_optimizer

from engine import AverageMeter, dispatch_clip_grad

# distributed training
import utils
import warnings

warnings.filterwarnings('ignore')
ModelEma = ModelEmaV2


class Expert(torch.nn.Module):
    """An expert model, which is a GNN in our case."""
    def __init__(self, gnn_model):
        super(Expert, self).__init__()
        self.gnn = gnn_model

    def forward(self, data):
        return self.gnn(data.x, data.edge_occu, data.edge_src, data.edge_dst, data.edge_vec, data.edge_attr, data.edge_num, data.batch)

    def get_graph_embedding(self, data):
        return self.gnn.get_graph_embedding(data.x, data.edge_occu, data.edge_src, data.edge_dst, data.edge_vec, data.edge_attr, data.edge_num, data.batch)


class GatingNetwork(torch.nn.Module):
    """A gating network that selects experts."""
    def __init__(self, input_dim, num_experts):
        super(GatingNetwork, self).__init__()
        self.layer = torch.nn.Linear(input_dim, num_experts)

    def forward(self, x):
        logits = self.layer(x)
        return torch.nn.functional.softmax(logits, dim=1)


class MoE_GNN(torch.nn.Module):
    """A Mixture of Experts model with GNN experts."""
    def __init__(self, num_experts, k, gnn_model_creator, model_args):
        super(MoE_GNN, self).__init__()
        self.num_experts = num_experts
        self.k = k
        
        # Create a single GNN to extract graph embeddings for the router
        # This GNN is not trained, it's just for getting the embeddings.
        # A more advanced implementation could train this "stem" as well.
        self.embedding_gnn = gnn_model_creator(**model_args)
        
        # The input dimension for the gating network is the number of scalar features (0e) 
        # in the GNN's output representation.
        input_dim = 0
        for mul, ir in self.embedding_gnn.irreps_feature:
            if ir.l == 0 and ir.p == 1:  # This checks for '0e' (scalar) irreps
                input_dim += mul
        
        if input_dim == 0:
            raise ValueError("The GNN's feature representation has no scalar (0e) components for the gating network.")

        self.gating_network = GatingNetwork(input_dim, num_experts)
        
        self.experts = torch.nn.ModuleList([Expert(gnn_model_creator(**model_args)) for _ in range(num_experts)])

    def forward(self, data):
        # Pass all tensors from the Batch object to the embedding GNN
        graph_embeddings = self.embedding_gnn.get_graph_embedding(
            data.x, data.edge_occu, data.edge_src, data.edge_dst, 
            data.edge_vec, data.edge_attr, data.edge_num, data.batch
        )
        
        routing_weights = self.gating_network(graph_embeddings)
        
        top_k_weights, top_k_indices = torch.topk(routing_weights, self.k, dim=1)
        
        # Normalize the top-k weights
        norm_top_k_weights = torch.nn.functional.softmax(top_k_weights, dim=1)
        
        # Efficient batching for experts
        final_output = torch.zeros_like(data.y, dtype=torch.float)
        
        # Flatten the batch and top-k dimensions
        batch_size = data.num_graphs
        flat_top_k_indices = top_k_indices.flatten()
        
        # Create a combined batch index
        batch_idx_repeated = torch.arange(batch_size, device=data.x.device).repeat_interleave(self.k)
        
        # This will store the outputs from the experts
        expert_outputs = torch.zeros(batch_size * self.k, 1, device=data.x.device)

        # Iterate through each expert and process a mini-batch of all data routed to it
        for i in range(self.num_experts):
            # Find which inputs in the flattened list are routed to this expert
            mask = (flat_top_k_indices == i)
            if mask.any():
                # Get the original batch indices for these inputs
                routed_batch_indices = batch_idx_repeated[mask]
                
                # Create a sub-batch for the current expert
                sub_data_list = [data.get_example(idx.item()) for idx in routed_batch_indices]
                sub_batch = Batch.from_data_list(sub_data_list).to(data.x.device)
                
                # Run the expert on the sub-batch
                output = self.experts[i](sub_batch)
                
                # Store the output
                expert_outputs[mask] = output

        # Reshape and apply weights
        expert_outputs = expert_outputs.view(batch_size, self.k, 1)
        weighted_outputs = expert_outputs.squeeze(-1) * norm_top_k_weights
        final_output = torch.sum(weighted_outputs, dim=1, keepdim=True)

        return final_output, routing_weights


def load_balancing_loss(routing_weights, num_experts):
    """
    Computes the load balancing loss for the MoE model.
    This loss encourages the gating network to distribute inputs evenly across all experts.
    """
    # routing_weights: (batch_size, num_experts)
    
    # f_i: Fraction of items routed to expert i. We use a soft version from the routing weights.
    f_i = torch.mean(routing_weights, dim=0)
    
    # P_i: Average routing probability for expert i
    P_i = torch.mean(routing_weights, dim=0)
    
    # Following some implementations, the loss is the dot product of these two vectors,
    # encouraging both high confidence and balanced load.
    # The multiplication by num_experts is a heuristic to scale the loss.
    loss = torch.dot(P_i, f_i) * num_experts
    return loss


def train_one_epoch_moe(model: torch.nn.Module, criterion: torch.nn.Module,
                    norm_factor: list, 
                    target: int,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, 
                    args,
                    model_ema: Optional[ModelEma] = None,  
                    amp_autocast=None,
                    loss_scaler=None,
                    clip_grad=None,
                    print_freq: int = 100, 
                    logger=None):
    
    model.train()
    criterion.train()
    
    loss_metric = AverageMeter()
    mae_metric = AverageMeter()

    start_time = time.perf_counter()
    
    task_mean = norm_factor[0]
    task_std  = norm_factor[1]

    for step, data in enumerate(data_loader):
        data = data.to(device)
        with amp_autocast():
            pred, routing_weights = model(data)
            pred = pred.squeeze()
            
            main_loss = criterion(pred, (data.y[:, target] - task_mean) / task_std)
            aux_loss = load_balancing_loss(routing_weights, args.num_experts)
            loss = main_loss + args.alpha * aux_loss
        
        optimizer.zero_grad()
        if loss_scaler is not None:
            loss_scaler(loss, optimizer, parameters=model.parameters())
        else:
            loss.backward()
            if clip_grad is not None:
                dispatch_clip_grad(model.parameters(), 
                    value=clip_grad, mode='norm')
            optimizer.step()
        
        loss_metric.update(loss.item(), n=pred.shape[0])
        y_pred = pred.detach() * task_std + task_mean
        y_true = data.y[:, target]

        mae_metric.update(torch.mean(torch.abs(y_pred-y_true)).item(), n=pred.shape[0])

        if model_ema is not None:
            model_ema.update(model)
        
        torch.cuda.synchronize()
        
        if step % print_freq == 0 or step == len(data_loader) - 1:
            w = time.perf_counter() - start_time
            e = (step + 1) / len(data_loader)
            info_str = 'Epoch: [{epoch}][{step}/{length}] loss: {loss:.5f}, MAE: {mae:.5f}, time/step={time_per_step:.0f}ms, '.format( 
                epoch=epoch, step=step, length=len(data_loader), 
                mae=mae_metric.avg, 
                loss=loss_metric.avg,
                time_per_step=(1e3 * w / e / len(data_loader))
                )
            info_str += 'lr={:.2e}'.format(optimizer.param_groups[0]["lr"])
            logger.info(info_str)
    return mae_metric.avg


def evaluate_moe(model, norm_factor, target, data_loader, device, amp_autocast=None, 
    print_freq=100, logger=None):
    
    model.eval()
    
    loss_metric = AverageMeter()
    mae_metric = AverageMeter()
    criterion = torch.nn.L1Loss()
    criterion.eval()
    
    task_mean = norm_factor[0]
    task_std  = norm_factor[1]
    
    with torch.no_grad():
        for data in data_loader:
            data = data.to(device)
            with amp_autocast():
                pred, _ = model(data) # We don't need routing weights for evaluation
                pred = pred.squeeze()
            
            loss = criterion(pred, (data.y[:, target] - task_mean) / task_std)
            loss_metric.update(loss.item(), n=pred.shape[0])
            y_pred = pred.detach() * task_std + task_mean
            y_true = data.y[:, target]
                
            mae_metric.update(torch.mean(torch.abs(y_pred-y_true)).item(), n=pred.shape[0])

    return mae_metric.avg, loss_metric.avg


def get_args_parser():
    parser = argparse.ArgumentParser('Training equivariant networks', add_help=False)
    parser.add_argument('--output-dir', type=str, default=None)
    # network architecture
    parser.add_argument('--model-name', type=str, default='transformer_ti')
    parser.add_argument('--input-irreps', type=str, default=None)
    parser.add_argument('--radius', type=float, default=2.0)
    parser.add_argument('--num-basis', type=int, default=32)
    parser.add_argument('--output-channels', type=int, default=1)
    # MoE parameters
    parser.add_argument('--num-experts', type=int, default=8, help='Number of experts in MoE')
    parser.add_argument('--k', type=int, default=2, help='Number of experts to route to for each input')
    parser.add_argument('--alpha', type=float, default=0.01, help='Coefficient for the load balancing loss')
    # training hyper-parameters
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument('--model-ema', action='store_true')
    parser.add_argument('--no-model-ema', action='store_false', dest='model_ema')
    parser.set_defaults(model_ema=True)
    parser.add_argument('--model-ema-decay', type=float, default=0.9999, help='')
    parser.add_argument('--model-ema-force-cpu', action='store_true', default=False, help='')
    # regularization
    parser.add_argument('--drop-path', type=float, default=0.0)
    # optimizer (timm)
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=0.01,
                        help='weight decay (default: 0.01)')
    # learning rate schedule parameters (timm)
    parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "cosine"')
    parser.add_argument('--lr', type=float, default=5e-4, metavar='LR',
                        help='learning rate (default: 5e-4)')
    parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                        help='learning rate noise on/off epoch percentages')
    parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                        help='learning rate noise limit percent (default: 0.67)')
    parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                        help='learning rate noise std-dev (default: 1.0)')
    parser.add_argument('--warmup-lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')
    parser.add_argument('--decay-epochs', type=float, default=30, metavar='N',
                        help='epoch interval to decay LR')
    parser.add_argument('--warmup-epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                        help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                        help='patience epochs for Plateau LR scheduler (default: 10')
    parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                        help='LR decay rate (default: 0.1)')
    # logging
    parser.add_argument("--print-freq", type=int, default=100)
    # task
    parser.add_argument("--target", type=int, default=0)
    parser.add_argument("--data-path", type=str, default='datasets/SuperCon')
    parser.add_argument('--run-fold', type=int, default=None)
    parser.add_argument('--order-type', type=str, default='all')
    parser.add_argument('--feature-type', type=str, default='one_hot')
    parser.add_argument('--compute-stats', action='store_true', dest='compute_stats')
    parser.set_defaults(compute_stats=False)
    parser.add_argument('--no-standardize', action='store_false', dest='standardize')
    parser.set_defaults(standardize=True)
    parser.add_argument('--loss', type=str, default='l1')
    # random
    parser.add_argument("--seed", type=int, default=0)
    # data loader config
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument('--pin-mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no-pin-mem', action='store_false', dest='pin_mem',help='')
    parser.set_defaults(pin_mem=True)
    # AMP
    parser.add_argument('--no-amp', action='store_false', dest='amp', 
                        help='Disable FP16 training.')
    parser.set_defaults(amp=True)
    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    return parser


def main(args):

    utils.init_distributed_mode(args)
    is_main_process = (args.rank == 0)
    _log = FileLogger(is_master=is_main_process, is_rank0=is_main_process, output_dir=args.output_dir)
    _log.info(args)
    root_path = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(root_path+'/best_models/')

    if not os.path.exists(save_path):
        os.mkdir(save_path)

    ''' Dataset '''
    
    data_source = get_Path(args.data_path+'/cif/')

    if args.order_type == 'order':
        order_data,disorder_data = classify(data_source)
        data_source = order_data
    elif args.order_type == 'disorder':
        order_data,disorder_data = classify(data_source)
        data_source = disorder_data
    elif args.order_type == 'all':  
        data_source = data_source
    else:
        print('please input the correct order_type')

    fold_num = 10
    train_idx,valid_idx,test_idx = splitdata(data_source,fold_num,args.run_fold)
    
    train = [data_source[i] for i in train_idx]
    valid = [data_source[i] for i in valid_idx]
    test = [data_source[i] for i in test_idx]

    train_dataset = SC(args.data_path,'train', train,args.run_fold, feature_type=args.feature_type)
    val_dataset  = SC(args.data_path, 'valid',valid,args.run_fold, feature_type=args.feature_type)
    test_dataset = SC(args.data_path, 'test',test,args.run_fold, feature_type=args.feature_type)

    _log.info('Training set mean: {}, std:{}'.format(
            train_dataset.mean(args.target), train_dataset.std(args.target)))
    # calculate dataset stats
    task_mean, task_std = 0, 1
    if args.standardize:
        task_mean, task_std = train_dataset.mean(args.target), train_dataset.std(args.target)
    norm_factor = [task_mean, task_std]
    
    # since dataset needs random 
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ''' Network '''
    create_gnn_model = model_entrypoint(args.model_name)
    gnn_model_args = {
        'irreps_in': args.input_irreps,
        'radius': args.radius,
        'num_basis': args.num_basis,
        'out_channels': args.output_channels,
        'task_mean': task_mean,
        'task_std': task_std,
        'atomref': None, #train_dataset.atomref(args.target),
        'drop_path': args.drop_path
    }
    
    model = MoE_GNN(
        num_experts=args.num_experts,
        k=args.k,
        gnn_model_creator=create_gnn_model,
        model_args=gnn_model_args
    )
    _log.info(model)
    model = model.to(device)
    
    model_ema = None
    if args.model_ema:
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
        model_ema = ModelEma(
                model,
                decay=args.model_ema_decay,
                device='cpu' if args.model_ema_force_cpu else None)

    # distributed training
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank])

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    _log.info('Number of params: {}'.format(n_parameters))
    
    ''' Optimizer and LR Scheduler '''
    optimizer = create_optimizer(args=args, model=model)
    lr_scheduler, _ = create_scheduler(args, optimizer)
    criterion = None
    if args.loss == 'l1':
        criterion = torch.nn.L1Loss()
    elif args.loss == 'l2':
        criterion = torch.nn.MSELoss()
    else:
        raise ValueError

    ''' AMP (from timm) '''
    # setup automatic mixed-precision (AMP) loss scaling and op casting
    amp_autocast = suppress  # do nothing
    loss_scaler = None
    if args.amp:
        amp_autocast = torch.cuda.amp.autocast
        loss_scaler = NativeScaler()
    
    ''' Data Loader '''
    if args.distributed:
        sampler_train = torch.utils.data.DistributedSampler(
                        train_dataset, num_replicas=utils.get_world_size(), rank=utils.get_rank(), shuffle=True
                )
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
                sampler=sampler_train, num_workers=args.workers, pin_memory=args.pin_mem, 
                drop_last=True)
    else:
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
                shuffle=True, num_workers=args.workers, pin_memory=args.pin_mem, 
                drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,num_workers=args.workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,num_workers=args.workers)
    

    best_epoch, best_train_err, best_val_err, best_test_err = 0, float('inf'), float('inf'), float('inf')
    best_ema_epoch, best_ema_val_err, best_ema_test_err = 0, 0, float('inf')
    
    for epoch in range(args.epochs):
            
        epoch_start_time = time.perf_counter()
        epoch_error = []
        lr_scheduler.step(epoch)

        if args.distributed:
            train_loader.sampler.set_epoch(epoch)
        
        train_err = train_one_epoch_moe(model=model, criterion=criterion, norm_factor=norm_factor,
                target=args.target, data_loader=train_loader, optimizer=optimizer,
                device=device, epoch=epoch, args=args, model_ema=model_ema,
                amp_autocast=amp_autocast, loss_scaler=loss_scaler,
                print_freq=args.print_freq, logger=_log)
        
        val_err, val_loss = evaluate_moe(model, norm_factor, args.target, val_loader, device, 
                amp_autocast=amp_autocast, print_freq=args.print_freq, logger=_log)
        
        test_err, test_loss = evaluate_moe(model, norm_factor, args.target, test_loader, device, 
                amp_autocast=amp_autocast, print_freq=args.print_freq, logger=_log)
        
        # record the best results
        if val_err < best_val_err:
            best_val_err = val_err
            best_test_err = test_err
            best_train_err = train_err
            best_epoch = epoch
            if best_test_err < 100:
                torch.save(model,save_path+str(args.run_fold)+'_save.pt')

        # print MAE
        
        info_str = 'Epoch: [{epoch}] train MAE: {train_mae:.5f}, '.format(epoch=epoch,train_mae=train_err)
        info_str += 'val MAE: {:.5f}, '.format(val_err)
        info_str += 'test MAE: {:.5f}, '.format(test_err)
        info_str += 'Time: {:.2f}s'.format(time.perf_counter() - epoch_start_time)
        _log.info(info_str)
        
        info_str = 'Best -- epoch={}, train MAE: {:.5f}, val MAE: {:.5f}, test MAE: {:.5f}\n'.format(
                best_epoch, best_train_err, best_val_err, best_test_err)
        _log.info(info_str)
        epoch_error.append(best_test_err)
        
        
        # evaluation with EMA
        if model_ema is not None:
            ema_val_err, _ = evaluate_moe(model_ema.module, norm_factor, args.target, val_loader, device, 
                    amp_autocast=amp_autocast, print_freq=args.print_freq, logger=_log)
            
            ema_test_err, _ = evaluate_moe(model_ema.module, norm_factor, args.target, test_loader, device, 
                    amp_autocast=amp_autocast, print_freq=args.print_freq, logger=_log)
            
            # record the best results
            if (ema_val_err) < best_ema_val_err:
                best_ema_val_err = ema_val_err
                best_ema_test_err = ema_test_err
                best_ema_epoch = epoch

            info_str = 'Epoch: [{epoch}]'.format(epoch=epoch)
            info_str += 'EMA val MAE: {:.5f}, '.format(ema_val_err)
            info_str += 'EMA test MAE: {:.5f}, '.format(ema_test_err)
            info_str += 'Time: {:.2f}s'.format(time.perf_counter() - epoch_start_time)
            _log.info(info_str)
            
            info_str = 'Best EMA -- epoch={}, val MAE: {:.5f}, test MAE: {:.5f}\n'.format(
                        best_ema_epoch, best_ema_val_err, best_ema_test_err)
            _log.info(info_str)
    
    all_err = 'fold_{} test MAE:{:.5f}'.format(str(args.run_fold),epoch_error[-1])
    _log.info(all_err)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser('Training equivariant networks', parents=[get_args_parser()])
    args = parser.parse_args()      
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
        
