# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import copy
import json
import os
import os.path as osp
import random
import time
import warnings
from pathlib import Path
import numpy as np


import mmcv
import torch
from mmcv import Config, DictAction
from mmcv.runner import get_dist_info, init_dist
from mmcv.utils import get_git_hash
from mmdet import __version__
from mmdet.apis import init_random_seed, set_random_seed, train_detector
from mmdet.models import build_detector
from mmdet.utils import collect_env, get_root_logger

from openpsg.datasets import build_dataset

# Here is aa
def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument('--resume-from',
                        help='the checkpoint file to resume from')
    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='whether not to evaluate the checkpoint during training',
    )
    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument(
        '--gpus',
        type=int,
        help='number of gpus to use '
        '(only applicable to non-distributed training)',
    )
    group_gpus.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='ids of gpus to use '
        '(only applicable to non-distributed training)',
    )
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.',
    )
    parser.add_argument(
        '--options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file (deprecate), '
        'change to --cfg-options instead.',
    )
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.',
    )
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher',
    )
    parser.add_argument('--local_rank', type=int, default=0)
    ## Add FL setting and storing information.
    # 어떤 알고리즘을 학습했는지 저장하기 위한 변수(motif, vctree, imp)
    parser.add_argument('--model_name', type=str, required=True)
    # predcls 또는 sgdet를 학습했는지 확인을 위한 저장 변수
    parser.add_argument('--job_name', type=str, default='sgdet')

    # communication round
    parser.add_argument('--n_rounds', type=int, required=True)

    # Total number of client
    parser.add_argument('--num_client', type=int, default=100)
    # Participation ratio
    parser.add_argument('--selected_client', type=int, required=True)
    # the number of cluster of data
    parser.add_argument('--num_cluster', type=int)
    # Data distribution
    parser.add_argument('--distribution', type=str, choices=['IID', 'nonIID', 'Diri_02', 'Diri_1', 'Diri_10'])
    # Cluster type of dataset
    parser.add_argument('--cluster_type', type=str, required=True, choices=['Random', 'Super'])

    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.options and args.cfg_options:
        raise ValueError(
            '--options and --cfg-options cannot be both '
            'specified, --options is deprecated in favor of --cfg-options')
    if args.options:
        warnings.warn('--options is deprecated in favor of --cfg-options')
        args.cfg_options = args.options

    return args


def main():

    args = parse_args()

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])
    if args.resume_from is not None:
        cfg.resume_from = args.resume_from
    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids
        # GPU_LIST = os.environ["CUDA_VISIBLE_DEVICES"]
        # torch.cuda.set_device(GPU_LIST[args.gpu_ids])
    else:
        cfg.gpu_ids = range(1) if args.gpus is None else range(args.gpus)

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)
        # re-set gpu_ids with distributed training mode
        _, world_size = get_dist_info()
        cfg.gpu_ids = range(world_size)

    # create work_dir
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    # dump config
    cfg.dump(osp.join(cfg.work_dir, osp.basename(args.config)))
    # init the logger before other steps
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)

    # init the meta dict to record some important information such as
    # environment info and seed, which will be logged
    meta = dict()
    # log env info
    env_info_dict = collect_env()
    env_info = '\n'.join([(f'{k}: {v}') for k, v in env_info_dict.items()])
    dash_line = '-' * 60 + '\n'
    logger.info('Environment info:\n' + dash_line + env_info + '\n' +
                dash_line)
    meta['env_info'] = env_info
    cfg.device = 'cuda'
    meta['config'] = cfg.pretty_text
    # log some basic info
    logger.info(f'Distributed training: {distributed}')
    logger.info(f'Config:\n{cfg.pretty_text}')

    # set random seeds
    seed = init_random_seed(args.seed)
    logger.info(f'Set random seed to {seed}, '
                f'deterministic: {args.deterministic}')
    set_random_seed(seed, deterministic=args.deterministic)
    cfg.seed = seed
    meta['seed'] = seed
    meta['exp_name'] = osp.basename(args.config)

    num_client = args.num_client
    n_rounds = args.n_rounds
    fl_cfg_annfiles = []

    if args.cluster_type == 'Super':
        folder_path = f'data/FL_DATA_SPLIT/mini_Super_cluster_{args.num_cluster}_{args.distribution}'
    elif args.cluster_type == 'Random':
        folder_path = 'data/FL_DATA_SPLIT/mini_Random'
    else:
        raise NotImplementedError

    num_datas = []

    for idx in range(num_client):
        for file in os.listdir(folder_path):
            if file.endswith(f"User{idx}.json"):
                fl_cfg_annfiles.append(os.path.join(folder_path, file))
                with open(os.path.join(folder_path, file)) as json_file:
                    json_data = json.load(json_file)
                    num_datas.append(len(json_data['data']))
                break
    num_datas = np.array(num_datas)

    fl_dataset = build_dataset(cfg.data.train)

    model = build_detector(cfg.model,
                           train_cfg=cfg.get('train_cfg'),
                           test_cfg=cfg.get('test_cfg'))
    model.init_weights()

    # NOTE: Freeze weights here
    if hasattr(cfg, 'freeze_modules'):
        if cfg.freeze_modules is not None:
            for module_name in cfg.freeze_modules:
                for name, p in model.named_parameters():
                    if name.startswith(module_name):
                        p.requires_grad = False
    # Unfreeze weights here
    # if hasattr(cfg, 'required_grad_modules'):
    #     if cfg.required_grad_modules is not None:
    #         for module_name in cfg.required_grad_modules:
    #             for name, p in model.named_parameters():
    #                 if name.startswith(module_name):
    #                     p.requires_grad = True


    model.CLASSES = fl_dataset.CLASSES

    clients = range(num_client)
    selected_client = args.selected_client

    check_flag = False

    if hasattr(model, 'relation_head'): # Motifs, VCtree, ...
        if check_flag:
            raise ValueError('bbox must not be in relation head')
        check_flag = True
        g_rel_model = [param.data.view(-1) for param in model.relation_head.state_dict().values()]
    if not check_flag:
        raise ValueError('check model parameters')

    g_rel_model = torch.cat(g_rel_model).cpu()

    avg_model = None
    for r in range(n_rounds):
        logger.info(f'Start round : {r}')
        selected_clients = random.sample(clients, selected_client)

        for i in selected_clients:
            tmp_cfg = copy.deepcopy(cfg)

            print('(1): Load state_dict of relation head')

            if hasattr(model, 'bbox_head'):
                set_model_transformer(model, g_rel_model)
            elif hasattr(model, 'relation_head'):
                set_model(model, g_rel_model)
            else:
                raise ValueError('check model parameters')
            model.cuda()

            print('(2): Split according to dict_data')
            tmp_cfg.data.train['ann_file'] = fl_cfg_annfiles[i]
            fl_dataset = build_dataset(tmp_cfg.data.train)


            train_detector(
                model,
                fl_dataset,
                tmp_cfg,
                distributed=distributed,
                validate=False,
                timestamp=timestamp,
                meta=meta,
            )

            del fl_dataset

            print('end of train user (3)')
            if hasattr(model, 'bbox_head'):  # transformer
                trained_model = [param.data.view(-1) for param in model.bbox_head.state_dict().values()]
            elif hasattr(model, 'relation_head'):  # motifs
                trained_model = [param.data.view(-1) for param in model.relation_head.state_dict().values()]
            else:
                raise ValueError('check model parameters')

            trained_model = torch.cat(trained_model).cpu()

            if avg_model == None:
                avg_model = trained_model.mul_(num_datas[i]/sum(num_datas[selected_clients]))
            else:
                avg_model.add_(trained_model.mul_(num_datas[i]/sum(num_datas[selected_clients])))


        print("Average and Set models")

        g_rel_model = torch.empty_like(avg_model).copy_(avg_model)
        avg_model = None


        if (r % 10 == 0) or (r == n_rounds-1):
            if hasattr(model, 'bbox_head'):  # transformer
                set_model_transformer(model, g_rel_model)
            elif hasattr(model, 'relation_head'):  # motifs
                set_model(model, g_rel_model)
            else:
                raise ValueError('check model parameters')


            if args.cluster_type == 'Super':
                folder_path = f'data/FL_TRAINED_MODEL/mini_{args.model_name}/{args.cluster_type}-{args.distribution}-Cluster{args.num_cluster}_Ratio{args.selected_client}_User{args.num_client}'
            elif args.cluster_type == 'Random':
                folder_path = f'data/FL_TRAINED_MODEL/mini_{args.model_name}/Random_Ratio{args.selected_client}_User{args.num_client}'
            else:
                raise NotImplementedError

            os.makedirs(folder_path, exist_ok=True)
            ckpt_path = os.path.join(folder_path, f'Round-{r}.pth')
            torch.save(model.state_dict(), ckpt_path)



def avg_models(ls_models):
    '''
    Average models in ls_models

    Args:
        ls_models: list of state_dict of models to average (i.e., server averages the clients model)

    Retu rns:
        averaged model
    '''
    weights = []
    for i in ls_models:
        weights.append(i[1])
    weights = torch.tensor(weights)
    weights = weights / torch.sum(weights)

    serialized_params_list = [i[0] for i in ls_models]
    serialized_parameters = torch.sum(
        torch.stack(serialized_params_list, dim=-1) * weights, dim=-1)
    return serialized_parameters

def set_model(model, rel_model, mode="copy"):
    """
        set model parameter as rel_model
    """
    current_index = 0  # keep track of where to read from grad_update

    for param in model.relation_head.state_dict().values():
        numel = param.numel()
        size = param.size()
        if mode == "copy":
            param.copy_(
                rel_model[current_index:current_index +
                                                    numel].view(size))
        elif mode == "add":
            param.add_(
                rel_model[current_index:current_index +
                                                    numel].view(size))
        elif mode == "sub":
            param.sub_(
                rel_model[current_index:current_index +
                                                    numel].view(size))
        else:
            raise ValueError(
                "Invalid deserialize mode {}, require \"copy\", \"add\" or \"sub\" "
                .format(mode))
        current_index += numel
def set_model_transformer(model, rel_model, mode="copy"):
    """
        set model parameter as rel_model
    """
    current_index = 0  # keep track of where to read from grad_update

    for param in model.bbox_head.state_dict().values():
        numel = param.numel()
        size = param.size()
        if mode == "copy":
            param.copy_(
                rel_model[current_index:current_index +
                                                    numel].view(size))
        elif mode == "add":
            param.add_(
                rel_model[current_index:current_index +
                                                    numel].view(size))
        elif mode == "sub":
            param.sub_(
                rel_model[current_index:current_index +
                                                    numel].view(size))
        else:
            raise ValueError(
                "Invalid deserialize mode {}, require \"copy\", \"add\" or \"sub\" "
                .format(mode))
        current_index += numel

if __name__ == '__main__':
    main()
