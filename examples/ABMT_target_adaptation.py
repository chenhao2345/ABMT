from __future__ import print_function, absolute_import
import argparse
import os.path as osp
import random
import numpy as np
import sys
import torch.nn.functional as F
from hdbscan import HDBSCAN
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
import time
import torch
from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader
# from scipy.special import softmax
from abmt import datasets
from abmt import models
from abmt.trainers import ABMTTrainer
from abmt.evaluators import Evaluator, extract_features
from abmt.utils.data import IterLoader
from abmt.utils.data import transforms as T
from abmt.utils.data.sampler import RandomMultipleGallerySampler
from abmt.utils.data.preprocessor import Preprocessor
from abmt.utils.logging import Logger
from abmt.utils.serialization import load_checkpoint, save_checkpoint, copy_state_dict
from abmt.utils.rerank import compute_jaccard_dist

start_epoch = best_mAP = 0

def get_data(name, data_dir):
    root = osp.join(data_dir, name)
    dataset = datasets.create(name, root)
    return dataset

def get_train_loader(dataset, height, width, batch_size, workers,
                    num_instances, iters, mutual=False):

    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    train_transformer = T.Compose([
             T.Resize((height, width), interpolation=3),
             T.RandomHorizontalFlip(p=0.5),
             T.Pad(10),
             T.RandomCrop((height, width)),
             T.ToTensor(),
             normalizer,
	         T.RandomErasing(probability=0.5, mean=[0.485, 0.456, 0.406])
         ])
    # print(dataset)
    train_set = dataset.train

    rmgs_flag = num_instances > 0
    if rmgs_flag:
        sampler = RandomMultipleGallerySampler(train_set, num_instances)
    else:
        sampler = None
    train_loader = IterLoader(
                DataLoader(Preprocessor(train_set, root=dataset.images_dir,
                                        transform=train_transformer, mutual=mutual),
                            batch_size=batch_size, num_workers=workers, sampler=sampler,
                            shuffle=not rmgs_flag, pin_memory=True, drop_last=True), length=iters)

    return train_loader

def get_test_loader(dataset, height, width, batch_size, workers, testset=None):
    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

    test_transformer = T.Compose([
             T.Resize((height, width), interpolation=3),
             T.ToTensor(),
             normalizer
         ])

    if (testset is None):
        testset = list(set(dataset.query) | set(dataset.gallery))

    test_loader = DataLoader(
        Preprocessor(testset, root=dataset.images_dir, transform=test_transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=True)

    return test_loader

def create_model(args):
    model_1 = models.create(args.arch, num_features=args.features, dropout=args.dropout, num_classes=1)

    model_1_ema = models.create(args.arch, num_features=args.features, dropout=args.dropout, num_classes=1)

    model_1.cuda()
    model_1_ema.cuda()
    model_1 = nn.DataParallel(model_1)
    model_1_ema = nn.DataParallel(model_1_ema)

    if args.no_source:
        print('No source pre-training')
    else:
        initial_weights = load_checkpoint(args.init_1)
        copy_state_dict(initial_weights['state_dict'], model_1)
        copy_state_dict(initial_weights['state_dict'], model_1_ema)

    for param in model_1_ema.parameters():
        param.detach_()


    return model_1, model_1_ema


def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True

    main_worker(args)


def main_worker(args):
    global start_epoch, best_mAP

    cudnn.benchmark = True

    sys.stdout = Logger(osp.join(args.logs_dir, 'log.txt'))
    print("==========\nArgs:{}\n==========".format(args))

    # Create data loaders
    iters = args.iters if (args.iters>0) else None
    dataset_target = get_data(args.dataset_target, args.data_dir)
    ori_train = dataset_target.train
    if not args.no_source:
        dataset_source = get_data(args.dataset_source, args.data_dir)
    test_loader_target = get_test_loader(dataset_target, args.height, args.width, args.batch_size, args.workers)

    # Create model
    model_1, model_1_ema = create_model(args)

    # Evaluator
    evaluator_1_ema = Evaluator(model_1_ema)

    best_mAP = 0

    for nc in range(args.epochs):

        cluster_loader = get_test_loader(dataset_target, args.height, args.width, args.batch_size, args.workers,
                                         testset=dataset_target.train)
        dict_f, _ = extract_features(model_1_ema, cluster_loader, print_freq=50)
        cf_1 = torch.stack(list(dict_f.values()))

        # DBSCAN cluster
        if args.no_source:
            rerank_dist = compute_jaccard_dist(cf_1, lambda_value=0, source_features=None,
                                               use_gpu=False).numpy()
        else:
            cluster_loader_source = get_test_loader(dataset_source, args.height, args.width, args.batch_size,
                                                    args.workers, testset=dataset_source.train)
            dict_f_source, _ = extract_features(model_1_ema, cluster_loader_source, print_freq=50)
            cf_1_source = torch.stack(list(dict_f_source.values()))
            rerank_dist = compute_jaccard_dist(cf_1, lambda_value=args.lambda_value, source_features=cf_1_source,
                                               use_gpu=False).numpy()
            del cf_1_source
        tri_mat = np.triu(rerank_dist, 1)  # tri_mat.dim=2
        tri_mat = tri_mat[np.nonzero(tri_mat)]  # tri_mat.dim=1
        tri_mat = np.sort(tri_mat, axis=None)
        top_num = np.round(args.rho * tri_mat.size).astype(int)
        eps = tri_mat[:top_num].mean()
        print('eps in cluster: {:.3f}'.format(eps))
        print('Clustering and labeling...')
        cluster = DBSCAN(eps=eps, min_samples=4, metric='precomputed', n_jobs=-1)
        labels = cluster.fit_predict(rerank_dist)
        num_ids = len(set(labels)) -1

        print('Epoch {} have {} training ids'.format(nc, num_ids))
        # generate new dataset
        labeled_ind, unlabeled_ind = [], []
        for ind, label in enumerate(labels):
            if label == -1:
                unlabeled_ind.append(ind)
            else:
                labeled_ind.append(ind)
        # print('Epoch {} have {} labeled samples and {} unlabeled samples'.format(nc + 1, len(labeled_ind), len(unlabeled_ind)))

        cf_1 = cf_1.numpy()
        centers = []
        for id in range(num_ids):
            centers.append(np.mean(cf_1[labels == id], axis=0))
        centers = np.stack(centers, axis=0)

        del cf_1, rerank_dist

        model_1.module.classifier = nn.Linear(2048, num_ids, bias=False).cuda()
        model_1_ema.module.classifier = nn.Linear(2048, num_ids, bias=False).cuda()
        model_1.module.classifier_max = nn.Linear(2048, num_ids, bias=False).cuda()
        model_1_ema.module.classifier_max = nn.Linear(2048, num_ids, bias=False).cuda()

        model_1.module.classifier.weight.data.copy_(
            torch.from_numpy(normalize(centers[:, :2048], axis=1)).float().cuda())
        model_1_ema.module.classifier.weight.data.copy_(
            torch.from_numpy(normalize(centers[:, :2048], axis=1)).float().cuda())

        model_1.module.classifier_max.weight.data.copy_(
            torch.from_numpy(normalize(centers[:, 2048:], axis=1)).float().cuda())
        model_1_ema.module.classifier_max.weight.data.copy_(
            torch.from_numpy(normalize(centers[:, 2048:], axis=1)).float().cuda())

        del centers

        target_label = labels

        for i in range(len(dataset_target.train)):
            dataset_target.train[i] = list(dataset_target.train[i])
            dataset_target.train[i][1] = int(target_label[i])
            dataset_target.train[i] = tuple(dataset_target.train[i])

        # Optimizer
        params = []
        for key, value in model_1.named_parameters():
            if not value.requires_grad:
                continue
            params += [{"params": [value], "lr": args.lr, "weight_decay": args.weight_decay}]

        optimizer = torch.optim.Adam(params)

        # Trainer
        trainer = ABMTTrainer(model_1, model_1_ema, num_cluster=num_ids, alpha=args.alpha)
        epoch = nc
        # # DBSCAN
        dataset_target.train = [ori_train[i] for i in labeled_ind]
        print(len(dataset_target.train), 'are labeled.')
        labeled_loader_target = get_train_loader(dataset_target, args.height, args.width,
                                               args.batch_size, args.workers, args.num_instances, iters, mutual=True)
        labeled_loader_target.new_epoch()

        trainer.train(epoch, labeled_loader_target, optimizer,
                    print_freq=args.print_freq, train_iters=len(labeled_loader_target))

        def save_model(model_ema, is_best, best_mAP, mid, num_ids):
            save_checkpoint({
                'state_dict': model_ema.state_dict(),
                'epoch': epoch + 1,
                'best_mAP': best_mAP,
                'num_ids': num_ids
            }, is_best, fpath=osp.join(args.logs_dir, 'model'+str(mid)+'_checkpoint.pth.tar'))

        if ((epoch+1)%args.eval_step==0 or (epoch==args.epochs-1)):
            print('Evaluating teacher net:')
            cmc, mAP_1 = evaluator_1_ema.evaluate(test_loader_target, dataset_target.query, dataset_target.gallery, cmc_flag=True)
            is_best = (mAP_1>best_mAP)
            best_mAP = max(mAP_1, best_mAP)

            save_model(model_1_ema, is_best, best_mAP, 1, num_ids)
            dataset_target.train = ori_train
    print ('Test on the best model.')
    checkpoint = load_checkpoint(osp.join(args.logs_dir, 'model_best.pth.tar'))
    model_best = models.create(args.arch, num_features=args.features, dropout=args.dropout, num_classes=checkpoint['num_ids'])
    model_best.cuda()
    model_best = nn.DataParallel(model_best)
    evaluator_best = Evaluator(model_best)
    model_best.load_state_dict(checkpoint['state_dict'])
    evaluator_best.evaluate(test_loader_target, dataset_target.query, dataset_target.gallery, cmc_flag=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="ABMT Training")
    # data
    parser.add_argument('-dt', '--dataset-target', type=str, default='market1501',
                        choices=datasets.names())
    parser.add_argument('-ds', '--dataset-source', type=str, default='dukemtmc-reid',
                        choices=datasets.names())
    parser.add_argument('-b', '--batch-size', type=int, default=64)
    parser.add_argument('-j', '--workers', type=int, default=4)
    parser.add_argument('--height', type=int, default=256,
                        help="input height")
    parser.add_argument('--width', type=int, default=128,
                        help="input width")
    parser.add_argument('--num-instances', type=int, default=4,
                        help="each minibatch consist of "
                             "(batch_size // num_instances) identities, and "
                             "each identity has num_instances instances, "
                             "default: 0 (NOT USE)")
    # model
    parser.add_argument('-a', '--arch', type=str, default='resnet50',
                        choices=models.names())
    parser.add_argument('--features', type=int, default=0)
    parser.add_argument('--dropout', type=float, default=0)
    # optimizer
    parser.add_argument('--lr', type=float, default=0.00035,
                        help="learning rate of new parameters, for pretrained "
                             "parameters it is 10 times smaller than this")
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--alpha', type=float, default=0.999)
    parser.add_argument('--moving-avg-momentum', type=float, default=0.9)
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--iters', type=int, default=800)
    # training configs
    parser.add_argument('--no-source', action='store_true')
    parser.add_argument('--init-1', type=str, default='', metavar='PATH')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--print-freq', type=int, default=100)
    parser.add_argument('--eval-step', type=int, default=1)
    # path
    working_dir = osp.dirname(osp.abspath(__file__))
    parser.add_argument('--data-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'data'))
    parser.add_argument('--logs-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'logs'))
    # cluster
    parser.add_argument('--lambda_value', type=float, default=0.1,
                        help="balancing parameter, default: 0.1")
    parser.add_argument('--rho', type=float, default=2e-3,
                        help="rho percentage, default: 2e-3")
    main()
