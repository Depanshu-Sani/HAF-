import argparse
import os
import json
import re

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torch.utils.data.distributed
import torchvision.datasets as datasets
import torch.optim
import numpy as np
from torchvision import transforms

from MBM.better_mistakes.data import cifar100
from MBM.better_mistakes.data.softmax_cascade import SoftmaxCascade
from MBM.better_mistakes.model.init import init_model_on_gpu
from MBM.better_mistakes.data.transforms import val_transforms
from MBM.better_mistakes.model.run_xent import run
from MBM.better_mistakes.model.run_nn import run_nn
from MBM.better_mistakes.model.labels import make_all_soft_labels
from MBM.better_mistakes.util.label_embeddings import create_embedding_layer
from MBM.better_mistakes.util.devise_and_bd import generate_sorted_embedding_tensor
from MBM.better_mistakes.util.config import load_config
from MBM.better_mistakes.model.losses import HierarchicalCrossEntropyLoss, CosineLoss, RankingLoss, CosinePlusXentLoss, YOLOLoss
from MBM.better_mistakes.trees import load_hierarchy, get_weighting, load_distances, get_classes
from util import data_loader, logger
from util.data_loader import is_sorted
import torch.nn.functional as F
from cifar100.cifar100_get_tree_target_level5 import get_targets
from iNat19.inat_get_target_tree import get_target_l7
from tiered_imagenet.tiered_get_target_tree import get_target_l12
from nltk.tree import Tree
from collections import deque

DATASET_NAMES = ["tiered-imagenet-84", "inaturalist19-84", "tiered-imagenet-224", "inaturalist19-224", "cifar-100"]
LOSS_NAMES = ["cross-entropy", "soft-labels", "hierarchical-cross-entropy", "cosine-distance", "ranking-loss", "cosine-plus-xent", "yolo-v2",
              "flamingo-l5", "flamingo-l7", "flamingo-l12",
              "ours-l5-cejsd", "ours-l7-cejsd", "ours-l12-cejsd",
              "ours-l5-cejsd-wtconst", "ours-l7-cejsd-wtconst", "ours-l12-cejsd-wtconst",
              "ours-l5-cejsd-wtconst-dissim", "ours-l7-cejsd-wtconst-dissim", "ours-l12-cejsd-wtconst-dissim"]

def main(test_opts):
    gpus_per_node = torch.cuda.device_count()

    assert test_opts.out_folder

    if test_opts.start:
        opts = test_opts
        opts.epochs = 0
    else:
        expm_json_path = os.path.join(test_opts.out_folder, "opts.json")
        assert os.path.isfile(expm_json_path)
        with open(expm_json_path) as fp:
            opts = json.load(fp)
            # convert dictionary to namespace
            opts = argparse.Namespace(**opts)
            opts.out_folder = None
            opts.epochs = 0

        if test_opts.data_path is None or opts.data_path is None:
            opts.data_paths = load_config(test_opts.data_paths_config)
            opts.data_path = opts.data_paths[opts.data]

        opts.start = test_opts.start # to update value of start if testing

    # Setup data loaders ------------------------------------------------------------------------------------------
    test_dataset, test_loader = data_loader.test_data_loader(opts)
    # train_dataset, val_dataset, train_loader, val_loader = data_loader.train_data_loader(opts)
    # test_dataset, test_loader = train_dataset, train_loader

    # Load hierarchy and classes --------------------------------------------------------------------------------------------------------------------------
    distances = load_distances(opts.data, 'ilsvrc', opts.data_dir)
    hierarchy = load_hierarchy(opts.data, opts.data_dir)

    if opts.loss == "yolo-v2":
        classes, _ = get_classes(hierarchy, output_all_nodes=True)
    else:
        if opts.data == "cifar-100":
            classes = test_dataset.class_to_idx
            classes = ["L5-" + str(classes[i]) for i in classes]
        else:
            classes = test_dataset.classes

    opts.num_classes = len(classes)

    # HAF++ node embedding generation
    if opts.feature_space == "haf++":
        def map_tree_to_ids_bfs(tree):
            node_to_id = {}
            current_id = 0

            # Use a queue to keep track of nodes to process
            queue = deque([tree])

            while queue:
                node = queue.popleft()

                if isinstance(node, Tree):
                    node_str = node.label()  # Convert the node to its string representation
                else:
                    node_str = node

                if node_str not in node_to_id and node_str != 'root':  # Check to avoid duplicate keys
                    node_to_id[node_str] = current_id
                    current_id += 1

                if isinstance(node, Tree):
                    for child in node:
                        queue.append(child)

            return node_to_id
        # def map_tree_to_ids(tree):
        #     node_to_id = {}
        #     current_id = [0]  # Use a list to keep the current ID mutable
        #
        #     def assign_ids(node):
        #         if isinstance(node, Tree):
        #             node_str = node.label()  # Convert the node to its string representation
        #         else:
        #             node_str = node
        #         if node_str not in node_to_id and node_str != 'root':  # Check to avoid duplicate keys
        #             node_to_id[node_str] = current_id[0]
        #             current_id[0] += 1
        #         if isinstance(node, Tree):
        #             for child in node:
        #                 assign_ids(child)
        #
        #     assign_ids(tree)
        #     return node_to_id

        node_to_id = map_tree_to_ids_bfs(hierarchy)
        opts.num_classes = len(node_to_id)

        # orthonormal_basis_vectors = torch.eye(opts.num_classes, device=opts.gpu, dtype=torch.float32)
        # leaf_node_embeddings = torch.zeros((len(classes), opts.num_classes), device=opts.gpu, dtype=torch.float32)
        #
        # leaf_values = hierarchy.leaves()
        # for class_ in classes:
        #     class_idx = classes.index(class_)
        #     leaf_index = leaf_values.index(class_)
        #     tree_location = hierarchy.leaf_treeposition(leaf_index)
        #     for level, i in enumerate(range(len(tree_location))):
        #         try:
        #             label = hierarchy[tree_location[:i + 1]].label()
        #         except:
        #             label = hierarchy[tree_location[:i + 1]]
        #         leaf_node_embeddings[class_idx] += orthonormal_basis_vectors[node_to_id[label]]
        #
        # leaf_node_embeddings = leaf_node_embeddings / torch.norm(leaf_node_embeddings, dim=1)[:, None]

    # Model, loss, optimizer ------------------------------------------------------------------------------------------------------------------------------

    # more setup for devise and b+d
    if opts.devise:
        assert not opts.barzdenzler
        embeddings_mat, sorted_keys = generate_sorted_embedding_tensor(opts)
        embeddings_mat = embeddings_mat / np.linalg.norm(embeddings_mat, axis=1, keepdims=True)
        emb_layer, _, opts.embedding_size = create_embedding_layer(embeddings_mat)
        assert is_sorted(sorted_keys)

    if opts.barzdenzler:
        assert not opts.devise
        embeddings_mat, sorted_keys = generate_sorted_embedding_tensor(opts)
        embeddings_mat = embeddings_mat / np.linalg.norm(embeddings_mat, axis=1, keepdims=True)
        emb_layer, _, opts.embedding_size = create_embedding_layer(embeddings_mat)
        assert is_sorted(sorted_keys)

    # setup loss
    if opts.loss == "cross-entropy" and opts.feature_space != "haf++":
        loss_function = nn.CrossEntropyLoss().cuda(opts.gpu)
    elif opts.loss == "soft-labels":
        loss_function = nn.KLDivLoss().cuda(opts.gpu)
    elif opts.loss == "hierarchical-cross-entropy":
        weights = get_weighting(hierarchy, "exponential", value=opts.alpha)
        loss_function = HierarchicalCrossEntropyLoss(hierarchy, classes, weights).cuda(opts.gpu)
    elif opts.loss == "yolo-v2":

        cascade = SoftmaxCascade(hierarchy, classes).cuda(opts.gpu)
        num_leaf_classes = len(hierarchy.treepositions("leaves"))
        weights = get_weighting(hierarchy, "exponential", value=opts.beta)
        loss_function = YOLOLoss(hierarchy, classes, weights).cuda(opts.gpu)

        def yolo2_corrector(output):
            return cascade.final_probabilities(output)[:, :num_leaf_classes]

    elif opts.loss == "cosine-distance":
        loss_function = CosineLoss(emb_layer).cuda(opts.gpu)
    elif opts.loss == "ranking-loss":
        loss_function = RankingLoss(emb_layer, batch_size=opts.batch_size, single_random_negative=opts.devise_single_negative, margin=0.1).cuda(opts.gpu)
    elif opts.loss == "cosine-plus-xent":
        loss_function = CosinePlusXentLoss(emb_layer).cuda(opts.gpu)
    elif opts.loss == "cross-entropy" and opts.feature_space == "haf++":
        def loss_func(logits, labels, m=0, log=0):
            out = -torch.sqrt((torch.norm(logits, dim=1)[:, None] ** 2) * (1 - F.cosine_similarity(logits[:, :, None], leaf_node_embeddings.t()[None, :, :]) ** 2))
            margin = 1 + torch.zeros_like(out).scatter_(1, labels.unsqueeze(1), m)
            # cross-entropy loss
            loss_ce = F.cross_entropy(out * margin, labels, reduce=False)
            # distance between the true hyperplane (gt) and feature point
            loss_true = -(out * margin)[range(labels.shape[0]), labels]
            # avg distance between false hyperplanes and feature point
            loss_false = 20 - torch.clamp(-(out.sum(1) - out[range(labels.shape[0]), labels]) / (out.shape[1] - 1), 0, 20)
            loss = loss_ce.mean() + loss_true.mean() + loss_false.mean()
            if log == 0:
                print(f"CE: {loss_ce.mean().item()} | True: {loss_true.mean().item()} | False: {loss_false.mean().item()}")
            return loss, out
        loss_function = loss_func
    elif opts.loss in LOSS_NAMES:
        loss_function = None
    else:
        raise RuntimeError("Unkown loss {}".format(opts.loss))

    # for yolo, we need to decode the output of the classifier as it outputs the conditional probabilities
    corrector = yolo2_corrector if opts.loss == "yolo-v2" else lambda x: x

    # create the solft labels
    soft_labels = make_all_soft_labels(distances, classes, opts.beta)

    # Test ------------------------------------------------------------------------------------------------------------------------------------------------
    summaries, summaries_table = dict(), dict()

    # setup model
    model = init_model_on_gpu(gpus_per_node, opts)
    
    if opts.checkpoint_path is None:
        checkpoint_id = "best.checkpoint.pth.tar"
        checkpoint_path = os.path.join(test_opts.out_folder, checkpoint_id)
    else:
        checkpoint_path = opts.checkpoint_path
    # checkpoint_path = os.path.join(test_opts.pretrained_folder, checkpoint_id)
    if os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint["state_dict"])
        leaf_node_embeddings = checkpoint["leaf_node_embeddings"]
        logger._print("=> loaded checkpoint '{}'".format(checkpoint_path), os.path.join(test_opts.out_folder, "logs.txt"))
    else:
        logger._print("=> no checkpoint found at '{}'".format(checkpoint_path), os.path.join(test_opts.out_folder, "logs.txt"))
        raise RuntimeError

    if opts.devise or opts.barzdenzler:
        summary, _ = run_nn(test_loader, model, loss_function, distances, classes, opts, 0, 0, emb_layer, embeddings_mat, is_inference=True)
    else:
        summary, _ = run(test_loader, model, loss_function, distances, soft_labels, classes, opts, 0, 0, is_inference=True, corrector=corrector)

    for k in summary.keys():
        val = summary[k]
        # if "accuracy_top" in k or "ilsvrc_dist_precision" in k or "ilsvrc_dist_mAP" in k:
        #     val *= 100
        # if "accuracy" in k:
        #     k_err = re.sub(r"accuracy", "error", k)
        #     val = 100 - val
        #     k = k_err
        if k not in summaries:
            summaries[k] = []
        summaries[k].append(val)

    for k in summaries.keys():
        avg = np.mean(summaries[k])
        conf95 = 1.96 * np.std(summaries[k]) / np.sqrt(len(summaries[k]))
        summaries_table[k] = (avg, conf95)
        logger._print("\t\t\t\t%20s: %.2f" % (k, summaries_table[k][0]) + " +/- %.4f" % summaries_table[k][1], os.path.join(test_opts.out_folder, "logs.txt"))

    with open(os.path.join(test_opts.out_folder, "test_summary.json"), "w") as fp:
        json.dump(summaries, fp, indent=4)
    with open(os.path.join(test_opts.out_folder, "test_summary_table.json"), "w") as fp:
        json.dump(summaries_table, fp, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_folder", help="Path to data paths yaml file", default=None)
    parser.add_argument("--data-paths-config", help="Path to data paths yaml file", default="../../data_paths.yml")
    parser.add_argument("--data-path", default=None, help="explicit location of the data folder, if None use config file.")
    parser.add_argument("--data_dir", default="../../data/", help="Folder containing the supplementary data")
    parser.add_argument("--workers", default=2, type=int, help="number of data loading workers")
    parser.add_argument("--gpu", default=0, type=int, help="GPU id to use.")
    test_opts = parser.parse_arg