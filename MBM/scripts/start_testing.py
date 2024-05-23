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
        if opts.data == "cifar-100":
            level_wise_targets = get_targets(torch.arange(len(classes))) + (torch.arange(len(classes)).to(opts.gpu), )
        elif opts.data == "inaturalist19-224":
            level_wise_targets = get_target_l7(torch.arange(len(classes))) + (torch.arange(len(classes)).to(opts.gpu), )
        elif opts.data == "tiered-imagenet-224":
            level_wise_targets = get_target_l12(torch.arange(len(classes))) + (torch.arange(len(classes)).to(opts.gpu), )
        else:
            raise Exception("datset not supported")

        def get_nodes_at_each_level(tree):
            levels = []
            def traverse(node, depth=0):
                if len(levels) <= depth:
                    levels.append([])  # Create a new level if it doesn't exist
                if not isinstance(node, str):
                    levels[depth].append(node.label())
                    for child in node:
                        traverse(child, depth + 1)
                else:
                    levels[depth].append(str(node))  # Leaf nodes are strings in NLTK trees

            traverse(tree)
            return levels

        def calculate_max_depth(tree):
            if isinstance(tree, str):  # Leaf node
                return 0
            return 1 + max(calculate_max_depth(child) for child in tree)

        def add_dummy_nodes(tree, current_depth=0, max_depth=None):
            if max_depth is None:
                max_depth = calculate_max_depth(tree)

            if isinstance(tree, str) and current_depth == max_depth:  # Leaf node
                return
            # if isinstance(tree, str):
            #     print(tree, current_depth)
            #     import pdb; pdb.set_trace()

            for i, child in enumerate(tree):
                if isinstance(child, str):  # If the child is a leaf node
                    if current_depth < max_depth - 1:  # Add a dummy node if not at max depth
                        tree[i] = Tree(child, [child])
                        child = tree[i]
                    add_dummy_nodes(child, current_depth + 1, max_depth)
                else:
                    add_dummy_nodes(child, current_depth + 1, max_depth)

        add_dummy_nodes(hierarchy)

        node_id_to_label = get_nodes_at_each_level(hierarchy)
        num_classes = 0
        level_wise_nodes = {}
        for i, level in enumerate(level_wise_targets):
            unique_nodes = level.unique()
            num_classes += len(unique_nodes)
            level_wise_nodes[i + 1] = sorted(unique_nodes.cpu().numpy())
        opts.num_classes = num_classes

        node_embeddings = {}
        max_level = max(list(level_wise_nodes.keys()))
        for level in level_wise_nodes:
            if level < max_level:
                for i, node in enumerate(level_wise_nodes[level]):
                    # node = f'L{level}-{node}'
                    node = f'L{level}-{node_id_to_label[level][node]}'
                    encoded_arr = np.zeros((len(level_wise_nodes[level]), len(level_wise_nodes[level])), dtype=int)
                    encoded_arr[i][i] = 1.
                    node_embeddings[node] = encoded_arr[i]
            else:
                for i, node in enumerate(classes):
                    encoded_arr = np.zeros((len(classes), len(classes)), dtype=int)
                    encoded_arr[i][i] = 1.
                    node_embeddings[node] = encoded_arr[i]

        leaf_node_embeddings = [[]] * len(classes)
        leaf_node_masks = [[]] * max_level
        leaf_values = hierarchy.leaves()
        for class_ in classes:
            class_idx = classes.index(class_)
            leaf_node_embeddings[class_idx] = []
            leaf_index = leaf_values.index(class_)
            tree_location = hierarchy.leaf_treeposition(leaf_index)
            for level, i in enumerate(range(len(tree_location))):
                try:
                    label = f'L{level+1}-{hierarchy[tree_location[:i + 1]].label()}'
                except:
                    label = hierarchy[tree_location[:i + 1]]
                leaf_node_embeddings[class_idx].extend(node_embeddings[label])
                leaf_node_masks[level] = np.zeros(opts.num_classes)
                start_index = 0
                for nested_level in range(1, level + 1):
                    start_index += len(level_wise_nodes[nested_level])
                end_index = start_index + len(level_wise_nodes[level + 1])
                leaf_node_masks[level][start_index: end_index] += 1
            leaf_node_masks[level] = np.zeros(opts.num_classes)
            leaf_node_masks[level][-len(classes):] += 1
        leaf_node_embeddings = torch.tensor(leaf_node_embeddings, device=opts.gpu, dtype=torch.float32)
        leaf_node_masks = torch.tensor(leaf_node_masks, device=opts.gpu, dtype=torch.float32)

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
        def loss_func(logits, labels, m=0):
            out = torch.zeros((labels.shape[0], int(leaf_node_masks[max_level - 1].sum().item())),
                              device=opts.gpu,
                              dtype=torch.float32)
            for j, y_label in enumerate(leaf_node_embeddings):
                out[:, j] = -torch.norm(logits * (1 - y_label), dim=1)
            margin = 1 - torch.zeros_like(out).scatter_(1, labels.unsqueeze(1), m)
            loss_ce = F.cross_entropy(out * margin, labels, reduce=False)
            loss = loss_ce.mean()
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
    test_opts = parser.parse_args()

    main(test_opts)
