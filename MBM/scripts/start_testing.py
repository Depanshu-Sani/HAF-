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
        if opts.data == "cifar-100":
            max_level = 5
            # level_wise_targets = get_targets(torch.arange(len(classes))) + (torch.arange(len(classes)).to(opts.gpu), )
        elif opts.data == "inaturalist19-224":
            max_level = 7
            # level_wise_targets = get_target_l7(torch.arange(len(classes))) + (torch.arange(len(classes)).to(opts.gpu), )
        elif opts.data == "tiered-imagenet-224":
            max_level = 12
            # level_wise_targets = get_target_l12(torch.arange(len(classes))) + (torch.arange(len(classes)).to(opts.gpu), )
        else:
            raise Exception("datset not supported")

        distance_matrix = torch.zeros((len(classes), len(classes)), device=opts.gpu)
        for c1 in classes:
            for c2 in classes:
                distance_matrix[classes.index(c1), classes.index(c2)] = distances[(c1, c2)]
        distance_matrix = distance_matrix.max() - distance_matrix
        opts.num_classes += (opts.expand_feat_dim * len(classes))

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

        node_to_id = map_tree_to_ids_bfs(hierarchy)
        opts.num_classes = len(node_to_id) + opts.expand_feat_dim * len(classes)

        def find_siblings(tree):
            # Initialize a dictionary to store siblings of leaf nodes
            siblings_dict = {}

            # Function to recursively find siblings of leaf nodes
            def get_siblings(subtree, parent=None):
                if isinstance(subtree, Tree):
                    for child in subtree:
                        get_siblings(child, subtree)
                else:
                    # If it's a leaf node, find its siblings
                    if parent:
                        siblings = [classes.index(child) for child in parent if child != subtree]
                        siblings_dict[classes.index(subtree)] = siblings

            # Start the recursive function
            get_siblings(tree)

            return siblings_dict

        leaf_siblings = find_siblings(hierarchy)
        min_n_siblings = min([len(leaf_siblings[k]) for k in leaf_siblings.keys()])
        siblings = torch.full((len(classes), min_n_siblings), -1)
        for k in leaf_siblings.keys():
            siblings[k] = torch.tensor(leaf_siblings[k][:min_n_siblings], device=opts.gpu)

        def get_distances(logits, batch_size=opts.batch_size, dim=opts.num_classes):
            n_classes = len(classes)
            # Step 1: Expand logits to shape (batch_size, n_classes, dim)
            logits_expanded = logits.unsqueeze(1).expand(batch_size, n_classes, dim)  # Shape (batch_size, n_classes, dim)

            # Step 2: Apply the projection matrices to each point in logits
            # Matrix multiplication of shape (batch_size, n_classes, dim) @ (n_classes, dim, dim) -> (batch_size, n_classes, dim)
            projected_points = torch.einsum('bcd,bced->bce', logits_expanded, projections_expanded)

            # Step 3: Compute the Euclidean distance ||logits - projected_points||
            # Shape (batch_size, n_classes, dim) - (batch_size, n_classes, dim) -> (batch_size, n_classes, dim) -> (batch_size, n_classes)
            distances = torch.norm(logits_expanded - projected_points, dim=2)
            return distances

        def get_reg_loss(out_distance, projection_labels):
            normalized_out_distance = ((out_distance - out_distance.min(dim=1)[0][:, None]) / (out_distance.max(dim=1)[0][:, None] - out_distance.min(dim=1)[0][:, None])) * max_level
            min_distances = distance_matrix[
                projection_labels.unsqueeze(1), torch.arange(out_distance.shape[1], device=out_distance.device)]

            # Create a mask for the range (min_distance, min_distance + 1)
            mask = (normalized_out_distance > min_distances) & (normalized_out_distance < min_distances + 1)

            # Initialize clamped_norm with absolute differences
            clamped_norm = torch.abs(normalized_out_distance - (min_distances + 0.5))

            # Set values within the specified range to zero
            clamped_norm[mask] = 0
            # import pdb; pdb.set_trace()

            return clamped_norm.sum(1) / clamped_norm.shape[0]

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
            out = -get_distances(logits)
            # cross-entropy loss
            loss_ce = F.cross_entropy(out, labels, reduce=False)
            loss_reg = get_reg_loss(-out, labels)
            loss_true = -out[range(labels.shape[0]), labels]
            dist_siblings = -out[torch.arange(labels.shape[0]).unsqueeze(1).expand(labels.shape[0], siblings.shape[1]), siblings[labels]]
            intra_class_dist = torch.concat((loss_true[:, None], dist_siblings), dim=1)
            loss_ce_intra = F.cross_entropy(-intra_class_dist, torch.zeros(labels.shape, device=opts.gpu, dtype=int))
            loss_siblings = dist_siblings
            mask = torch.ones((labels.shape[0], out.shape[1]), dtype=torch.bool)
            mask[torch.arange(labels.shape[0]), labels] = False
            for i in range(labels.shape[0]):
                mask[i, siblings[labels][i]] = False
            dist_not_siblings = -out[mask].view(labels.shape[0], -1)
            # print(siblings)
            # distance between the true hyperplane (gt) and feature vector
            # loss_dist = -out[range(labels.shape[0]), labels]
            # length of feature vectors
            # loss_norm = m - torch.clamp(torch.norm(logits, dim=1), 0, m)
            # loss = loss_ce.mean() + loss_dist.mean() + loss_norm.mean()
            loss = loss_ce.mean()
            if log == 0:
                print(f"CE: {loss_ce.mean().item()} | CE-Intra: {loss_ce_intra.mean().item()} | True: {loss_true.mean().item()} | Siblings: {loss_siblings.min(1)[0].mean().item()} | Not Siblings: {dist_not_siblings.min(1)[0].mean().item()}")
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
        if opts.feature_space == 'haf++':
            projections = checkpoint["projections"]
            projections_expanded = projections.unsqueeze(0).expand(opts.batch_size, len(classes), opts.num_classes, opts.num_classes)
            print("loaded level_wise_projections")
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
