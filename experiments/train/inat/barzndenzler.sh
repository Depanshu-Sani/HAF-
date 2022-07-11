python main.py --start training --arch resnet18 --loss cosine-plus-xent --barzdenzler True --train_backbone_after 0 --use_2fc False --use_fc_batchnorm True --weight_decay_fc 5e-4 --lr 1e-4 --lr_fc 1e-4 --data inaturalist19-224 --workers 16 --output out/inat/barzdenzler-cosine-plux-xent --epochs 100 --seed 0