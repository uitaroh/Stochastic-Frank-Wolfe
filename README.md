# Pruning DNNs with Stochastic Frank-Wolfe
## Table of contents

* [Overview](#overview)
* [Technologies](#technologies)
* [Experiments](#experiments)
* [Loading data](#loading-data)
* [Optimizers](#optimizers)
* [Pruning](#pruning)
* [Conclusion](#conclusion)

## Overview
We study the paper ”Learning Pruning-Friendly Networks via Frank-Wolfe: One-Shot,
Any-Sparsity, And No Retraining” by Lu et al, which presents a new approach for
training a large deep neural network (DNN) that only requires one training session. The trained
DNN can then be pruned to any desired level of sparsity while still maintaining comparable accuracy,
without the need for any additional retraining. Traditional methods typically involve iterative pruning
and retraining, which not only adds a significant amount of additional work beyond the original DNN
training but can also be sensitive to retraining hyperparameters. The central idea is to reframe DNN
training as an explicit pruning-aware process, which is formulated with an auxiliary K-sparse polytope
constraint. This encourages network weights to lie within a convex hull spanned by K-sparse vectors,
potentially resulting in weight matrices that are more sparse. 

Consequently, we planned to personally code these methods by replicating the experiments done in
the paper.Unfortunately, due to time constraints and the limited power of our personal mundane computers,
we opted for a normal, simpler CNN, in order to arrive at the same conclusions as the original authors,
while having lower accuracy scores.

## Technologies
Project created in google colab with:
* python 3.8.10
Main libraries:

* torch
* tensorflow
* pandas
* sklearn
* matplotlib
* numpy

## Experiments
The experiments from the paper  **"Learning Pruning-Friendly Networks via Frank-Wolfe: One-Shot, Any-
Sparsity, And No Retraining"*, conducted with Res-Net-18 and VCG-
16 on the datasets CIFAR-10 and TinyImageNet. 

The running of the entire model performed by the paper’s authors would take hundreds of hours, which was
obviously not empirically doable in our case.
Therefore we did implement a simplified version of it, with our own model. Our results are obviously much worse in terms of accuracy, but serve well the purpose of comparing results between optimizers.

To make our code easier to read, we opted for the use
of OOP instead of procedural programming. Creating
our own classes, objects and methods makes
our code easily reusable and extensible, whilst
being able to extend the functionality of already
known packages.

## Loading data
On both datasets we used data augmentation techniques so that our models generalize
better. Moreover, all images have been rescaled
and normalized, in order to reduce data skewness.

## Optimizers
We compare the classical SGD (with extra retraining costs) with our SFW optimizer. For both, a learning rate of 0.1 is optimal. When implementing SFW, we found that a momentum of 0.9 yields the best results for these particular datasets. Moreover, we made it so that one can experiment with different types of constraints: K-sparse polytopes and unconstrained weights.

## Pruning
For the implementation of pruning we use the library prune from Pytorch (torch.nn.utils.prune). This library allows us to implement both unstructured and structured pruning. We chose the pruning technique the paper addresses as more efficient, namely weight-magnitude unstructured pruning (prune.l1 unstructured(module, name=’weight’, amount= 0.1)). This type of pruning does not eliminate connections among layers, instead it sets them close to zero, creating a sort of ’mask’. These weights are chosen through the weight-magnitude criterion, under which less important weights are pushed smaller, while important ones are enhanced. This process is further aided by restricting the feasible parameter j in a convex region C (a K-sparse polytope), which results in more sparse weight matrices and will be more pruning-friendly for only a small percentage of the weights (the ones of large magnitudes).

## Conclusion

All in all, with our simpler model, which remains faithful to the implementations of most requests of
section (5) of the original paper: SFW/SGD, constrained/unconstrained, with/without pruning, with/without
retraining. It is clear that training the network weights within in a convex polytope spanned by
K-sparse vectors, yields much lower weights, yet “smoother” magnitude distribution, that is more
amendable to gradual weight removal towards any pruning ratio.
