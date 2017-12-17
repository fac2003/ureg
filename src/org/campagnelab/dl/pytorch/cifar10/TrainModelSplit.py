import itertools
import math
import os
from random import uniform, randint, random, shuffle

import numpy
import torch
from torch.autograd import Variable
from torch.backends import cudnn
from torch.nn import MSELoss, MultiLabelSoftMarginLoss
from torchnet.meter import ConfusionMeter

from org.campagnelab.dl.pytorch.cifar10.AccuracyHelper import AccuracyHelper
from org.campagnelab.dl.pytorch.cifar10.FloatHelper import FloatHelper
from org.campagnelab.dl.pytorch.cifar10.LRHelper import LearningRateHelper
from org.campagnelab.dl.pytorch.cifar10.LossHelper import LossHelper
from org.campagnelab.dl.pytorch.cifar10.PerformanceList import PerformanceList
from org.campagnelab.dl.pytorch.cifar10.utils import progress_bar, grad_norm, batch, init_params
from org.campagnelab.dl.pytorch.ureg.LRSchedules import construct_scheduler


def _format_nice(n):
    try:
        if n == int(n):
            return str(n)
        if n == float(n):
            if n < 0.001:
                return "{0:.3E}".format(n)
            else:
                return "{0:.4f}".format(n)
    except:
        return str(n)


def rmse_dim_1(y, y_hat):
    """Compute root mean squared error"""
    return torch.sqrt(torch.mean((y - y_hat).pow(2), 1))


flatten = lambda l: [item for sublist in l for item in sublist]


def print_params(epoch, net):
    params = []
    for param in net.parameters():
        params += [p for p in param.view(-1).data]
    print("epoch=" + str(epoch) + " " + " ".join(map(str, params)))


class TrainModelSplit:
    """Train a model using the split unsupervised approach."""

    def __init__(self, args, problem, use_cuda):
        """
        Initialize model training with arguments and problem.

        :param args command line arguments.
        :param use_cuda When True, use the GPU.
         """

        self.max_regularization_examples = args.num_shaving if hasattr(args, "num_shaving") else 0
        self.max_validation_examples = args.num_validation if hasattr(args, "num_validation") else 0
        self.max_training_examples = args.num_training if hasattr(args, "num_training") else 0
        max_examples_per_epoch = args.max_examples_per_epoch if hasattr(args, 'max_examples_per_epoch') else None
        self.max_examples_per_epoch = max_examples_per_epoch if max_examples_per_epoch is not None else self.max_regularization_examples
        self.criterion = problem.loss_function()
        self.criterion_multi_label = MultiLabelSoftMarginLoss()  # problem.loss_function()

        self.split_enabled = args.split if hasattr(args, "split") else None
        self.args = args
        self.problem = problem
        self.best_acc = 0
        self.start_epoch = 0
        self.use_cuda = use_cuda
        self.mini_batch_size = problem.mini_batch_size()
        self.net = None
        self.optimizer_training = None
        self.scheduler_train = None
        self.unsuploader = self.problem.reg_loader()
        self.trainloader = self.problem.train_loader()
        self.testloader = self.problem.test_loader()
        self.split = None
        self.is_parallel = False
        self.best_performance_metrics = None
        self.failed_to_improve = 0
        self.confusion_matrix = None
        self.best_model_confusion_matrix = None

    def init_model(self, create_model_function):
        """Resume training if necessary (args.--resume flag is True), or call the
        create_model_function to initialize a new model. This function must be called
        before train.

        The create_model_function takes one argument: the name of the model to be
        created.
        """
        args = self.args
        mini_batch_size = self.mini_batch_size
        # restrict limits to actual size of datasets:
        training_set_length = (len(self.problem.train_loader())) * mini_batch_size
        if hasattr(args, 'num_training') and args.num_training > training_set_length:
            args.num_training = training_set_length
        unsup_set_length = (len(self.problem.reg_loader())) * mini_batch_size
        if hasattr(args, 'num_shaving') and args.num_shaving > unsup_set_length:
            args.num_shaving = unsup_set_length
        test_set_length = (len(self.problem.test_loader())) * mini_batch_size
        if hasattr(args, 'num_validation') and args.num_validation > test_set_length:
            args.num_validation = test_set_length

        self.max_regularization_examples = args.num_shaving if hasattr(args, 'num_shaving') else 0
        self.max_validation_examples = args.num_validation if hasattr(args, 'num_validation') else 0
        self.max_training_examples = args.num_training if hasattr(args, 'num_training') else 0
        self.unsuploader = self.problem.reg_loader()
        model_built = False
        self.best_performance_metrics = None
        self.best_model = None

        def rmse(y, y_hat):
            """Compute root mean squared error"""
            return torch.sqrt(torch.mean((y - y_hat).pow(2)))

        self.agreement_loss = MSELoss()

        if args.resume:
            # Load checkpoint.

            print('==> Resuming from checkpoint..')

            assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
            checkpoint = None
            try:
                checkpoint = torch.load('./checkpoint/ckpt_{}.t7'.format(args.checkpoint_key))
            except                FileNotFoundError:
                pass
            if checkpoint is not None:

                self.net = checkpoint['net']
                self.best_acc = checkpoint['acc']
                self.start_epoch = checkpoint['epoch']
                self.split_enabled = checkpoint['split']
                self.best_model = checkpoint['best-model']
                self.best_model_confusion_matrix = checkpoint['confusion-matrix']
                model_built = True
            else:
                print("Could not load model checkpoint, unable to --resume.")
                model_built = False
        else:
            if hasattr(self.args, 'load_pre_trained_model') and self.args.load_pre_trained_model:
                self.net=self.load_pretrained_model()
                model_built=self.net is not None

        if not model_built:
            print('==> Building model {}'.format(args.model))

            self.net = create_model_function(args.model, self.problem)

        if self.use_cuda:
            self.net.cuda()
        cudnn.benchmark = True

        self.optimizer_training = torch.optim.SGD(self.net.parameters(), lr=args.lr, momentum=args.momentum,
                                                  weight_decay=args.L2)

        self.scheduler_train = \
            construct_scheduler(self.optimizer_training, 'max', factor=0.5,
                                lr_patience=self.args.lr_patience,
                                ureg_reset_every_n_epoch=self.args.reset_lr_every_n_epochs
                                if hasattr(self.args, 'reset_lr_every_n_epochs')
                                else None)



    def train(self, epoch,
              performance_estimators=None,
              train_supervised_model=True,
              train_split=False,
              ):

        if performance_estimators is None:
            performance_estimators = PerformanceList()
            performance_estimators += [LossHelper("optimized_loss")]
            performance_estimators += [LossHelper("train_loss")]
            if train_split:
                performance_estimators += [LossHelper("split_loss")]
            performance_estimators += [AccuracyHelper("train_")]
            performance_estimators += [FloatHelper("train_grad_norm")]
            performance_estimators += [FloatHelper("factor")]
            print('\nTraining, epoch: %d' % epoch)
        self.net.train()
        supervised_grad_norm = 1.
        for performance_estimator in performance_estimators:
            performance_estimator.init_performance_metrics()

        unsupervised_loss_acc = 0
        num_batches = 0
        train_loader_subset = self.problem.train_loader_subset_range(0, self.args.num_training)
        unsuploader_shuffled = self.problem.reg_loader_subset_range(0, self.args.num_shaving)
        unsupiter = itertools.cycle(unsuploader_shuffled)
        performance_estimators.set_metric(epoch, "factor", self.args.factor)

        for batch_idx, (inputs, targets) in enumerate(train_loader_subset):
            num_batches += 1

            if self.use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()

            inputs, targets = Variable(inputs), Variable(targets, requires_grad=False)
            # outputs used to calculate the loss of the supervised model
            # must be done with the model prior to regularization:
            self.net.train()
            self.optimizer_training.zero_grad()
            outputs = self.net(inputs)
            average_unsupervised_loss = 0
            if train_split:

                # obtain an unsupervised sample, put it in uinputs autograd Variable:
                # first, read a minibatch from the unsupervised dataset:
                ufeatures, _ = next(unsupiter)

                if self.use_cuda: ufeatures = ufeatures.cuda()
                # then use it to calculate the unsupervised regularization contribution to the loss:

                unsup_train_loss = self.split_loss(ufeatures)
                if unsup_train_loss is not None:
                    performance_estimators.set_metric(batch_idx, "split_loss", unsup_train_loss.data[0])
                    average_unsupervised_loss = unsup_train_loss

            if train_supervised_model:
                # if self.ureg._which_one_model is not None:
                #    self.ureg.estimate_example_weights(inputs)

                supervised_loss = self.criterion(outputs, targets)
                alpha = self.args.factor
                optimized_loss = supervised_loss * ( alpha) + (1.-alpha) * average_unsupervised_loss
                optimized_loss.backward()
                self.optimizer_training.step()
                supervised_grad_norm = grad_norm(self.net.parameters())
                performance_estimators.set_metric(batch_idx, "train_grad_norm", supervised_grad_norm)

                performance_estimators.set_metric_with_outputs(batch_idx, "train_loss", supervised_loss.data[0],
                                                               outputs, targets)
                performance_estimators.set_metric_with_outputs(batch_idx, "optimized_loss", optimized_loss.data[0],
                                                               outputs, targets)
                performance_estimators.set_metric_with_outputs(batch_idx, "train_accuracy", supervised_loss.data[0],
                                                               outputs, targets)

            progress_bar(batch_idx * self.mini_batch_size,
                         min(self.max_regularization_examples, self.max_training_examples),
                         " ".join([performance_estimator.progress_message() for performance_estimator in
                                   performance_estimators]))

            if (batch_idx + 1) * self.mini_batch_size > self.max_training_examples:
                break

            print("\n")

        # increase factor by 10% at the end of each epoch:
        self.args.factor *= self.args.increase_decrease
        return performance_estimators

    def pre_train_with_half_images(self, num_cycles=100, epochs_per_cycle=10,
                                   performance_estimators=None,
                                   num_classes=None):
        """
        This method pretrains a network: - Try a new spin on split: Select a subset of unsup images.
         Assign a random class to each image. Split each image in two and give each half the same class.
         Randomize the set of half images. Train the network to predict the class of the image from the
         half image in competition with all the other images in the random subset. Train to convergence
         or for some set number of epoch, pick a different subset and repeat. Use this as pre-training
         before going into the supervised task (which can use mixup).
         Strategy inspired by exemplar convolutional neural net (Dosovitskiy et al 2015).
        :param num_cycles: number of pre-training cycles.
        :param performance_estimators:
        :param num_classes: number of classes to put in competition.
        :return:
        """
        if num_classes is None:
            num_classes = self.problem.num_classes()
        print("Pretraining with num classes={} ".format(num_classes))
        self.net.train()
        if performance_estimators is None:
            performance_estimators = PerformanceList()
            performance_estimators += [FloatHelper("pretrain_loss")]
            performance_estimators += [FloatHelper("pretrain_start_accuracy")]
            performance_estimators += [AccuracyHelper("pretrain_")]
        # we create an optimizer that changes only the classifier part of the model:

        for cycle in range(0, num_cycles):
            self.net.remake_classifier(num_classes, self.use_cuda)
            optimizer_classifier = torch.optim.SGD(self.net.get_classifier().parameters(),
                                                   lr=self.args.lr, momentum=self.args.momentum,
                                                   weight_decay=self.args.L2)

            unsuploader_shuffled = self.problem.reg_loader_subset_range(0, self.args.num_shaving)
            # construct the training set:
            pre_training_set = []
            offset = 0
            for class_index, (unsup_inputs, _) in enumerate(unsuploader_shuffled):

                (image1, image2) = self.half_images(unsup_inputs, slope=self.get_random_slope())

                class_indices = torch.from_numpy(numpy.array(range(offset, offset + self.mini_batch_size)))

                pre_training_set += [(image1.data, class_indices)]
                pre_training_set += [(image2.data, class_indices)]
                offset += self.mini_batch_size
                if offset >= num_classes:
                    break

            # shuffle the pre-training set:
            shuffle(pre_training_set)
            # we train the classifier part only when the cycle is not the first one:
            train_classifier = cycle > 0
            best_acc = 0
            for epoch in range(0, epochs_per_cycle):
                for performance_estimator in performance_estimators:
                    performance_estimator.init_performance_metrics()
                optimizer = optimizer_classifier if train_classifier else self.optimizer_training
                # init_params(self.net.classifier)

                for (batch_idx, (half_images, class_indices)) in enumerate(pre_training_set):
                    class_indices = class_indices.type(torch.LongTensor)
                    if self.use_cuda:
                        half_images = half_images.cuda()
                        class_indices = class_indices.cuda()

                    inputs, targets = Variable(half_images), Variable(class_indices, requires_grad=False)
                    optimizer.zero_grad()

                    outputs = self.net(inputs)
                    pre_training_loss = self.criterion(outputs, targets)
                    pre_training_loss.backward()
                    optimizer.step()
                    performance_estimators.set_metric(batch_idx, "pretrain_loss", pre_training_loss.data[0])
                    performance_estimators.set_metric_with_outputs(batch_idx, "pretrain_accuracy",
                                                                   pre_training_loss,
                                                                   outputs, targets)

                pretrain_acc = performance_estimators.get_metric("pretrain_accuracy")

                if train_classifier:
                    if epoch > 0 and abs(pretrain_acc - best_acc) < 1E-3 and pretrain_acc <= best_acc:
                        # the classifier has converged. Now enable training the model:
                        train_classifier = False
                        print("Finished training the classifier with start_acc=" + str(pretrain_acc))
                        best_acc =0
                    else:
                        best_acc = pretrain_acc
                        performance_estimators.set_metric(epoch, "pretrain_start_accuracy", pretrain_acc)
                if pretrain_acc >= 100.0:
                    break
                progress_bar(epoch, (epochs_per_cycle),
                             msg=performance_estimators.progress_message(["pretrain_accuracy", "pretrain_loss"]))
                # print()
                # print("epoch {} pretrainin-loss={}".format(epoch, performance_estimators.get_metric("pretrain_loss")))

            print("cycle {} pretraining-loss={} start_accuracy={} accuracy={}".format(cycle,
                                                                                      performance_estimators.get_metric(
                                                                                          "pretrain_loss"),
                                                                                      performance_estimators.get_metric(
                                                                                          "pretrain_start_accuracy"),
                                                                                      pretrain_acc))
            self.net.remake_classifier(self.problem.num_classes(), self.use_cuda)

    def train_mixup(self, epoch,
                    performance_estimators=None,
                    train_supervised_model=True,
                    alpha=0.5,
                    ratio_unsup=0,
                    ):

        if performance_estimators is None:
            performance_estimators = PerformanceList()
            performance_estimators += [LossHelper("optimized_loss")]
            performance_estimators += [LossHelper("train_loss")]
            # performance_estimators += [AccuracyHelper("train_")]
            performance_estimators += [FloatHelper("train_grad_norm")]
            performance_estimators += [FloatHelper("alpha")]
            performance_estimators += [FloatHelper("unsup_proportion")]
            print('\nTraining, epoch: %d' % epoch)
        self.net.train()
        supervised_grad_norm = 1.
        for performance_estimator in performance_estimators:
            performance_estimator.init_performance_metrics()

        unsupervised_loss_acc = 0
        num_batches = 0
        train_loader_subset = self.problem.train_loader_subset_range(0, self.args.num_training)
        sec_train_loader_subset = self.problem.train_loader_subset_range(0, self.args.num_training)
        unsuploader_shuffled = self.problem.reg_loader_subset_range(0, self.args.num_shaving)
        unsupiter = itertools.cycle(unsuploader_shuffled)

        performance_estimators.set_metric(epoch, "alpha", alpha)
        performance_estimators.set_metric(epoch, "unsup_proportion", ratio_unsup)

        for batch_idx, ((inputs1, targets1),
                        (inputs2, targets2),
                        (uinputs1, _)) in enumerate(zip(train_loader_subset,
                                                        sec_train_loader_subset,
                                                        unsupiter)):
            num_batches += 1

            use_unsup = random() < ratio_unsup

            if use_unsup:
                # use an example from the unsupervised set to mixup with inputs1:
                inputs2 = uinputs1

            if self.use_cuda:
                inputs1 = inputs1.cuda()
                inputs2 = inputs2.cuda()

            inputs, targets = self.mixup_inputs_targets(alpha, inputs1, inputs2, targets1, targets2)

            # outputs used to calculate the loss of the supervised model
            # must be done with the model prior to regularization:
            self.net.train()
            self.optimizer_training.zero_grad()
            outputs = self.net(inputs)

            if train_supervised_model:
                supervised_loss = self.criterion_multi_label(outputs, targets)
                optimized_loss = supervised_loss
                optimized_loss.backward()
                self.optimizer_training.step()
                supervised_grad_norm = grad_norm(self.net.parameters())
                performance_estimators.set_metric(batch_idx, "train_grad_norm", supervised_grad_norm)
                performance_estimators.set_metric(batch_idx, "optimized_loss", optimized_loss.data[0])

                performance_estimators.set_metric_with_outputs(batch_idx, "train_loss", supervised_loss.data[0],
                                                               outputs, targets)
                # performance_estimators.set_metric_with_outputs(batch_idx, "train_accuracy", supervised_loss.data[0],
                #                                               outputs, targets)

            progress_bar(batch_idx * self.mini_batch_size,
                         min(self.max_regularization_examples, self.max_training_examples),
                         " ".join([performance_estimator.progress_message() for performance_estimator in
                                   performance_estimators]))

            if (batch_idx + 1) * self.mini_batch_size > self.max_training_examples:
                break

            print("\n")

        return performance_estimators

    def mixup_inputs_targets(self, alpha, inputs1, inputs2, targets1, targets2):
        """" Implement mixup: combine inputs and targets in alpha amounts. """
        if self.args.one_mixup_per_batch:

            # one distinct lam scalar per example:
            lam = torch.from_numpy(numpy.random.beta(alpha, alpha, self.mini_batch_size)).type(torch.FloatTensor)
            targets1 = self.problem.one_hot(targets1)
            inputs_gpu = inputs2.cuda(self.args.second_gpu_index) if self.use_cuda else inputs2
            targets2 = self.dream_up_target2(inputs_gpu, targets2)
            inputs = torch.zeros(inputs1.size())
            targets = torch.zeros(targets1.size())
            if self.use_cuda:
                lam = lam.cuda()
                targets1 = targets1.cuda()
                targets2 = targets2.cuda()
                inputs = inputs.cuda()
                targets = targets.cuda()
            for example_index in range(0, self.mini_batch_size):
                inputs[example_index] = inputs1[example_index] * lam[example_index] + inputs2[example_index] * (
                        1. - lam[example_index])
                targets[example_index] = targets1[example_index] * lam[example_index] + targets2[example_index] * (
                        1. - lam[example_index])
        else:
            lam = numpy.random.beta(alpha, alpha)
            targets1 = self.problem.one_hot(targets1)
            inputs_gpu = inputs2.cuda(self.args.second_gpu_index) if self.use_cuda else inputs2
            targets2 = self.dream_up_target2(inputs_gpu, targets2)

            if self.use_cuda:
                targets1 = targets1.cuda()
                targets2 = targets2.cuda()

            inputs = inputs1 * lam + inputs2 * (1. - lam)
            targets = targets1 * lam + targets2 * (1. - lam)

        inputs, targets = Variable(inputs), Variable(targets, requires_grad=False)
        return inputs, targets

    def dream_up_target2(self, inputs_gpu, targets2):
        if self.args.label_strategy == "CERTAIN" or self.best_model is None:
            # we don't know the target on the unsup set, so we just let the training set make it up (this guess is correct
            # with probability 1/num_classes times):
            targets2 = self.problem.one_hot(targets2)
        elif self.args.label_strategy == "UNIFORM":
            # we use uniform labels that represent the expectation of the correct answer if classes where equally represented:
            targets2 = torch.ones(self.mini_batch_size, self.problem.num_classes()) / self.problem.num_classes()
        elif self.args.label_strategy == "MODEL":
            # we use the best model we trained so far to predict the outputs. These labels will overfit to the
            # training set as training progresses:
            best_model_output = self.best_model(Variable(inputs_gpu, requires_grad=False))
            _, predicted = torch.max(best_model_output.data)
            targets2 = best_model_output.data
        elif self.args.label_strategy == "VAL_CONFUSION":
            # we use the best model we trained so far to predict the outputs. These labels will overfit to the
            # training set as training progresses:
            best_model_output = self.best_model(Variable(inputs_gpu, requires_grad=False))
            _, predicted = torch.max(best_model_output.data, 1)
            predicted = predicted.type(torch.LongTensor)
            if self.use_cuda:
                predicted = predicted.cuda(self.args.second_gpu_index)
            # we use the confusion matrix to set the target on the unsupervised example. We simply normalize the
            # row of the confusion matrix corresponding to the  label predicted by the best model:
            select = torch.index_select(self.best_model_confusion_matrix, dim=0, index=predicted).type(
                torch.FloatTensor)
            targets2 = torch.renorm(select, p=1, dim=0, maxnorm=1)
            # print("normalized: "+str(targets2))
        elif self.args.label_strategy == "VAL_CONFUSION_SAMPLING":
            # we use the best model we trained so far to predict the outputs. These labels will overfit to the
            # training set as training progresses:
            self.best_model.eval()
            best_model_output = self.best_model(Variable(inputs_gpu, requires_grad=False))
            _, predicted = torch.max(best_model_output.data, 1)
            predicted = predicted.type(torch.LongTensor)

            if self.use_cuda:
                predicted = predicted.cuda(self.args.second_gpu_index)
            # we lookup the confusion matrix, but instead of using it directly as output, we sample from it to
            # create a one-hot encoded unsupervised output label:
            select = torch.index_select(self.best_model_confusion_matrix, dim=0, index=predicted).type(
                torch.FloatTensor)
            if random() > (1 - self.args.exploration_rate):
                # remove the class with the most correct answers from consideration (set its probability to zero):
                max_value, max_index = select.max(dim=1)
                select.scatter_(dim=1, index=max_index.view(self.mini_batch_size, 1), value=0.)

            normalized_confusion_matrix = torch.renorm(select, p=1, dim=0, maxnorm=1)
            confusion_cumulative = torch.cumsum(normalized_confusion_matrix, dim=1)
            class_indices = []
            random_choice = torch.rand(self.mini_batch_size)
            for index, example in enumerate(confusion_cumulative):
                nonzero = example.ge(random_choice[index]).nonzero()
                if len(nonzero) > 0:
                    class_indices += [nonzero.min()]
                else:
                    class_indices += [self.problem.num_classes() - 1]

            targets2 = self.problem.one_hot(torch.from_numpy(numpy.array(class_indices)))
            # print("targets2: "+str(targets2))
        else:
            print("Incorrect label strategy name: " + self.args.label_strategy)
            exit(1)
        return targets2

    def test(self, epoch, performance_estimators=(LossHelper("test_loss"), AccuracyHelper("test_"))):
        print('\nTesting, epoch: %d' % epoch)

        self.net.eval()
        for performance_estimator in performance_estimators:
            performance_estimator.init_performance_metrics()
        cm = ConfusionMeter(self.problem.num_classes(), normalized=False)

        for batch_idx, (inputs, targets) in enumerate(self.problem.test_loader_range(0, self.args.num_validation)):

            if self.use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            inputs, targets = Variable(inputs, volatile=True), Variable(targets)
            outputs = self.net(inputs)
            loss = self.criterion(outputs, targets)
            # accumulate the confusion matrix:
            _, predicted = torch.max(outputs.data, 1)

            cm.add(predicted=predicted, target=targets.data)
            for performance_estimator in performance_estimators:
                performance_estimator.observe_performance_metric(batch_idx, loss.data[0], outputs, targets)

            progress_bar(batch_idx * self.mini_batch_size, self.max_validation_examples,
                         " ".join([performance_estimator.progress_message() for performance_estimator in
                                   performance_estimators]))

            if ((batch_idx + 1) * self.mini_batch_size) > self.max_validation_examples:
                break
        print()

        # Apply learning rate schedule:
        test_accuracy = self.get_metric(performance_estimators, "test_accuracy")
        assert test_accuracy is not None, "test_accuracy must be found among estimated performance metrics"
        if not self.args.constant_learning_rates:
            self.scheduler_train.step(test_accuracy, epoch)
        self.confusion_matrix = cm.value().transpose()
        return performance_estimators

    def log_performance_header(self, performance_estimators):
        global best_test_loss
        if (self.args.resume):
            return

        metrics = ["epoch", "checkpoint"]

        for performance_estimator in performance_estimators:
            metrics = metrics + performance_estimator.metric_names()

        if not self.args.resume:
            with open("all-perfs-{}.tsv".format(self.args.checkpoint_key), "w") as perf_file:
                perf_file.write("\t".join(map(str, metrics)))
                perf_file.write("\n")

            with open("best-perfs-{}.tsv".format(self.args.checkpoint_key), "w") as perf_file:
                perf_file.write("\t".join(map(str, metrics)))
                perf_file.write("\n")

    def log_performance_metrics(self, epoch, performance_estimators):
        """
        Log metrics and returns the best performance metrics seen so far.
        :param epoch:
        :param performance_estimators:
        :return: a list of performance metrics corresponding to the epoch where test accuracy was maximum.
        """
        metrics = [epoch, self.args.checkpoint_key]
        for performance_estimator in performance_estimators:
            metrics = metrics + performance_estimator.estimates_of_metric()
        early_stop = False
        with open("all-perfs-{}.tsv".format(self.args.checkpoint_key), "a") as perf_file:
            perf_file.write("\t".join(map(_format_nice, metrics)))
            perf_file.write("\n")
        if self.best_performance_metrics is None:
            self.best_performance_metrics = performance_estimators

        metric = self.get_metric(performance_estimators, "test_accuracy")
        if metric is not None and metric > self.best_acc:
            self.failed_to_improve = 0

            with open("best-perfs-{}.tsv".format(self.args.checkpoint_key), "a") as perf_file:
                perf_file.write("\t".join(map(_format_nice, metrics)))
                perf_file.write("\n")

        if metric is not None and metric >= self.best_acc:

            self.save_checkpoint(epoch, metric)
            self.best_performance_metrics = performance_estimators
            if self.args.mode == "mixup":
                if self.args.two_models:
                    # we load the best model we saved previously as a second model:
                    self.best_model = self.load_checkpoint()
                    if self.use_cuda:
                        self.best_model = self.best_model.cuda(self.args.second_gpu_index)
                else:
                    self.best_model = self.net

                self.best_model_confusion_matrix = torch.from_numpy(self.confusion_matrix)
                if self.use_cuda:
                    self.best_model_confusion_matrix = self.best_model_confusion_matrix.cuda(self.args.second_gpu_index)

        if metric is not None and metric <= self.best_acc:
            self.failed_to_improve += 1
            if self.failed_to_improve > self.args.abort_when_failed_to_improve:
                print("We failed to improve for {} epochs. Stopping here as requested.")
                early_stop = True  # request early stopping

        return early_stop, self.best_performance_metrics

    def get_metric(self, performance_estimators, metric_name):
        for pe in performance_estimators:
            metric = pe.get_metric(metric_name)
            if metric is not None:
                return metric
        return None

    def load_pretrained_model(self):
        assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
        checkpoint = None
        try:
            model_filename = './checkpoint/pretrained_{}.t7'.format(self.args.checkpoint_key)
            checkpoint = torch.load(model_filename)
        except FileNotFoundError:
            pass
        if checkpoint is not None:
            print("Loaded pre-trained model from "+model_filename)
            return checkpoint['net']
        else:
            print("Could not load pre-trained model checkpoint.")
            return None

    def save_pretrained_model(self):
        # Save pretrained model.

        print('Saving pre-trained model..')
        model = self.net
        model.eval()

        state = {
            'net': model.module if self.is_parallel else model,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/pretrained_{}.t7'.format(self.args.checkpoint_key))

    def save_checkpoint(self, epoch, acc):

        # Save checkpoint.

        if acc > self.best_acc:
            print('Saving..')
            model = self.net
            model.eval()

            state = {
                'net': model.module if self.is_parallel else model,
                'best-model': self.best_model,
                'confusion-matrix': self.best_model_confusion_matrix,
                'acc': acc,
                'epoch': epoch,
                'split': self.split_enabled,
            }
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            torch.save(state, './checkpoint/ckpt_{}.t7'.format(self.args.checkpoint_key))
            self.best_acc = acc

    def load_checkpoint(self):

        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        state = torch.load('./checkpoint/ckpt_{}.t7'.format(self.args.checkpoint_key))
        model = state['net']
        model.cpu()
        model.eval()
        return model

    def training_mixup(self):
        """Train the model in a completely supervised manner. Returns the performance obtained
           at the end of the configured training run.
        :return list of performance estimators that observed performance on the last epoch run.
        """
        header_written = False

        lr_train_helper = LearningRateHelper(scheduler=self.scheduler_train, learning_rate_name="train_lr")
        previous_test_perfs = None
        perfs = PerformanceList()
        for epoch in range(self.start_epoch, self.start_epoch + self.args.num_epochs):

            perfs = PerformanceList()
            perfs += [self.train_mixup(epoch,
                                       train_supervised_model=True,
                                       alpha=self.args.alpha,
                                       ratio_unsup=self.args.unsup_proportion
                                       )]
            # increase ratio_unsup by 10% at the end of each epoch:
            self.args.unsup_proportion *= self.args.increase_decrease
            if self.args.unsup_proportion > 1:
                self.args.unsup_proportion = 1
                self.args.alpha *= 1. / self.args.increase_decrease
            if self.args.unsup_proportion < 1E-5:
                self.args.unsup_proportion = 0
                self.args.alpha *= 1. / self.args.increase_decrease
            if self.args.alpha < 0:
                self.args.alpha = 0
            if self.args.alpha > 1:
                self.args.alpha = 1

            perfs += [(lr_train_helper,)]
            if previous_test_perfs is None or self.epoch_is_test_epoch(epoch):
                perfs += [self.test(epoch)]

            perfs = flatten(perfs)
            if (not header_written):
                header_written = True
                self.log_performance_header(perfs)

            early_stop, perfs = self.log_performance_metrics(epoch, perfs)
            if early_stop:
                # early stopping requested.
                return perfs

        return perfs

    def training_supervised(self):
        """Train the model in a completely supervised manner. Returns the performance obtained
           at the end of the configured training run.
        :return list of performance estimators that observed performance on the last epoch run.
        """
        header_written = False

        lr_train_helper = LearningRateHelper(scheduler=self.scheduler_train, learning_rate_name="train_lr")
        previous_test_perfs = None
        perfs = PerformanceList()
        for epoch in range(self.start_epoch, self.start_epoch + self.args.num_epochs):

            perfs = PerformanceList()
            perfs += [self.train(epoch,
                                 train_supervised_model=True,
                                 train_split=False
                                 )]
            perfs += [(lr_train_helper,)]
            if previous_test_perfs is None or self.epoch_is_test_epoch(epoch):
                perfs += [self.test(epoch)]

            perfs = flatten(perfs)
            if (not header_written):
                header_written = True
                self.log_performance_header(perfs)

            early_stop, perfs = self.log_performance_metrics(epoch, perfs)
            if early_stop:
                # early stopping requested.
                return perfs

        return perfs

    def training_split(self):
        """Train the model with a single pass through the training set using the unsup split strategy.
         Returns the performance obtained  at the end of the configured training run.
        :return list of performance estimators that observed performance on the last epoch run.
        """
        header_written = False

        lr_train_helper = LearningRateHelper(scheduler=self.scheduler_train, learning_rate_name="train_lr")
        previous_test_perfs = None

        perfs = []
        for epoch in range(self.start_epoch, self.start_epoch + self.args.num_epochs):

            perfs = []

            perfs += [self.train(epoch,
                                 train_supervised_model=True,
                                 train_split=True)]

            if previous_test_perfs is None or self.epoch_is_test_epoch(epoch):
                perfs += [self.test(epoch)]

            perfs += [(lr_train_helper,)]

            perfs = flatten(perfs)
            if (not header_written):
                header_written = True
                self.log_performance_header(perfs)

            early_stop, perfs = self.log_performance_metrics(epoch, perfs)
            if early_stop:
                # early stopping requested.
                return perfs

        return perfs

    def epoch_is_test_epoch(self, epoch):
        epoch_is_one_of_last_ten = epoch > (self.start_epoch + self.args.num_epochs - 10)
        return (epoch % self.args.test_every_n_epochs + 1) == 1 or epoch_is_one_of_last_ten

    def split_loss(self, uinputs):
        if self.use_cuda:
            uinputs = uinputs.cuda()
        slope = self.get_random_slope()
        (image_1, image_2) = self.half_images(uinputs, slope)
        answer_1 = self.net(image_1)
        answer_2 = self.net(image_2)
        targets = Variable(answer_2.data, requires_grad=False)
        return self.agreement_loss(answer_1, targets)

    def get_random_slope(self):
        pi = math.atan(1) * 4
        angle = uniform(pi, -pi)
        slope = math.tan(angle)
        return slope

    def half_images(self, uinputs, slope):
        def above_line(xp, yp, slope, b):
            # remember that y coordinate increase downward in images
            yl = slope * xp + b
            return yp < yl

        uinputs = Variable(uinputs, requires_grad=False)

        mask_up = torch.ByteTensor(uinputs.size()[2:])  # index 2 drops minibatch size and channel dimension.
        mask_down = torch.ByteTensor(uinputs.size()[2:])  # index 2 drops minibatch size and channel dimension.

        channels = uinputs.size()[1]
        width = uinputs.size()[2]
        height = uinputs.size()[3]
        # print("mask down-------------")
        # fill in the first channel:
        for x in range(0, width):
            for y in range(0, height):
                above_the_line = above_line(x - width / 2, y, slope=slope, b=height / 2.0)
                mask_up[x, y] = 1 if above_the_line else 0
                mask_down[x, y] = 0 if above_the_line else 1
                # print("." if mask_down[x, y] else " ",end="")
                # print("|")

        # print("-----------mask down")
        mask1 = torch.ByteTensor(uinputs.size()[1:])
        mask2 = torch.ByteTensor(uinputs.size()[1:])

        # if self.use_cuda:
        #     mask_up = mask_up.cuda()
        #     mask_down = mask_down.cuda()
        #     mask1 = mask1.cuda()
        #     mask2 = mask2.cuda()
        for c in range(0, channels):
            mask1[c] = mask_up
            mask2[c] = mask_down
        mask1 = Variable(mask1, requires_grad=False)
        mask2 = Variable(mask2, requires_grad=False)
        return uinputs.masked_fill(mask1, 0.), uinputs.masked_fill(mask2, 0.)
