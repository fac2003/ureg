import itertools
import math
import os
from random import uniform, randint

import torch
from torch.autograd import Variable
from torch.backends import cudnn

from org.campagnelab.dl.pytorch.cifar10.AccuracyHelper import AccuracyHelper
from org.campagnelab.dl.pytorch.cifar10.FloatHelper import FloatHelper
from org.campagnelab.dl.pytorch.cifar10.LRHelper import LearningRateHelper
from org.campagnelab.dl.pytorch.cifar10.LossHelper import LossHelper
from org.campagnelab.dl.pytorch.cifar10.PerformanceList import PerformanceList
from org.campagnelab.dl.pytorch.cifar10.utils import progress_bar, grad_norm
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
    return torch.sqrt(torch.mean((y - y_hat).pow(2),1))


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
        self.max_regularization_examples = args.num_shaving
        self.max_validation_examples = args.num_validation
        self.max_training_examples = args.num_training
        self.max_examples_per_epoch = args.max_examples_per_epoch if args.max_examples_per_epoch is not None else self.max_regularization_examples
        self.criterion = problem.loss_function()
        self.split_enabled = args.split
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
        self.failed_to_improve = 0

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
        if args.num_training > training_set_length:
            args.num_training = training_set_length
        unsup_set_length = (len(self.problem.reg_loader())) * mini_batch_size
        if args.num_shaving > unsup_set_length:
            args.num_shaving = unsup_set_length
        test_set_length = (len(self.problem.test_loader())) * mini_batch_size
        if args.num_validation > test_set_length:
            args.num_validation = test_set_length

        self.max_regularization_examples = args.num_shaving
        self.max_validation_examples = args.num_validation
        self.max_training_examples = args.num_training
        self.unsuploader = self.problem.reg_loader()
        model_built = False

        def rmse(y, y_hat):
            """Compute root mean squared error"""
            return torch.sqrt(torch.mean((y - y_hat).pow(2)))


        self.agreement_loss = rmse
        if self.use_cuda:
            self.agreement_loss = self.agreement_loss

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
                model_built = True
            else:
                print("Could not load model checkpoint, unable to --resume.")
                model_built = False

        if not model_built:
            print('==> Building model {}'.format(args.model))

            self.net = create_model_function(args.model)

        if self.use_cuda:
            self.net.cuda()
        cudnn.benchmark = True

        self.optimizer_training = torch.optim.SGD(self.net.parameters(), lr=args.lr, momentum=args.momentum,
                                                  weight_decay=args.L2)

        self.scheduler_train = \
            construct_scheduler(self.optimizer_training, 'max', factor=0.9,
                                lr_patience=self.args.lr_patience)

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
            average_unsupervised_loss=0
            if train_split:

                # obtain an unsupervised sample, put it in uinputs autograd Variable:
                # first, read a minibatch from the unsupervised dataset:
                ufeatures, _ = next(unsupiter)

                if self.use_cuda: ufeatures = ufeatures.cuda()
                # then use it to calculate the unsupervised regularization contribution to the loss:

                unsup_train_loss = self.split_loss(ufeatures,outputs)
                if unsup_train_loss is not None:
                    performance_estimators.set_metric(batch_idx, "split_loss", unsup_train_loss.data[0])
                    average_unsupervised_loss=unsup_train_loss


            if train_supervised_model:
                # if self.ureg._which_one_model is not None:
                #    self.ureg.estimate_example_weights(inputs)

                supervised_loss = self.criterion(outputs, targets)
                alpha=self.args.factor
                optimized_loss=supervised_loss*(1-alpha)+alpha * average_unsupervised_loss
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
        self.args.factor*=self.args.increase_decrease
        return performance_estimators

    def test(self, epoch, performance_estimators=(LossHelper("test_loss"), AccuracyHelper("test_"))):
        print('\nTesting, epoch: %d' % epoch)

        self.net.eval()
        for performance_estimator in performance_estimators:
            performance_estimator.init_performance_metrics()
        for batch_idx, (inputs, targets) in enumerate(self.problem.test_loader_range(0, self.args.num_validation)):

            if self.use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            inputs, targets = Variable(inputs, volatile=True), Variable(targets)
            outputs = self.net(inputs)
            loss = self.criterion(outputs, targets)

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

        metrics = [epoch, self.args.checkpoint_key]
        for performance_estimator in performance_estimators:
            metrics = metrics + performance_estimator.estimates_of_metric()

        with open("all-perfs-{}.tsv".format(self.args.checkpoint_key), "a") as perf_file:
            perf_file.write("\t".join(map(_format_nice, metrics)))
            perf_file.write("\n")

        metric = self.get_metric(performance_estimators, "test_accuracy")
        if metric is not None and metric > self.best_acc:
            self.save_checkpoint(epoch, metric)
            self.failed_to_improve = 0
            with open("best-perfs-{}.tsv".format(self.args.checkpoint_key), "a") as perf_file:
                perf_file.write("\t".join(map(_format_nice, metrics)))
                perf_file.write("\n")
        if metric is not None and metric <= self.best_acc:
            self.failed_to_improve += 1
            if self.failed_to_improve > self.args.abort_when_failed_to_improve:
                print("We failed to improve for {} epochs. Stopping here as requested.")
                return True  # request early stopping

        return False

    def get_metric(self, performance_estimators, metric_name):
        for pe in performance_estimators:
            metric = pe.get_metric(metric_name)
            if metric is not None:
                return metric
        return None

    def save_checkpoint(self, epoch, acc):
        # Save checkpoint.

        if acc > self.best_acc:
            print('Saving..')
            state = {
                'net': self.net.module if self.is_parallel else self.net,
                'acc': acc,
                'epoch': epoch,
                'split': self.split_enabled,
            }
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            torch.save(state, './checkpoint/ckpt_{}.t7'.format(self.args.checkpoint_key))
            self.best_acc = acc

    def training_supervised(self):
        """Train the model in a completely supervised manner. Returns the performance obtained
           at the end of the configured training run.
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
                                 train_split=False
                                 )]
            perfs += [(lr_train_helper,)]
            if previous_test_perfs is None or self.epoch_is_test_epoch(epoch):
                perfs += [self.test(epoch)]

            perfs = flatten(perfs)
            if (not header_written):
                header_written = True
                self.log_performance_header(perfs)

            if self.log_performance_metrics(epoch, perfs):
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

            if self.log_performance_metrics(epoch, perfs):
                # early stopping requested.
                return perfs

        return perfs

    def epoch_is_test_epoch(self, epoch):
        epoch_is_one_of_last_ten = epoch > (self.start_epoch + self.args.num_epochs - 10)
        return (epoch % self.args.test_every_n_epochs + 1) == 1 or epoch_is_one_of_last_ten

    def split_loss(self, uinputs,training_outputs):
        pi = math.atan(1) * 4
        angle = uniform(pi , -pi )
        slope = math.tan(angle)
        (image_1, image_2) = self.half_images(uinputs, slope)
        answer_1 = self.net(image_1)
        answer_2 = self.net(image_2)
        _, predicted_classes=torch.max(answer_2.data, 1)
        predicted_classes_var = Variable(predicted_classes, requires_grad=False)
        rmse_loss=self.agreement_loss(training_outputs,answer_2)
        return self.problem.loss_function()(answer_1, predicted_classes_var)+rmse_loss

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
        #print("mask down-------------")
        # fill in the first channel:
        for x in range(0, width):
            for y in range(0, height):
                above_the_line = above_line(x - width / 2, y, slope=slope, b=height / 2.0 )
                mask_up[x, y] = 1 if above_the_line else 0
                mask_down[x, y] = 0 if above_the_line else 1
                #print("." if mask_down[x, y] else " ",end="")
            #print("|")

        #print("-----------mask down")
        mask1 = torch.ByteTensor(uinputs.size()[1:])
        mask2 = torch.ByteTensor(uinputs.size()[1:])

        if self.use_cuda:
            mask_up = mask_up.cuda()
            mask_down = mask_down.cuda()
            mask1 = mask1.cuda()
            mask2 = mask2.cuda()
        for c in range(0, channels):
            mask1[c] = mask_up
            mask2[c] = mask_down
        return uinputs.masked_fill(mask1, 0.), uinputs.masked_fill(mask2, 0.)
