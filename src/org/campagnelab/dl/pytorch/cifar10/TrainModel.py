import os
import torch
from torch.autograd import Variable
from torch.backends import cudnn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data.sampler import RandomSampler, SequentialSampler, Sampler

from org.campagnelab.dl.pytorch.cifar10.AccuracyHelper import AccuracyHelper
from org.campagnelab.dl.pytorch.cifar10.Evaluate2 import TrainingPerf
from org.campagnelab.dl.pytorch.cifar10.LossHelper import LossHelper
from org.campagnelab.dl.pytorch.cifar10.utils import progress_bar
from org.campagnelab.dl.pytorch.ureg.LRSchedules import LearningRateAnnealing
from org.campagnelab.dl.pytorch.ureg.URegularizer import URegularizer


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


class TrimSampler(Sampler):
    """Samples elements sequentially, always in the same order, within the provided index bounds.

    Arguments:
        data_source (Dataset): dataset to sample from
    """

    def __init__(self, data_source, start=0, end=None):
        if end is None:
            end = len(data_source)
        self.start = start
        self.end = end

        self.data_source = data_source

    def __iter__(self):
        return iter(range(self.start, self.end))

    def __len__(self):
        return self.end - self.start


class TrainModel:
    """Train a model."""

    def __init__(self, args, problem, use_cuda):
        """
        Initialize model training with arguments and problem.

        :param args command line arguments.
        :param use_cuda When True, use the GPU.
         """
        self.max_regularization_examples = args.num_shaving
        self.max_validation_examples = args.num_validation
        self.max_training_examples = args.num_training
        self.criterion = problem.loss_function()
        self.ureg_enabled = args.ureg
        self.args = args
        self.problem = problem
        self.best_acc = None
        self.start_epoch = 0
        self.use_cuda = use_cuda
        self.mini_batch_size = problem.mini_batch_size()
        self.net = None
        self.optimizer_training = None
        self.optimizer_reg = None
        self.scheduler_train = None
        self.scheduler_reg = None
        self.unsuploader = self.problem.reg_loader()
        self.trainloader = self.problem.train_loader()
        self.testloader = self.problem.test_loader()
        self.ureg = None
        self.is_parallel = False

    def init_model(self, create_model_function):
        """Resume training if necessary (args.--resume flag is True), or call the
        create_model_function to initialize a new model. This function must be called
        before train.

        The create_model_function takes one argument: the name of the model to be
        created.
        """
        args = self.args

        self.unsuploader = self.problem.reg_loader()
        if args.resume:
            # Load checkpoint.

            print('==> Resuming from checkpoint..')

            mini_batch_size = self.mini_batch_size

            assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
            checkpoint = torch.load('./checkpoint/ckpt_{}.t7'.format(args.checkpoint_key))
            net = checkpoint['net']
            self.best_acc = checkpoint['acc']
            self.start_epoch = checkpoint['epoch']
            ureg_enabled = checkpoint['ureg']

            if ureg_enabled:
                ureg = URegularizer(net, mini_batch_size, args.ureg_num_features,
                                    args.ureg_alpha, args.ureg_learning_rate)

                ureg.set_num_examples(min(len(self.trainloader), args.num_training),
                                      min(len(self.unsuploader), args.num_shaving))
                ureg.enable()
                if not args.drop_ureg_model:
                    ureg.resume(checkpoint['ureg_model'])
        else:
            print('==> Building model {}'.format(args.model))

            self.net = create_model_function(args.model)

        if self.use_cuda:
            self.net.cuda()
        cudnn.benchmark = True

        self.optimizer_training = torch.optim.SGD(self.net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
        self.optimizer_reg = torch.optim.SGD(self.net.parameters(), lr=args.shave_lr, momentum=0.9, weight_decay=5e-4)
        self.ureg = URegularizer(self.net, self.mini_batch_size, num_features=args.ureg_num_features,
                                 alpha=args.ureg_alpha,
                                 learning_rate=args.ureg_learning_rate)
        if args.ureg:
            self.ureg.enable()
            self.ureg.set_num_examples(args.num_training, len(self.unsuploader))
            self.ureg.forget_model(args.ureg_reset_every_n_epoch)
            print(
                "ureg is enabled with alpha={}, reset every {} epochs. ".format(args.ureg_alpha,
                                                                                args.ureg_reset_every_n_epoch))

        else:
            self.ureg.disable()
            self.ureg.set_num_examples(args.num_training, len(self.unsuploader))
            print("ureg is disabled")

        self.scheduler_train = self.construct_scheduler(self.optimizer_training, 'min')
        # use max for regularization lr because the more regularization
        # progresses, the harder it becomes to differentiate training from test activations, we want larger ureg training losses,
        # so we drop the ureg learning rate whenever the metric does not improve:
        self.scheduler_reg = self.construct_scheduler(self.optimizer_reg, 'max')

    def train(self, epoch, unsupiter,
              performance_estimators=(LossHelper("train_loss"), AccuracyHelper("train_"))):
        print('\nTraining, epoch: %d' % epoch)
        self.net.train()

        for performance_estimator in performance_estimators:
            performance_estimator.init_performance_metrics()
        unsupervised_loss_acc = 0
        num_batches = 0
        for batch_idx, (inputs, targets) in enumerate(self.trainloader):
            num_batches += 1
            if self.use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            self.optimizer_training.zero_grad()
            inputs, targets = Variable(inputs), Variable(targets)
            outputs = self.net(inputs)

            supervised_loss = self.criterion(outputs, targets)
            supervised_loss.backward()
            self.optimizer_training.step()
            # the unsupervised regularization part goes here:
            try:
                # first, read a minibatch from the unsupervised dataset:
                ufeatures, ulabels = next(unsupiter)

            except StopIteration:
                unsupiter = iter(self.unsuploader)
                ufeatures, ulabels = next(unsupiter)
            if self.use_cuda: ufeatures = ufeatures.cuda()
            # then use it to calculate the unsupervised regularization contribution to the loss:
            uinputs = Variable(ufeatures)

            unsupervised_loss_acc += self.ureg.train_ureg(inputs, uinputs)

            optimized_loss = supervised_loss

            for performance_estimator in performance_estimators:
                performance_estimator.observe_performance_metric(batch_idx, optimized_loss, outputs, targets)

            progress_bar(batch_idx, len(self.problem.train_loader()),
                         " ".join([performance_estimator.progress_message() for performance_estimator in
                                   performance_estimators]))
            if (batch_idx + 1) * self.mini_batch_size > self.max_training_examples:
                break
        unsupervised_loss = unsupervised_loss_acc / (num_batches)
        self.scheduler_reg.step(unsupervised_loss, epoch)
        print()

        return performance_estimators

    def regularize(self, epoch, performance_estimators=(LossHelper("reg_loss"),)):
        """
        Performs training vs test regularization/shaving phase.
        :param epoch:
        :param performance_estimators: estimators for performance metrics to collect.
        :return:
        """
        print('\nRegularizing, epoch: %d' % epoch)
        self.net.train()

        trainiter = iter(self.trainloader)
        train_examples_used = 0
        for performance_estimator in performance_estimators:
            performance_estimator.init_performance_metrics()

        for shaving_index in range(self.args.shaving_epochs):
            print("Shaving step {}".format(shaving_index))
            # produce a random subset of the unsupervised samples, exactly matching the number of training examples:
            unsupsampler = TrimSampler(RandomSampler(self.unsuploader), 0, self.args.num_training)
            for batch_idx, (inputs, targets) in enumerate(unsupsampler):

                if self.use_cuda:
                    inputs, targets = inputs.cuda(), targets.cuda()

                self.optimizer_reg.zero_grad()
                uinputs, _ = Variable(inputs), Variable(targets)

                # don't use more training examples than allowed (-n) even if we don't use
                # their labels:
                if train_examples_used > self.args.num_training:
                    trainiter = iter(self.trainloader)
                    train_examples_used = 0
                try:
                    # first, read a minibatch from the unsupervised dataset:
                    features, ulabels = next(trainiter)

                except StopIteration:
                    trainiter = iter(self.trainloader)
                    features, _ = next(trainiter)
                train_examples_used += 1

                if self.use_cuda: features = features.cuda()

                # then use it to calculate the unsupervised regularization contribution to the loss:
                inputs = Variable(features)
                regularization_loss = self.ureg.regularization_loss(inputs, uinputs)
                if regularization_loss is not None:

                    regularization_loss.backward()
                    self.optimizer_reg.step()
                    optimized_loss = regularization_loss.data[0]

                else:
                    optimized_loss = 0
                for performance_estimator in performance_estimators:
                    performance_estimator.observe_performance_metric(batch_idx, optimized_loss,
                                                                     inputs, uinputs)

                progress_bar(batch_idx, len(self.unsuploader),
                             " ".join([performance_estimator.progress_message() for performance_estimator in
                                       performance_estimators]))
                if ((batch_idx + 1) * self.mini_batch_size) > self.max_regularization_examples:
                    break
                print()

        return performance_estimators

    def test(self, epoch, performance_estimators=(LossHelper("test_loss"), AccuracyHelper("test_"))):
        global best_acc
        self.net.eval()

        self.ureg.new_epoch(epoch)
        for batch_idx, (inputs, targets) in enumerate(self.testloader):

            if self.use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            inputs, targets = Variable(inputs, volatile=True), Variable(targets)
            outputs = self.net(inputs)
            loss = self.criterion(outputs, targets)

            for performance_estimator in performance_estimators:
                performance_estimator.observe_performance_metric(batch_idx, loss, outputs, targets)

            self.ureg.estimate_accuracy(inputs)
            progress_bar(batch_idx, len(self.problem.train_loader()),
                         " ".join([performance_estimator.progress_message() for performance_estimator in
                                   performance_estimators]))

            if ((batch_idx + 1) * self.mini_batch_size) > self.max_validation_examples:
                break
        print()

        # Apply learning rate schedule:
        test_accuracy = self.get_metric(performance_estimators, "test_accuracy")
        assert test_accuracy is not None, "test_accuracy must be found among estimated performance metrics"
        self.scheduler_train.step(test_accuracy, epoch)
        self.ureg.schedule(test_accuracy, epoch)

        return performance_estimators

    def construct_scheduler(self, optimizer, direction='min'):
        delegate_scheduler = ReduceLROnPlateau(optimizer, direction, factor=0.5,
                                               patience=self.args.lr_patience, verbose=True)

        if self.args.ureg_reset_every_n_epoch is None:
            scheduler = delegate_scheduler
        else:
            scheduler = LearningRateAnnealing(optimizer,
                                              anneal_every_n_epoch=self.args.ureg_reset_every_n_epoch,
                                              delegate=delegate_scheduler)
        return scheduler

    def log_performance_header(self, performance_estimators):
        global best_test_loss

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

        global best_acc

        metric = self.get_metric(performance_estimators, "test_accuracy")
        if metric is not None and metric > best_acc:
            self.save_checkpoint(metric)
            with open("best-perfs-{}.tsv".format(self.args.checkpoint_key), "a") as perf_file:
                perf_file.write("\t".join(map(_format_nice, metrics)))
                perf_file.write("\n")

    def get_metric(self, performance_estimators, metric_name):
        for pe in performance_estimators:
            metric = pe.get_metric(metric_name)
            if metric is not None:
                return metric
        return None

    def save_checkpoint(self, epoch, acc):
        # Save checkpoint.
        global best_acc
        if acc > best_acc:
            print('Saving..')
            state = {
                'net': self.net.module if self.is_parallel else self.net,
                'acc': acc,
                'epoch': epoch,
                'ureg': self.ureg_enabled,
                'ureg_model': self.ureg._which_one_model
            }
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            torch.save(state, './checkpoint/ckpt_{}.t7'.format(args.checkpoint_key))
            best_acc = acc
