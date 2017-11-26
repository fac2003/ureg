import os

import torch
from torch.autograd import Variable
from torch.backends import cudnn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data.sampler import RandomSampler, BatchSampler

from org.campagnelab.dl.pytorch.cifar10.AccuracyHelper import AccuracyHelper
from org.campagnelab.dl.pytorch.cifar10.LRHelper import LearningRateHelper
from org.campagnelab.dl.pytorch.cifar10.LossHelper import LossHelper
from org.campagnelab.dl.pytorch.cifar10.Samplers import TrimSampler
from org.campagnelab.dl.pytorch.cifar10.utils import progress_bar
from org.campagnelab.dl.pytorch.ureg.LRSchedules import LearningRateAnnealing, construct_scheduler
from org.campagnelab.dl.pytorch.ureg.URegularizer import URegularizer

best_acc = 0


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


flatten = lambda l: [item for sublist in l for item in sublist]


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
        self.max_examples_per_epoch = args.max_examples_per_epoch
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
        self.scheduler_ureg = None
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
        mini_batch_size = self.mini_batch_size
        # restrict limits to actual size of datasets:
        training_set_length = (len(self.problem.train_loader()) - 1) * mini_batch_size
        if args.num_training > training_set_length:
            args.num_training = training_set_length
        unsup_set_length = (len(self.problem.reg_loader()) - 1) * mini_batch_size
        if args.num_shaving > unsup_set_length:
            args.num_shaving = unsup_set_length
        test_set_length = (len(self.problem.test_loader()) - 1) * mini_batch_size
        if args.num_validation > test_set_length:
            args.num_validation = test_set_length

        self.max_regularization_examples = args.num_shaving
        self.max_validation_examples = args.num_validation
        self.max_training_examples = args.num_training
        self.unsuploader = self.problem.reg_loader()

        if args.resume:
            # Load checkpoint.

            print('==> Resuming from checkpoint..')

            assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
            checkpoint = torch.load('./checkpoint/ckpt_{}.t7'.format(args.checkpoint_key))
            self.net = checkpoint['net']
            self.best_acc = checkpoint['acc']
            self.start_epoch = checkpoint['epoch']
            ureg_enabled = checkpoint['ureg']

            if ureg_enabled:
                ureg = URegularizer(self.net, mini_batch_size, args.ureg_num_features,
                                    args.ureg_alpha, args.ureg_learning_rate,
                                    reset_every_epochs=args.ureg_reset_every_n_epoch,
                                    do_not_use_scheduler=self.args.constant_learning_rates)

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
                                 learning_rate=args.ureg_learning_rate,
                                 reset_every_epochs=args.ureg_reset_every_n_epoch,
                                 do_not_use_scheduler=self.args.constant_learning_rates)
        if args.ureg:
            self.ureg.enable()
            self.ureg.set_num_examples(min(len(self.trainloader), args.num_training),
                                       min(len(self.unsuploader), args.num_shaving))
            self.ureg.forget_model(args.ureg_reset_every_n_epoch)
            print(
                "ureg is enabled with alpha={}, reset every {} epochs. ".format(args.ureg_alpha,
                                                                                args.ureg_reset_every_n_epoch))

        else:
            self.ureg.disable()
            self.ureg.set_num_examples(min(len(self.trainloader), args.num_training),
                                       min(len(self.unsuploader), args.num_shaving))
            print("ureg is disabled")

        self.scheduler_train = \
            construct_scheduler(self.optimizer_training, 'min',
                                lr_patience=self.args.lr_patience,
                                ureg_reset_every_n_epoch=self.args.ureg_reset_every_n_epoch)

        # use max for regularization lr because the more regularization
        # progresses, the harder it becomes to differentiate training from test activations, we want larger ureg training losses,
        # so we drop the ureg learning rate whenever the metric does not improve:
        self.scheduler_reg = \
            construct_scheduler(self.optimizer_reg,
                                'max', extra_patience=5,
                                lr_patience=self.args.lr_patience,
                                ureg_reset_every_n_epoch=self.args.ureg_reset_every_n_epoch)
        self.num_shaving_epochs = self.args.shaving_epochs
        if self.args.num_training > self.args.num_shaving:
            self.num_shaving_epochs = round((self.args.num_training + 1) / self.args.num_shaving)
            print("--shaving-epochs overridden to " + str(self.num_shaving_epochs))
        else:
            print("shaving-epochs set  to " + str(self.num_shaving_epochs))

    def train(self, epoch,
              performance_estimators=(LossHelper("train_loss"), AccuracyHelper("train_")),
              train_supervised_model=True,
              train_ureg=True):
        print('\nTraining, epoch: %d' % epoch)
        self.net.train()

        for performance_estimator in performance_estimators:
            performance_estimator.init_performance_metrics()
        unsupervised_loss_acc = 0
        num_batches = 0
        train_loader_subset = self.problem.train_loader_subset(0, self.args.num_training)
        unsuploader_shuffled = self.problem.reg_loader_subset(0, self.args.num_shaving)
        unsupiter = iter(unsuploader_shuffled)
        for batch_idx, (inputs, targets) in enumerate(train_loader_subset):
            num_batches += 1

            if train_supervised_model:
                if self.use_cuda:
                    inputs, targets = inputs.cuda(), targets.cuda()
                self.optimizer_training.zero_grad()
                inputs, targets = Variable(inputs), Variable(targets)
                outputs = self.net(inputs)

                # if self.ureg._which_one_model is not None:
                #    self.ureg.estimate_example_weights(inputs)

                supervised_loss = self.criterion(outputs, targets)
                supervised_loss.backward()
                self.optimizer_training.step()

            if train_ureg:
                # the unsupervised regularization part goes here:
                if (batch_idx + 1) * self.mini_batch_size > self.max_regularization_examples:
                    break
                try:
                    # first, read a minibatch from the unsupervised dataset:
                    ufeatures, ulabels = next(unsupiter)

                except StopIteration:
                    unsupiter = iter(unsuploader_shuffled)
                    ufeatures, ulabels = next(unsupiter)
                if self.use_cuda: ufeatures = ufeatures.cuda()
                # then use it to calculate the unsupervised regularization contribution to the loss:
                uinputs = Variable(ufeatures)
                ureg_loss = self.ureg.train_ureg(inputs, uinputs)
                if (ureg_loss is not None):
                    unsupervised_loss_acc += ureg_loss.data[0]

            optimized_loss = supervised_loss

            for performance_estimator in performance_estimators:
                performance_estimator.observe_performance_metric(batch_idx, optimized_loss.data[0], outputs, targets)

            progress_bar(batch_idx, len(train_loader_subset),
                         " ".join([performance_estimator.progress_message() for performance_estimator in
                                   performance_estimators]))
            if (batch_idx + 1) * self.mini_batch_size > self.max_training_examples:
                break
        unsupervised_loss = unsupervised_loss_acc / (num_batches)
        if not self.args.constant_learning_rates:
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
        use_max_shaving_records = self.args.num_shaving
        # make sure we process the entire training set, but limit how many regularization_examples we scan (randomly
        # from the entire set):
        if self.max_examples_per_epoch > self.args.num_training:
            max_loop_index = min(self.max_examples_per_epoch, self.max_regularization_examples)
        else:
            max_loop_index = self.args.num_training

        for performance_estimator in performance_estimators:
            performance_estimator.init_performance_metrics()

        for shaving_index in range(self.num_shaving_epochs):
            print("Shaving step {}".format(shaving_index))
            # produce a random subset of the unsupervised samples, exactly matching the number of training examples:
            unsupsampler = self.problem.reg_loader_subset(0, use_max_shaving_records)
            for performance_estimator in performance_estimators:
                performance_estimator.init_performance_metrics()

            for batch_idx, (inputs, targets) in enumerate(unsupsampler):

                if self.use_cuda:
                    inputs = inputs.cuda()

                self.optimizer_reg.zero_grad()
                uinputs = Variable(inputs)

                # don't use more training examples than allowed (-n) even if we don't use
                # their labels:
                if train_examples_used > self.args.num_training:
                    trainiter = iter(self.trainloader)
                    train_examples_used = 0
                try:
                    # first, read a minibatch from the unsupervised dataset:
                    features, _ = next(trainiter)

                except StopIteration:
                    trainiter = iter(self.trainloader)
                    features, _ = next(trainiter)
                train_examples_used += 1

                if self.use_cuda: features = features.cuda()

                # then use it to calculate the unsupervised regularization contribution to the loss:
                inputs = Variable(features)
                regularization_loss = self.ureg.regularization_loss(inputs, uinputs)
                if regularization_loss is not None:
                    optimized_loss = regularization_loss.data[0]
                    regularization_loss.backward()
                    self.optimizer_reg.step()
                else:
                    print("Found None in regularize")
                    optimized_loss = 0

                for performance_estimator in performance_estimators:
                    performance_estimator.observe_performance_metric(batch_idx, optimized_loss,
                                                                     inputs, uinputs)

                progress_bar(batch_idx, max_loop_index,
                             " ".join([performance_estimator.progress_message() for performance_estimator in
                                       performance_estimators]))
                if ((batch_idx + 1) * self.mini_batch_size) > max_loop_index:
                    break

            print()

        return performance_estimators

    def test(self, epoch, performance_estimators=(LossHelper("test_loss"), AccuracyHelper("test_"))):
        print('\nTesting, epoch: %d' % epoch)
        global best_acc
        self.net.eval()
        for performance_estimator in performance_estimators:
            performance_estimator.init_performance_metrics()
        for batch_idx, (inputs, targets) in enumerate(self.testloader):

            if self.use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            inputs, targets = Variable(inputs, volatile=True), Variable(targets)
            outputs = self.net(inputs)
            loss = self.criterion(outputs, targets)

            for performance_estimator in performance_estimators:
                performance_estimator.observe_performance_metric(batch_idx, loss.data[0], outputs, targets)

            self.ureg.estimate_accuracy(inputs)
            progress_bar(batch_idx, len(self.testloader),
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
            metrics = metrics + [performance_estimator.metric_names()]

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
            self.save_checkpoint(epoch, metric)
            self.failed_to_improve=0
            with open("best-perfs-{}.tsv".format(self.args.checkpoint_key), "a") as perf_file:
                perf_file.write("\t".join(map(_format_nice, metrics)))
                perf_file.write("\n")
        if metric is not None and metric <= best_acc:
            self.failed_to_improve+=1
            if self.failed_to_improve>self.args.abort_when_failed_to_improve:
                print("We failed to improve for {} epochs. Stopping here as requested.")
                exit(0)

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
            torch.save(state, './checkpoint/ckpt_{}.t7'.format(self.args.checkpoint_key))
            best_acc = acc

    def training_combined(self):
        header_written = False
        if self.ureg_enabled:
            # tell ureg to use a scheduler:
            self.ureg.install_scheduler()
        lr_train_helper = LearningRateHelper(scheduler=self.scheduler_train, learning_rate_name="train_lr")
        lr_reg_helper = LearningRateHelper(scheduler=self.scheduler_reg, learning_rate_name="reg_lr")
        lr_ureg_helper = None  # will be installed on the fly when the ureg model is built, below.

        for epoch in range(self.start_epoch, self.start_epoch + self.args.num_epochs):
            self.ureg.new_epoch(epoch)
            perfs = []

            perfs += [self.train(epoch)]

            if (self.args.ureg):
                ureg_loss = self.ureg.get_ureg_loss()
                # adjust ureg model learning rate as needed:
                self.ureg.schedule(ureg_loss, epoch)
                # log the ureg loss:
                perf_ureg_loss = LossHelper("ureg_loss")
                perf_ureg_loss.observe_performance_metric(1, ureg_loss, None, None)
                perfs += [(perf_ureg_loss,)]
                perfs += [self.regularize(epoch)]

            perfs += [self.test(epoch)]
            if self.ureg_enabled:
                if lr_ureg_helper is None:
                    lr_ureg_helper = LearningRateHelper(scheduler=self.ureg._scheduler, learning_rate_name="ureg_lr")

                perfs += [(lr_train_helper, lr_reg_helper, lr_ureg_helper)]
            else:
                perfs += [(lr_train_helper, lr_reg_helper)]
            perfs = flatten(perfs)
            if (not header_written):
                header_written = True
                self.log_performance_header(perfs)

            self.log_performance_metrics(epoch, perfs)

    def training_interleaved(self, epsilon=1E-6):
        header_written = False
        self.ureg.install_scheduler()
        for epoch in range(self.start_epoch, self.start_epoch + self.args.num_epochs):
            perfs = []

            train_perfs = self.train(epoch, train_ureg=False, train_supervised_model=True, )
            perfs += [train_perfs]
            if self.args.ureg:
                self.ureg.new_epoch(epoch)
                train_loader_subset = self.problem.train_loader_subset(0, self.args.num_training)
                unsuploader_shuffled = self.problem.reg_loader_subset(0, self.args.num_shaving)

                print("Training ureg to convergence.")
                ureg_training_perf = self.ureg.train_ureg_to_convergence(train_loader_subset, unsuploader_shuffled,
                                                                         epsilon=epsilon, max_epochs=50,
                                                                         max_examples=self.args.max_examples_per_epoch)
                perfs += [ureg_training_perf]

            if self.args.ureg:
                reg_perfs = self.regularize(epoch)
                perfs += [reg_perfs]

            perfs += [self.test(epoch)]
            perfs = flatten(perfs)
            if (not header_written):
                header_written = True
                self.log_performance_header(perfs)

            self.log_performance_metrics(epoch, perfs)
