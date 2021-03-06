import os

import torch
from torch.autograd import Variable
from torch.backends import cudnn

from org.campagnelab.dl.pytorch.images.AccuracyHelper import AccuracyHelper
from org.campagnelab.dl.pytorch.images.FloatHelper import FloatHelper
from org.campagnelab.dl.pytorch.images.LRHelper import LearningRateHelper
from org.campagnelab.dl.pytorch.images.LossHelper import LossHelper
from org.campagnelab.dl.pytorch.images.PerformanceList import PerformanceList
from org.campagnelab.dl.pytorch.images.utils import progress_bar, grad_norm, scale_gradient
from org.campagnelab.dl.pytorch.ureg.LRSchedules import construct_scheduler
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


flatten = lambda l: [item for sublist in l for item in sublist]


def print_params(epoch, net):
    params = []
    for param in net.parameters():
        params += [p for p in param.view(-1).data]
    print("epoch=" + str(epoch) + " " + " ".join(map(str, params)))


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
        self.max_examples_per_epoch = args.max_examples_per_epoch if args.max_examples_per_epoch is not None else self.max_regularization_examples
        self.criterion = problem.loss_function()
        self.ureg_enabled = args.ureg
        self.args = args
        self.problem = problem
        self.best_acc = 0
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

        # setup the function which decides which layer outputs to use for ureg training:
        if args.threshold_activation_size is not None:
            include_output_function = lambda layer_index, activations: True \
                if activations.size()[1] <= args.threshold_activation_size else False
        elif args.include_layer_indices is not None:
            layer_indices = []
            for index in map(int, args.include_layer_indices.split(",")):
                layer_indices.append(index)
            include_output_function = lambda layer_index, activations: True \
                if layer_index in layer_indices else False
        else:
            include_output_function = None

        if args.threshold_activation_size is not None and args.include_layer_indices is not None:
            print("--include-layer-indices and --threshold-activation-size are mutually exclusive, please pick one.")
            exit(1)

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
                ureg_enabled = checkpoint['ureg']

                if ureg_enabled:
                    ureg = URegularizer(self.net, mini_batch_size, args.ureg_num_features,
                                        args.ureg_alpha, args.ureg_learning_rate,
                                        reset_every_epochs=args.ureg_reset_every_n_epoch,
                                        do_not_use_scheduler=self.args.constant_learning_rates,
                                        include_output_function=include_output_function)

                    ureg.set_num_examples(min(len(self.trainloader), args.num_training),
                                          min(len(self.unsuploader), args.num_shaving))
                    ureg.enable()
                    if not args.drop_ureg_model:
                        ureg.resume(checkpoint['ureg_model'])
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
        self.optimizer_reg = torch.optim.SGD(self.net.parameters(), lr=args.shave_lr, momentum=args.momentum,
                                             weight_decay=args.L2)
        self.ureg = URegularizer(self.net, self.mini_batch_size, num_features=args.ureg_num_features,
                                 alpha=args.ureg_alpha,
                                 learning_rate=args.ureg_learning_rate,
                                 reset_every_epochs=args.ureg_reset_every_n_epoch,
                                 do_not_use_scheduler=self.args.constant_learning_rates,
                                 include_output_function=include_output_function)
        if args.ureg:
            self.ureg.enable()
            self.ureg.set_num_examples(min(len(self.trainloader) * mini_batch_size, args.num_training),
                                       min(len(self.unsuploader) * mini_batch_size, args.num_shaving))
            print("weights: {} ".format(self.ureg.loss_weights(None, None)))
            self.ureg.forget_model(args.ureg_reset_every_n_epoch)
            print(
                "ureg is enabled with alpha={}, reset every {} epochs. ".format(args.ureg_alpha,
                                                                                args.ureg_reset_every_n_epoch))

        else:
            self.ureg.disable()
            self.ureg.set_num_examples(min(len(self.trainloader) * mini_batch_size, args.num_training),
                                       min(len(self.unsuploader) * mini_batch_size, args.num_shaving))
            print("ureg is disabled")

        self.scheduler_train = \
            construct_scheduler(self.optimizer_training, 'max', factor=0.9,
                                lr_patience=self.args.lr_patience,
                                ureg_reset_every_n_epoch=self.args.ureg_reset_every_n_epoch)
        # the regularizer aims to increase uncertainty between training and unsup set. Larger accuracies  are closer to
        # success, optimize for max of the accuracy.
        self.scheduler_reg = \
            construct_scheduler(self.optimizer_reg,
                                'max', extra_patience=0 if args.mode == "one_pass" else 5,
                                lr_patience=self.args.lr_patience, factor=0.9,
                                ureg_reset_every_n_epoch=self.args.ureg_reset_every_n_epoch)
        self.num_shaving_epochs = self.args.shaving_epochs
        if self.args.num_training > self.args.num_shaving:
            self.num_shaving_epochs = round((self.args.num_training + 1) / self.args.num_shaving)
            print("--shaving-epochs overridden to " + str(self.num_shaving_epochs))
        else:
            print("shaving-epochs set  to " + str(self.num_shaving_epochs))

    def train(self, epoch,
              performance_estimators=None,
              train_supervised_model=True,
              train_ureg=True,
              regularize=False,
              previous_training_loss=1.0,
              previous_ureg_loss=1.0):

        if performance_estimators is None:
            performance_estimators = PerformanceList()
            performance_estimators += [LossHelper("train_loss"), AccuracyHelper("train_")]

            if regularize:
                performance_estimators += [LossHelper("reg_loss")]
            if train_ureg:
                performance_estimators += [LossHelper("ureg_loss"), FloatHelper("ureg_accuracy")]
        performance_estimators += [FloatHelper("train_grad_norm")]
        if regularize:
            performance_estimators += [FloatHelper("reg_grad_norm")]
        print('\nTraining, epoch: %d' % epoch)
        self.net.train()
        supervised_grad_norm = 1.
        for performance_estimator in performance_estimators:
            performance_estimator.init_performance_metrics()

        unsupervised_loss_acc = 0
        num_batches = 0
        train_loader_subset = self.problem.train_loader_subset_range(0, self.args.num_training)
        unsuploader_shuffled = self.problem.reg_loader_subset_range(0, self.args.num_shaving)
        unsupiter = iter(unsuploader_shuffled)
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

            if train_ureg or regularize:
                # obtain an unsupervised sample, put it in uinputs autograd Variable:

                try:
                    # first, read a minibatch from the unsupervised dataset:
                    ufeatures, ulabels = next(unsupiter)

                except StopIteration:
                    unsupiter = iter(unsuploader_shuffled)
                    ufeatures, ulabels = next(unsupiter)
                if self.use_cuda: ufeatures = ufeatures.cuda()
                # then use it to calculate the unsupervised regularization contribution to the loss:
                uinputs = Variable(ufeatures)

            if train_ureg:

                ureg_loss = self.ureg.train_ureg(inputs, uinputs)
                if (ureg_loss is not None):
                    performance_estimators.set_metric(batch_idx, "ureg_loss", ureg_loss.data[0])
                    performance_estimators.set_metric(batch_idx, "ureg_accuracy", self.ureg.ureg_accuracy())

                    # adjust ureg model learning rate as needed:
                    self.ureg.schedule(ureg_loss.data[0], epoch)

            if train_supervised_model:
                # if self.ureg._which_one_model is not None:
                #    self.ureg.estimate_example_weights(inputs)

                supervised_loss = self.criterion(outputs, targets)
                supervised_loss.backward()
                self.optimizer_training.step()
                supervised_grad_norm = grad_norm(self.net.parameters())
                performance_estimators.set_metric(batch_idx, "train_grad_norm", supervised_grad_norm)

                performance_estimators.set_metric_with_outputs(batch_idx, "train_loss", supervised_loss.data[0],
                                                               outputs, targets)
                performance_estimators.set_metric_with_outputs(batch_idx, "train_accuracy", supervised_loss.data[0],
                                                               outputs, targets)

            if regularize:
                # then use it to calculate the unsupervised regularization contribution to the loss:

                self.optimizer_reg.zero_grad()
                regularization_loss = self.estimate_regularization_loss(inputs, uinputs, 1., 1.)
                if regularization_loss is not None:
                    # print_params(epoch, self.net)
                    alpha = self.args.ureg_alpha

                    regularization_loss*=alpha
                    reg_loss_float = regularization_loss.data[0]
                    self.net.train()
                    regularization_loss.backward()
                    self.optimizer_reg.step()
                    reg_grad_norm = grad_norm(self.net.parameters())
                    performance_estimators.set_metric(batch_idx, "reg_grad_norm", reg_grad_norm)
                    # print_params(epoch, self.net)
                    # print("\n")
                else:
                    reg_loss_float = 0

                performance_estimators.set_metric(batch_idx, "reg_loss", reg_loss_float)
                performance_estimators.set_metric(batch_idx, "ureg_alpha", self.ureg._alpha)

            progress_bar(batch_idx * self.mini_batch_size,
                         min(self.max_regularization_examples, self.max_training_examples),
                         " ".join([performance_estimator.progress_message() for performance_estimator in
                                   performance_estimators]))

            if (batch_idx + 1) * self.mini_batch_size > self.max_regularization_examples:
                break

            if (batch_idx + 1) * self.mini_batch_size > self.max_training_examples:
                break

            print("\n")

        return performance_estimators

    def train_linear_combination(self, epoch,
                                 performance_estimators=None,
                                 train_supervised_model=True,
                                 train_ureg=True,
                                 ):

        if performance_estimators is None:
            performance_estimators = PerformanceList()
            performance_estimators += [LossHelper("train_loss"), AccuracyHelper("train_")]
            performance_estimators += [LossHelper("reg_loss")]
            if train_ureg:
                performance_estimators += [LossHelper("ureg_loss"), FloatHelper("ureg_accuracy")]
            performance_estimators += [FloatHelper("train_grad_norm")]
        print('\nTraining, epoch: %d' % epoch)
        self.net.train()

        for performance_estimator in performance_estimators:
            performance_estimator.init_performance_metrics()

        num_batches = 0
        train_loader_subset = self.problem.train_loader_subset_range(0, self.args.num_training)
        unsuploader_shuffled = self.problem.reg_loader_subset_range(0, self.args.num_shaving)
        unsupiter = iter(unsuploader_shuffled)
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

            if train_ureg:
                # obtain an unsupervised sample, put it in uinputs autograd Variable:

                try:
                    # first, read a minibatch from the unsupervised dataset:
                    ufeatures, ulabels = next(unsupiter)

                except StopIteration:
                    unsupiter = iter(unsuploader_shuffled)
                    ufeatures, ulabels = next(unsupiter)
                if self.use_cuda: ufeatures = ufeatures.cuda()
                # then use it to calculate the unsupervised regularization contribution to the loss:
                uinputs = Variable(ufeatures)

                performance_estimators.set_metric(batch_idx, "ureg_alpha", self.ureg._alpha)

            if train_ureg:

                ureg_loss = self.ureg.train_ureg(inputs, uinputs)
                if (ureg_loss is not None):
                    performance_estimators.set_metric(batch_idx, "ureg_loss", ureg_loss.data[0])
                    performance_estimators.set_metric(batch_idx, "ureg_accuracy", self.ureg.ureg_accuracy())

                    # adjust ureg model learning rate as needed:
                    self.ureg.schedule(ureg_loss.data[0], epoch)

            if train_supervised_model:
                alpha = self.args.ureg_alpha
                # if self.ureg._which_one_model is not None:
                #    self.ureg.estimate_example_weights(inputs)

                supervised_loss = self.criterion(outputs, targets)

                regularization_loss = self.estimate_regularization_loss(inputs, uinputs, 1., 1.)
                if regularization_loss is not None:
                    optimized_loss = supervised_loss * (1. - alpha) + regularization_loss * alpha
                    performance_estimators.set_metric(batch_idx, "reg_loss", regularization_loss.data[0])

                else:
                    optimized_loss = supervised_loss

                optimized_loss.backward()
                supervised_grad_norm = grad_norm(self.net.parameters())
                performance_estimators.set_metric(batch_idx, "train_grad_norm", supervised_grad_norm)
                self.optimizer_training.step()

                performance_estimators.set_metric_with_outputs(batch_idx, "train_loss", optimized_loss.data[0],
                                                               outputs, targets)
                performance_estimators.set_metric_with_outputs(batch_idx, "train_accuracy", optimized_loss.data[0],
                                                               outputs, targets)

            progress_bar(batch_idx * self.mini_batch_size,
                         min(self.max_regularization_examples, self.max_training_examples),
                         " ".join([performance_estimator.progress_message() for performance_estimator in
                                   performance_estimators]))

            if (batch_idx + 1) * self.mini_batch_size > self.max_regularization_examples:
                break

            if (batch_idx + 1) * self.mini_batch_size > self.max_training_examples:
                break

            print("\n")

        return performance_estimators

    def regularize(self, epoch, performance_estimators=None,
                   previous_ureg_loss=1.0, previous_training_loss=1.0):
        """
        Performs training vs test regularization/shaving phase.
        :param epoch:
        :param performance_estimators: estimators for performance metrics to collect.
        :return:
        """
        print('\nRegularizing, epoch: %d' % epoch)
        self.net.train()
        if performance_estimators is None:
            performance_estimators=PerformanceList()
            performance_estimators.append(FloatHelper("reg_grad_norm"))
            performance_estimators.append(LossHelper("reg_loss"))
            performance_estimators.append(FloatHelper("ureg_alpha"))

        trainiter = iter(self.trainloader)
        train_examples_used = 0
        use_max_shaving_records = self.args.num_shaving
        # make sure we process the entire training set, but limit how many regularization_examples we scan (randomly
        # from the entire set):
        if self.max_examples_per_epoch > self.args.num_training:
            max_loop_index = min(self.max_examples_per_epoch, self.max_regularization_examples)
        else:
            max_loop_index = self.args.num_training

        performance_estimators.init_performance_metrics()
        unsuper_records_to_be_seen = min(max_loop_index, use_max_shaving_records)
        # max_loop_index is the number of times training examples are seen,
        # use_max_shaving_records is the number of times unsupervised examples are seen,
        # estimate weights:
        a = unsuper_records_to_be_seen / max_loop_index
        b = 1
        weight_s = a / (a + b)
        weight_u = 1 / (a + b)
        print("weight_s={} weight_u={} unsuper_records_to_be_seen={} max_loop_index={}".format(
            weight_s, weight_u, unsuper_records_to_be_seen, max_loop_index))

        for shaving_index in range(self.num_shaving_epochs):
            print("Shaving step {}".format(shaving_index))
            # produce a random subset of the unsupervised samples, exactly matching the number of training examples:
            unsupsampler = self.problem.reg_loader_subset_range(0, use_max_shaving_records)

            performance_estimators.init_performance_metrics()
            performance_estimators.set_metric(1, "ureg_alpha",self.ureg._alpha)

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
                regularization_loss = self.estimate_regularization_loss(inputs, uinputs,
                                                                        weight_s=weight_s,
                                                                        weight_u=weight_u)
                if regularization_loss is not None:
                    regularization_loss = regularization_loss * self.args.ureg_alpha
                    reg_grad_norm = grad_norm(self.net.parameters())
                    performance_estimators.set_metric(batch_idx, "reg_grad_norm", reg_grad_norm)
                    optimized_loss = regularization_loss.data[0]
                    performance_estimators.set_metric(batch_idx, "reg_loss", optimized_loss)

                    regularization_loss.backward()
                    self.optimizer_reg.step()
                else:
                    print("Found None in regularize")
                    optimized_loss = 0

                performance_estimators[0].observe_performance_metric(batch_idx, optimized_loss,
                                                                     inputs, uinputs)

                # keep training the ureg model while regularizing, this is needed to keep ureg relevant to the
                # regularized weights:
                self.ureg.train_ureg(inputs, uinputs)

                progress_bar(batch_idx * self.mini_batch_size, max_loop_index,
                             " ".join([performance_estimator.progress_message() for performance_estimator in
                                       performance_estimators]))
                if ((batch_idx + 1) * self.mini_batch_size) > max_loop_index:
                    break

            print()

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

            self.ureg.estimate_accuracy(inputs)
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
            self.scheduler_reg.step(test_accuracy, epoch)

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
                'ureg': self.ureg_enabled,
                'ureg_model': self.ureg._which_one_model
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
        if self.ureg_enabled and not self.args.constant_ureg_learning_rate:
            # tell ureg to use a scheduler:
            self.ureg.install_scheduler()
        lr_train_helper = LearningRateHelper(scheduler=self.scheduler_train, learning_rate_name="train_lr")
        previous_test_perfs = None

        for epoch in range(self.start_epoch, self.start_epoch + self.args.num_epochs):
            self.ureg.new_epoch(epoch)
            perfs = []

            perfs += [self.train(epoch,
                                 train_supervised_model=True,
                                 train_ureg=False,
                                 regularize=False
                                 )]

            if previous_test_perfs is None or self.epoch_is_test_epoch(epoch):
                previous_test_perfs = self.test(epoch)

            perfs += [previous_test_perfs]

            perfs = flatten(perfs)
            if (not header_written):
                header_written = True
                self.log_performance_header(perfs)

            if self.log_performance_metrics(epoch, perfs):
                # early stopping requested.
                return perfs

        return perfs

    def training_one_pass(self):
        """Train the model with a single pass through the training set. Returns the performance obtained
           at the end of the configured training run.
        :return list of performance estimators that observed performance on the last epoch run.
        """
        header_written = False
        if self.ureg_enabled and not self.args.constant_ureg_learning_rate:
            # tell ureg to use a scheduler:
            self.ureg.install_scheduler()
        lr_train_helper = LearningRateHelper(scheduler=self.scheduler_train, learning_rate_name="train_lr")
        lr_reg_helper = LearningRateHelper(scheduler=self.scheduler_reg, learning_rate_name="reg_lr")
        lr_ureg_helper = None  # will be installed on the fly when the ureg model is built, below.
        previous_test_perfs = None
        previous_ureg_loss = 1.0
        previous_training_loss = 1.0
        if self.args.resume and self.ureg_enabled:
            print("Training ureg to convergence on --resume.")
            # train ureg to convergence on resume:
            train_dataset = self.problem.train_set()
            unsup_dataset = self.problem.unsup_set()
            self.ureg.train_ureg_to_convergence(self.problem, train_dataset, unsup_dataset,
                                                epsilon=self.args.ureg_epsilon, max_epochs=10,
                                                max_examples=self.args.max_examples_per_epoch)
        for epoch in range(self.start_epoch, self.start_epoch + self.args.num_epochs):
            self.ureg.new_epoch(epoch)
            perfs = []

            perfs += [self.train(epoch,
                                 train_supervised_model=True,
                                 train_ureg=True,
                                 regularize=True)]

            if previous_test_perfs is None or self.epoch_is_test_epoch(epoch):
                previous_test_perfs = self.test(epoch)

            perfs += [previous_test_perfs]
            lr_ureg_helper = self.install_ureg_learning_rate_helper(lr_ureg_helper)
            if lr_ureg_helper is not None:
                perfs += [(lr_train_helper, lr_reg_helper, lr_ureg_helper)]
            else:
                perfs += [(lr_train_helper, lr_reg_helper)]
            perfs = flatten(perfs)
            if (not header_written):
                header_written = True
                self.log_performance_header(perfs)

            if self.log_performance_metrics(epoch, perfs):
                # early stopping requested.
                return perfs

            self.grow_unsupervised_examples_per_epoch()
            previous_training_loss = self.get_metric(performance_estimators=perfs, metric_name="train_loss")
            previous_ureg_loss = self.get_metric(performance_estimators=perfs, metric_name="ureg_loss")
        return perfs

    def training_two_passes(self):
        """Train the model with the combined approach. Returns the performance obtained
        at the end of the configured training run.
        :return list of performance estimators that observed performance on the last epoch run.
        """
        header_written = False
        if self.ureg_enabled and not self.args.constant_ureg_learning_rate:
            # tell ureg to use a scheduler:
            self.ureg.install_scheduler()
        lr_train_helper = LearningRateHelper(scheduler=self.scheduler_train, learning_rate_name="train_lr")
        lr_reg_helper = LearningRateHelper(scheduler=self.scheduler_reg, learning_rate_name="reg_lr")
        lr_ureg_helper = None  # will be installed on the fly when the ureg model is built, below.
        previous_test_perfs = None
        previous_ureg_loss = 1.0
        previous_training_loss = 1.0

        for epoch in range(self.start_epoch, self.start_epoch + self.args.num_epochs):
            self.ureg.new_epoch(epoch)
            perfs = []

            perfs += [self.train(epoch,
                                 previous_ureg_loss=previous_ureg_loss,
                                 previous_training_loss=previous_training_loss)]

            if (self.args.ureg):
                perfs += [self.regularize(epoch, previous_ureg_loss=previous_ureg_loss,
                                          previous_training_loss=previous_training_loss)]

            if previous_test_perfs is None or self.epoch_is_test_epoch(epoch):
                previous_test_perfs = self.test(epoch)

            perfs += [previous_test_perfs]
            lr_ureg_helper = self.install_ureg_learning_rate_helper(lr_ureg_helper)
            if lr_ureg_helper is not None:
                perfs += [(lr_train_helper, lr_reg_helper, lr_ureg_helper)]
            else:
                perfs += [(lr_train_helper, lr_reg_helper)]
            perfs = flatten(perfs)
            if (not header_written):
                header_written = True
                self.log_performance_header(perfs)

            if self.log_performance_metrics(epoch, perfs):
                # early stopping requested.
                return perfs

            self.grow_unsupervised_examples_per_epoch()

        previous_training_loss = self.get_metric(performance_estimators=perfs, metric_name="train_loss")
        previous_ureg_loss = self.get_metric(performance_estimators=perfs, metric_name="ureg_loss")

        return perfs

    def install_ureg_learning_rate_helper(self, lr_ureg_helper):
        if self.ureg_enabled:
            if lr_ureg_helper is None:
                lr_ureg_helper = LearningRateHelper(scheduler=self.ureg._scheduler, learning_rate_name="ureg_lr",
                                                    initial_learning_rate=self.args.ureg_learning_rate)
        return lr_ureg_helper

    def training_three_passes(self, epsilon=1E-6):
        header_written = False
        if self.ureg_enabled and not self.args.constant_ureg_learning_rate:
            self.ureg.install_scheduler()
        lr_train_helper = LearningRateHelper(scheduler=self.scheduler_train, learning_rate_name="train_lr")
        lr_reg_helper = LearningRateHelper(scheduler=self.scheduler_reg, learning_rate_name="reg_lr")
        lr_ureg_helper = None

        to_reset_ureg_model = 0
        previous_ureg_loss = 1.0
        previous_training_loss = 1.0

        perfs = []
        for epoch in range(self.start_epoch, self.start_epoch + self.args.num_epochs):
            perfs = []

            train_perfs = self.train(epoch, train_ureg=False, train_supervised_model=True,
                                     previous_ureg_loss=previous_ureg_loss,
                                     previous_training_loss=previous_training_loss)
            perfs += [train_perfs]
            if self.args.ureg:
                self.ureg.new_epoch(epoch)
                train_dataset = self.problem.train_set()
                unsup_dataset = self.problem.unsup_set()

                print("Training ureg to convergence.")
                ureg_training_perf = self.ureg.train_ureg_to_convergence(self.problem, train_dataset, unsup_dataset,
                                                                         epsilon=epsilon, max_epochs=epoch + 1,
                                                                         max_examples=self.args.max_examples_per_epoch)
                perfs += [ureg_training_perf]

            if self.args.ureg:
                reg_perfs = self.regularize(epoch, previous_ureg_loss=previous_ureg_loss,
                                            previous_training_loss=previous_training_loss)
                perfs += [reg_perfs]
                lr_ureg_helper = self.install_ureg_learning_rate_helper(lr_ureg_helper)

            perfs += [self.test(epoch)]

            perfs += [(lr_train_helper, lr_reg_helper, lr_ureg_helper)]

            perfs = flatten(perfs)
            if (not header_written):
                header_written = True
                self.log_performance_header(perfs)

            if self.log_performance_metrics(epoch, perfs):
                # early stopping requested.
                return perfs

            self.grow_unsupervised_examples_per_epoch()
            to_reset_ureg_model += 1
            if self.args.constant_learning_rates and self.args.ureg_reset_every_n_epoch is not None and \
                            to_reset_ureg_model > self.args.ureg_reset_every_n_epoch:
                # when learning rates are constant and ureg_reset_every_n_epoch is specified,
                # reset the ureg model periodically:
                self.ureg.reset_model()
                to_reset_ureg_model = 0
            previous_training_loss = self.get_metric(performance_estimators=perfs, metric_name="train_loss")
            previous_ureg_loss = self.get_metric(performance_estimators=perfs, metric_name="ureg_loss")

        return perfs

    def grow_unsupervised_examples_per_epoch(self):
        if self.args.grow_unsupervised_each_epoch is not None:
            self.args.max_examples_per_epoch += self.args.grow_unsupervised_each_epoch
            if self.args.num_shaving < len(self.problem.unsup_set()):
                self.args.num_shaving += self.args.grow_unsupervised_each_epoch
            self.max_examples_per_epoch += self.args.grow_unsupervised_each_epoch
            print("Increase unsupervised: --max-examples-per-epoch to {} and --num-shaving to {}."
                  .format(self.max_examples_per_epoch,
                          self.args.num_shaving))

    def epoch_is_test_epoch(self, epoch):
        epoch_is_one_of_last_ten = epoch > (self.start_epoch + self.args.num_epochs - 10)
        return (epoch % self.args.test_every_n_epochs + 1) == 1 or epoch_is_one_of_last_ten

    def estimate_regularization_loss(self, inputs, uinputs, weight_s, weight_u):
        if self.args.optimize == "uncertainty":
            return self.ureg.regularization_loss(inputs, uinputs,
                                                 weight_s=weight_s,
                                                 weight_u=weight_u)
        if self.args.optimize == "similarity":
            return self.ureg.regularization_loss_unsup_similarity(inputs)
        assert False, 'self.args.optimize must be "uncertainty" or "similarity"'
