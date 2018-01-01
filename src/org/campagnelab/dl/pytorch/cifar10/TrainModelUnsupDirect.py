import os
from random import randint

import numpy
import sys
import torch
from torch.autograd import Variable
from torch.backends import cudnn
from torch.nn import MultiLabelSoftMarginLoss, BCELoss
from torchnet.dataset import ConcatDataset
from torchnet.meter import ConfusionMeter

from org.campagnelab.dl.pytorch.cifar10.AccuracyHelper import AccuracyHelper
from org.campagnelab.dl.pytorch.cifar10.FloatHelper import FloatHelper
from org.campagnelab.dl.pytorch.cifar10.LRHelper import LearningRateHelper
from org.campagnelab.dl.pytorch.cifar10.LossHelper import LossHelper
from org.campagnelab.dl.pytorch.cifar10.PerformanceList import PerformanceList
from org.campagnelab.dl.pytorch.cifar10.datasets.SubsetDataset import SubsetDataset
from org.campagnelab.dl.pytorch.cifar10.utils import progress_bar, grad_norm, init_params
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


def print_params(epoch, net):
    params = []
    for param in net.parameters():
        params += [p for p in param.view(-1).data]
    print("epoch=" + str(epoch) + " " + " ".join(map(str, params)))


class TrainModelUnsupDirect:
    """Train a model using the unsupervised direct approach. This approach uses unsupervised samples
    to complement the training set, and makes up labels. """

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

        if hasattr(args, 'resume') and args.resume:
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
                self.best_model = checkpoint['best-model']
                self.best_model_confusion_matrix = checkpoint['confusion-matrix']
                model_built = True
            else:
                print("Could not load model checkpoint, unable to --resume.")
                model_built = False

        if not model_built:
            print('==> Building model {}'.format(args.model))

            self.net = create_model_function(args.model, self.problem)

        if self.use_cuda:
            self.net.cuda()
            if self.best_model is not None:
                self.best_model.cuda()
                self.best_model_confusion_matrix = self.best_model_confusion_matrix.cuda()

        cudnn.benchmark = True

        self.optimizer_training = torch.optim.SGD(self.net.parameters(), lr=args.lr, momentum=args.momentum,
                                                  weight_decay=args.L2)

        self.scheduler_train = \
            construct_scheduler(self.optimizer_training, 'max', factor=0.5,
                                lr_patience=self.args.lr_patience if hasattr(self.args, 'lr_patience') else 10,
                                ureg_reset_every_n_epoch=self.args.reset_lr_every_n_epochs
                                if hasattr(self.args, 'reset_lr_every_n_epochs')
                                else None)

    def train(self, epoch,
              performance_estimators=None,
              train_supervised_model=True,
              ):

        if performance_estimators is None:
            performance_estimators = PerformanceList()
            performance_estimators += [LossHelper("optimized_loss")]
            performance_estimators += [LossHelper("train_loss")]

            performance_estimators += [AccuracyHelper("train_")]
            performance_estimators += [FloatHelper("train_grad_norm")]
            print('\nTraining, epoch: %d' % epoch)

        self.net.train()
        supervised_grad_norm = 1.
        for performance_estimator in performance_estimators:
            performance_estimator.init_performance_metrics()

        unsupervised_loss_acc = 0
        num_batches = 0
        unsup_examples = numpy.random.random_integers(0, self.args.num_shaving - 1,
                                                      int(self.args.unsup_proportion * self.args.num_training))

        if self.args.label_strategy == "RANDOM_UNIFORM":
            made_up_label = lambda index: randint(0, self.problem.num_classes() - 1)
        else:
            print(
                "Unsupported --label-strategy: " + self.args.label_strategy + " only RANDOM_UNIFORM is supported with this mode.")
            exit(1)

        training_dataset = ConcatDataset(
            datasets=[
                SubsetDataset(self.problem.train_set(), range(0, self.args.num_training)),
                SubsetDataset(self.problem.unsup_set(), unsup_examples, get_label=made_up_label)])
        length = len(training_dataset)
        train_loader_subset = torch.utils.data.DataLoader(training_dataset,
                                                          batch_size=self.problem.mini_batch_size(),
                                                          shuffle=True,
                                                          num_workers=0)

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

            if train_supervised_model:
                supervised_loss = self.criterion(outputs, targets)
                optimized_loss = supervised_loss
                optimized_loss.backward()
                self.optimizer_training.step()
                supervised_grad_norm = grad_norm(self.net.parameters())
                performance_estimators.set_metric(batch_idx, "train_grad_norm", supervised_grad_norm)
                performance_estimators.set_metric_with_outputs(batch_idx, "optimized_loss", optimized_loss.data[0],
                                                               outputs, targets)
                performance_estimators.set_metric_with_outputs(batch_idx, "train_accuracy", supervised_loss.data[0],
                                                               outputs, targets)
                performance_estimators.set_metric_with_outputs(batch_idx, "train_loss", supervised_loss.data[0],
                                                               outputs, targets)

            progress_bar(batch_idx * self.mini_batch_size,
                         length,
                         performance_estimators.progress_message(["train_loss", "train_accuracy"]))

            if (batch_idx + 1) * self.mini_batch_size > self.max_training_examples:
                break

        return performance_estimators

    def train_unsup_only(self, epoch, unsup_index_to_label,
                         performance_estimators=None
                         ):
        """ Continue training a model on the unsupervised set with labels.
        :param epoch:
        :param unsup_index_to_label: map from index of the unsupervised example to label (in one hot encoding format, one element per class)
        :param performance_estimators:
        :param train_supervised_model:
        :return:
        """
        if performance_estimators is None:
            performance_estimators = PerformanceList()
            performance_estimators += [LossHelper("optimized_loss")]
            performance_estimators += [LossHelper("train_loss")]
            performance_estimators += [FloatHelper("train_grad_norm")]

        # reset the model before training:
        #init_params(self.net)
        print('\nTraining, epoch: %d' % epoch)

        self.net.train()
        train_supervised_model = True
        for performance_estimator in performance_estimators:
            performance_estimator.init_performance_metrics()

        num_batches = 0
        training_dataset = SubsetDataset(self.problem.unsup_set(),
                                         range(0,self.args.num_shaving), get_label=lambda index: unsup_index_to_label[index])
        length = len(training_dataset)
        train_loader_subset = torch.utils.data.DataLoader(training_dataset,
                                                          batch_size=self.problem.mini_batch_size(),
                                                          shuffle=False,
                                                          num_workers=0)
        self.optimizer_training = torch.optim.SGD(self.net.parameters(), lr=self.args.lr, momentum=self.args.momentum,
                                                  weight_decay=self.args.L2)

        # we use binary cross-entropy for single label with smoothing.
        self.net.train()
        criterion = BCELoss()
        for batch_idx, (inputs, targets) in enumerate(train_loader_subset):
            num_batches += 1

            if self.use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()

            inputs, targets = Variable(inputs), Variable(targets, requires_grad=False)
            # outputs used to calculate the loss of the supervised model
            # must be done with the model prior to regularization:

            self.optimizer_training.zero_grad()
            outputs = self.net(inputs)
            # renormalize outputs by example, from multi-label to single label prediction::
            outputs=torch.renorm(torch.exp(outputs), p=1, maxnorm=1, dim=1)

            supervised_loss = criterion(outputs, targets)
            optimized_loss = supervised_loss
            optimized_loss.backward()
            self.optimizer_training.step()
            supervised_grad_norm = grad_norm(self.net.parameters())
            performance_estimators.set_metric(batch_idx, "train_grad_norm", supervised_grad_norm)
            performance_estimators.set_metric_with_outputs(batch_idx, "optimized_loss", optimized_loss.data[0],
                                                           outputs, targets)
            performance_estimators.set_metric_with_outputs(batch_idx, "train_loss", supervised_loss.data[0],
                                                           outputs, targets)

            progress_bar(batch_idx * self.mini_batch_size,
                         length,
                         performance_estimators.progress_message(["train_loss", "train_accuracy"]))

        return performance_estimators

    def test(self, epoch, performance_estimators=None):
        print('\nTesting, epoch: %d' % epoch)
        if performance_estimators is None:
            performance_estimators = PerformanceList()
            performance_estimators += [LossHelper("test_loss"), AccuracyHelper("test_")]

        self.net.eval()
        for performance_estimator in performance_estimators:
            performance_estimator.init_performance_metrics()
        cm = ConfusionMeter(self.problem.num_classes(), normalized=False)

        for batch_idx, (inputs, targets) in enumerate(self.problem.test_loader_range(0, self.args.num_validation)):

            if self.use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            inputs, targets = Variable(inputs, volatile=True), Variable(targets, volatile=True)
            outputs = self.net(inputs)
            loss = self.criterion(outputs, targets)
            # accumulate the confusion matrix:
            _, predicted = torch.max(outputs.data, 1)

            cm.add(predicted=predicted, target=targets.data)
            performance_estimators.set_metric_with_outputs(batch_idx, "test_loss", loss.data[0], outputs, targets)
            performance_estimators.set_metric_with_outputs(batch_idx, "test_accuracy", loss.data[0], outputs, targets)

            progress_bar(batch_idx * self.mini_batch_size, self.max_validation_examples,
                         performance_estimators.progress_message(["test_loss", "test_accuracy"]))

            if ((batch_idx + 1) * self.mini_batch_size) > self.max_validation_examples:
                break
        # print()

        # Apply learning rate schedule:
        test_accuracy = performance_estimators.get_metric("test_accuracy")
        assert test_accuracy is not None, "test_accuracy must be found among estimated performance metrics"
        if not self.args.constant_learning_rates:
            self.scheduler_train.step(test_accuracy, epoch)
        self.confusion_matrix = cm.value().transpose()
        return performance_estimators

    def log_performance_header(self, performance_estimators, kind="perfs"):
        global best_test_loss
        if (self.args.resume):
            return

        metrics = ["epoch", "checkpoint"]

        metrics = metrics + performance_estimators.metric_names()

        if not self.args.resume:
            with open("all-{}-{}.tsv".format(kind, self.args.checkpoint_key), "w") as perf_file:
                perf_file.write("\t".join(map(str, metrics)))
                perf_file.write("\n")

            with open("best-{}-{}.tsv".format(kind, self.args.checkpoint_key), "w") as perf_file:
                perf_file.write("\t".join(map(str, metrics)))
                perf_file.write("\n")

    def log_performance_metrics(self, epoch, performance_estimators, kind="perfs"):
        """
        Log metrics and returns the best performance metrics seen so far.
        :param epoch:
        :param performance_estimators:
        :return: a list of performance metrics corresponding to the epoch where test accuracy was maximum.
        """
        metrics = [epoch, self.args.checkpoint_key]
        metrics = metrics + performance_estimators.estimates_of_metrics()

        early_stop = False
        with open("all-{}-{}.tsv".format(kind, self.args.checkpoint_key), "a") as perf_file:
            perf_file.write("\t".join(map(_format_nice, metrics)))
            perf_file.write("\n")
        if self.best_performance_metrics is None:
            self.best_performance_metrics = performance_estimators

        metric = performance_estimators.get_metric("test_accuracy")
        if metric is not None and metric > self.best_acc:
            self.failed_to_improve = 0

            with open("best-{}-{}.tsv".format(kind, self.args.checkpoint_key), "a") as perf_file:
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

    def epoch_is_test_epoch(self, epoch):
        epoch_is_one_of_last_ten = epoch > (self.start_epoch + self.args.num_epochs - 10)
        return (epoch % self.args.test_every_n_epochs + 1) == 1 or epoch_is_one_of_last_ten

    def training_supervised(self, unsup_only=False):
        """Train the model in a completely supervised manner. Returns the performance obtained
           at the end of the configured training run.
        :param unsup_only Set to true to train with dreamed-up labels on the unsupervised examples only.
        :return list of performance estimators that observed performance on the last epoch run.
        """
        header_written = False

        lr_train_helper = LearningRateHelper(scheduler=self.scheduler_train, learning_rate_name="train_lr")
        previous_test_perfs = None
        perfs = PerformanceList()
        best_test_loss = sys.maxsize
        num_rollbacks = 0
        epochs_since_rollback = 0

        if unsup_only:
            assert self.best_model is not None, "best model cannot be None to continue training with unsup only."
            # scan the unsupervised set to calculate labels using the previously trained best model:
            print("Calculating labels for unsupervised set..")

            unsup_index_to_labels = {}
            unsup_set_loader = self.problem.loader_for_dataset(self.problem.unsup_set())
            for batch_idx, (inputs, _) in enumerate(unsup_set_loader):
                if self.use_cuda:
                    inputs = inputs.cuda()
                inputs = Variable(inputs, volatile=True)
                predicted = self.best_model(inputs)
                if predicted.size()[1]==self.problem.num_classes():
                    # need to take the argmax to find the index of predicted class.
                    _, predicted = torch.max(predicted.data, 1)
                    predicted = predicted.type(torch.cuda.LongTensor) if self.use_cuda else predicted.type(torch.LongTensor)

                select = torch.index_select(self.best_model_confusion_matrix, dim=0, index=predicted)
                select=select.type(torch.cuda.FloatTensor)  if self.use_cuda else select.type(torch.FloatTensor)
                confusion_labels = torch.renorm(select, p=1, dim=1, maxnorm=1)
                start_of_range=batch_idx * self.problem.mini_batch_size()
                for example_index in range(start_of_range,
                                           start_of_range + self.problem.mini_batch_size()):
                    label_for_example=confusion_labels[example_index-start_of_range]
                    unsup_index_to_labels[example_index]=label_for_example
                progress_bar(batch_idx, self.args.num_shaving/self.problem.mini_batch_size())
                if batch_idx*self.problem.mini_batch_size()>self.args.num_shaving:
                    break
            self.net=self.best_model
            print("Training with unsupervised set..")
        for epoch in range(self.start_epoch, self.start_epoch + self.args.num_epochs):

            perfs = PerformanceList()

            perfs += self.train_unsup_only(epoch, unsup_index_to_label=unsup_index_to_labels) if unsup_only else \
                self.train(epoch, train_supervised_model=True)

            perfs += [lr_train_helper]
            if previous_test_perfs is None or self.epoch_is_test_epoch(epoch):
                perfs += self.test(epoch)

            test_loss = perfs.get_metric("test_loss")
            if (not header_written):
                header_written = True
                self.log_performance_header(perfs)

            early_stop, perfs = self.log_performance_metrics(epoch, perfs)
            if early_stop:
                # early stopping requested.
                return perfs
            if self.args.rollback_when_worse and epochs_since_rollback > 5 and test_loss > best_test_loss:
                self.net = self.load_checkpoint()
                if self.use_cuda: self.net.cuda()
                print("Rolled-back")
                num_rollbacks += 1
                epochs_since_rollback = 0
            else:
                best_test_loss = test_loss
                epochs_since_rollback += 1
                print("best test loss={} rolled-back {} times.".format(best_test_loss, num_rollbacks))
        return perfs
