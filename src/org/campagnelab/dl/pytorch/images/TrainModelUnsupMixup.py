import itertools
import os
from random import random

import numpy
import torch
from torch.autograd import Variable
from torch.backends import cudnn
from torch.nn import MSELoss, MultiLabelSoftMarginLoss
from torchnet.meter import ConfusionMeter

from org.campagnelab.dl.pytorch.images.AccuracyHelper import AccuracyHelper
from org.campagnelab.dl.pytorch.images.FloatHelper import FloatHelper
from org.campagnelab.dl.pytorch.images.LRHelper import LearningRateHelper
from org.campagnelab.dl.pytorch.images.LossHelper import LossHelper
from org.campagnelab.dl.pytorch.images.PerformanceList import PerformanceList
from org.campagnelab.dl.pytorch.images.models.dual import LossEstimator_L1, LossEstimator_L1_cpu, \
    LossEstimator_orthogonal, LossEstimator_sim_orthogonal, LossEstimator_sim
from org.campagnelab.dl.pytorch.images.models.preact_resnet_dual import PreActResNet18Dual
from org.campagnelab.dl.pytorch.images.models.vgg_dual import VGGDual
from org.campagnelab.dl.pytorch.images.utils import progress_bar, grad_norm, init_params
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


class TrainModelUnsupMixup:
    """Train a model using the unsupervised mixup apporach."""

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

        def rmse(y, y_hat):
            """Compute root mean squared error"""
            return torch.sqrt(torch.mean((y - y_hat).pow(2)))

        self.agreement_loss = MSELoss()

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

            self.net = create_model_function(args.model, self.problem, dual=self.args.mode=="fm_loss")

        if self.use_cuda:
            self.net.cuda()
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
        train_loader_subset = self.problem.train_loader_subset_range(0, self.args.num_training)

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
                # if self.ureg._which_one_model is not None:
                #    self.ureg.estimate_example_weights(inputs)

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
                         min(self.max_regularization_examples, self.max_training_examples),
                         " ".join([performance_estimator.progress_message() for performance_estimator in
                                   performance_estimators]))

            if (batch_idx + 1) * self.mini_batch_size > self.max_training_examples:
                break

        return performance_estimators

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
            self.net.zero_grad()
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
                         performance_estimators.progress_message(["train_loss","train_accuracy"]))

            if (batch_idx + 1) * self.mini_batch_size > self.max_training_examples:
                break

        return performance_estimators

    def train_with_fm_loss(self, epoch,
                    gamma=1E-5, performance_estimators=None):

        if performance_estimators is None:
            performance_estimators = PerformanceList()
            performance_estimators += [LossHelper("optimized_loss")]
            performance_estimators += [LossHelper("train_loss")]
            performance_estimators += [FloatHelper("fm_loss")]
            performance_estimators += [AccuracyHelper("train_")]
            performance_estimators += [FloatHelper("train_grad_norm")]

            print('\nTraining, epoch: %d' % epoch)

        self.net.train()
        supervised_grad_norm = 1.
        for performance_estimator in performance_estimators:
            performance_estimator.init_performance_metrics()

        unsupervised_loss_acc = 0
        num_batches = 0
        train_loader_subset = self.problem.train_loader_subset_range(0, self.args.num_training)
        #sec_train_loader_subset = self.problem.train_loader_subset_range(0, self.args.num_training)
        unsuploader_shuffled = self.problem.reg_loader_subset_range(0, self.args.num_shaving)
        unsupiter = itertools.cycle(unsuploader_shuffled)


        for batch_idx, ((inputs, targets),
                        (uinputs, _)) in enumerate(zip(train_loader_subset                                                        ,
                                                        unsupiter)):
            num_batches += 1

            if self.use_cuda:
                inputs = inputs.cuda()
                uinputs = uinputs.cuda()
                targets = targets.cuda()

            # outputs used to calculate the loss of the supervised model
            # must be done with the model prior to regularization:
            self.net.train()
            self.net.zero_grad()
            self.optimizer_training.zero_grad()
            inputs, targets, uinputs = Variable(inputs), Variable(targets, requires_grad=False), Variable(uinputs, requires_grad=True)
            if self.use_cuda:
                inputs, targets, uinputs=inputs.cuda(),targets.cuda(), uinputs.cuda()
            outputs, outputu, fm_loss = self.net(inputs,uinputs)

            supervised_loss = self.criterion(outputs, targets)
            optimized_loss = supervised_loss+gamma*fm_loss
            optimized_loss.backward()
            self.optimizer_training.step()
            supervised_grad_norm = grad_norm(self.net.parameters())
            performance_estimators.set_metric(batch_idx, "train_grad_norm", supervised_grad_norm)
            performance_estimators.set_metric(batch_idx, "optimized_loss", optimized_loss.data[0])
            performance_estimators.set_metric(batch_idx, "fm_loss", fm_loss.data[0])

            performance_estimators.set_metric_with_outputs(batch_idx, "train_loss", supervised_loss.data[0],
                                                           outputs, targets)
            performance_estimators.set_metric_with_outputs(batch_idx, "train_accuracy", supervised_loss.data[0],
                                                           outputs, targets)
            # performance_estimators.set_metric_with_outputs(batch_idx, "train_accuracy", supervised_loss.data[0],
            #                                               outputs, targets)

            progress_bar(batch_idx * self.mini_batch_size,
                         min(self.max_regularization_examples, self.max_training_examples),
                         performance_estimators.progress_message(["optimized_loss","train_loss","train_accuracy"]))

            if (batch_idx + 1) * self.mini_batch_size > self.max_training_examples:
                break

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

        inputs, targets = Variable(inputs, requires_grad=True), Variable(targets, requires_grad=False)
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
            best_model_output = self.best_model(Variable(inputs_gpu, volatile=True))
            _, predicted = torch.max(best_model_output.data)
            targets2 = best_model_output.data
        elif self.args.label_strategy == "VAL_CONFUSION":
            self.best_model.eval()
            # we use the best model we trained so far to predict the outputs. These labels will overfit to the
            # training set as training progresses:
            best_model_output = self.best_model(Variable(inputs_gpu, volatile=True))
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
            best_model_output = self.best_model(Variable(inputs_gpu, volatile=True))
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
            if not hasattr(self.net, 'is_dual'):
                outputs = self.net(inputs)
            else:
                outputs, _, _ =self.net(inputs,None)
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

    def training_mixup(self):
        """Train the model with unsupervised mixup. Returns the performance obtained
           at the end of the configured training run.
        :return list of performance estimators that observed performance on the last epoch run.
        """
        header_written = False

        lr_train_helper = LearningRateHelper(scheduler=self.scheduler_train, learning_rate_name="train_lr")
        previous_test_perfs = None
        perfs = PerformanceList()
        train_loss = None
        test_loss = None
        for epoch in range(self.start_epoch, self.start_epoch + self.args.num_epochs):

            perfs = PerformanceList()
            perfs += self.train_mixup(epoch,
                                       train_supervised_model=True,
                                       alpha=self.args.alpha,
                                       ratio_unsup=self.args.unsup_proportion
                                       )

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

            perfs += [lr_train_helper]
            if previous_test_perfs is None or self.epoch_is_test_epoch(epoch):
                perfs += self.test(epoch)

            if (not header_written):
                header_written = True
                self.log_performance_header(perfs)

            early_stop, perfs = self.log_performance_metrics(epoch, perfs)
            if early_stop:
                # early stopping requested.
                return perfs

        return perfs

    def training_fm_loss(self):
        """Train the model with unsupervised mixup. Returns the performance obtained
           at the end of the configured training run.
        :return list of performance estimators that observed performance on the last epoch run.
        """
        header_written = False
        loss_estimator=LossEstimator_sim
        # replace the loss function of the dual model:
        def set_loss(x):
            if  hasattr(x, 'loss_estimator'): x.loss_estimator = loss_estimator
        self.net.apply(set_loss)

        self.optimizer_training = torch.optim.SGD(self.net.parameters(), lr=self.args.lr, momentum=self.args.momentum,
                                                  weight_decay=self.args.L2)

        lr_train_helper = LearningRateHelper(scheduler=self.scheduler_train, learning_rate_name="train_lr")
        previous_test_perfs = None
        perfs = PerformanceList()
        train_loss = None
        test_loss = None
        for epoch in range(self.start_epoch, self.start_epoch + self.args.num_epochs):

            perfs = PerformanceList()
            perfs += self.train_with_fm_loss(epoch, self.args.gamma)

            perfs += [lr_train_helper]
            if previous_test_perfs is None or self.epoch_is_test_epoch(epoch):
                perfs += self.test(epoch)

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

            perfs += self.train(epoch,
                                     train_supervised_model=True)

            perfs += [lr_train_helper]
            if previous_test_perfs is None or self.epoch_is_test_epoch(epoch):
                perfs += self.test(epoch)


            if (not header_written):
                header_written = True
                self.log_performance_header(perfs)

            early_stop, perfs = self.log_performance_metrics(epoch, perfs)
            if early_stop:
                # early stopping requested.
                return perfs

        return perfs

