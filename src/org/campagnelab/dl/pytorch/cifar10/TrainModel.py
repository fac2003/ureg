import os
import torch
from torch.autograd import Variable
from torch.backends import cudnn
from torch.optim.lr_scheduler import ReduceLROnPlateau

from org.campagnelab.dl.pytorch.cifar10.Evaluate2 import TrainingPerf
from org.campagnelab.dl.pytorch.cifar10.utils import progress_bar
from org.campagnelab.dl.pytorch.ureg.LRSchedules import LearningRateAnnealing
from org.campagnelab.dl.pytorch.ureg.URegularizer import URegularizer


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
        self.tesstloader = self.problem.test_loader()
        self.ureg=None

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

        delegate_scheduler = ReduceLROnPlateau(self.optimizer_training, 'min', factor=0.5,
                                               patience=args.lr_patience, verbose=True)

        if args.ureg_reset_every_n_epoch is None:
            self.scheduler_train = delegate_scheduler
        else:
            self.scheduler_train = LearningRateAnnealing(self.optimizer_training,
                                                         anneal_every_n_epoch=args.ureg_reset_every_n_epoch,
                                                         delegate=delegate_scheduler)

        self.scheduler_reg = ReduceLROnPlateau(self.optimizer_reg, 'min', factor=0.5, patience=args.lr_patience,
                                               verbose=True)

    def train(self,epoch, unsupiter):
        print('\nTraining, epoch: %d' % epoch)
        self.net.train()
        average_total_loss = 0
        average_supervised_loss = 0
        average_unsupervised_loss = 0
        correct = 0
        total = 0
        average_total_loss = 0
        training_accuracy = 0

        for batch_idx, (inputs, targets) in enumerate(self.trainloader):

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

            self.ureg.train_ureg(inputs, uinputs)

            optimized_loss = supervised_loss
            average_total_loss += optimized_loss.data[0]
            average_supervised_loss += supervised_loss.data[0]

            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()

            denominator = batch_idx + 1
            average_total_loss = average_total_loss / denominator
            average_supervised_loss = average_supervised_loss / denominator
            average_unsupervised_loss = average_unsupervised_loss / denominator
            training_accuracy = 100. * correct / total
            progress_bar(batch_idx, len(self.problem.train_loader()),
                         ('loss: %.3f s: %.3f u: %.3f | Acc: %.3f%% (%d/%d)'
                          % (average_total_loss,
                             average_supervised_loss,
                             average_unsupervised_loss,
                             training_accuracy,
                             correct,
                             total)))
            if (batch_idx + 1) * self.mini_batch_size > self.max_training_examples:
                break
        self.scheduler_reg.step(average_supervised_loss, epoch)
        print()

        return TrainingPerf(average_total_loss, average_supervised_loss, training_accuracy)
