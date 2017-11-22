import os
import torch
from torch.backends import cudnn
from torch.optim.lr_scheduler import ReduceLROnPlateau

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

    def init_model(self, create_model_function):
        """Resume training if necessary (args.--resume flag is True), or call the
        create_model_function to initialize a new model. This function must be called
        before train.

        The create_model_function takes one argument: the name of the model to be
        created.
        """
        args = self.args
        unsuploader = self.problem.reg_loader()
        trainloader = self.problem.train_loader()
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

                ureg.set_num_examples(min(len(trainloader), args.num_training),
                                      min(len(unsuploader), args.num_shaving))
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
        ureg = URegularizer(self.net, self.mini_batch_size, num_features=args.ureg_num_features,
                            alpha=args.ureg_alpha,
                            learning_rate=args.ureg_learning_rate)
        if args.ureg:
            ureg.enable()
            ureg.set_num_examples(args.num_training, len(unsuploader))
            ureg.forget_model(args.ureg_reset_every_n_epoch)
            print(
                "ureg is enabled with alpha={}, reset every {} epochs. ".format(args.ureg_alpha,
                                                                                args.ureg_reset_every_n_epoch))

        else:
            ureg.disable()
            ureg.set_num_examples(args.num_training, len(unsuploader))
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
