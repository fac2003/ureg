import os
from random import randint

import numpy
import sys
import torch
from torch.autograd import Variable
from torch.backends import cudnn
from torch.nn import MultiLabelSoftMarginLoss, BCELoss, MSELoss
from torchnet.dataset import ConcatDataset
from torchnet.meter import ConfusionMeter
from torchvision.utils import save_image

from org.campagnelab.dl.pytorch.images.AccuracyHelper import AccuracyHelper
from org.campagnelab.dl.pytorch.images.FloatHelper import FloatHelper
from org.campagnelab.dl.pytorch.images.ImageEncoder import ImageEncoder
from org.campagnelab.dl.pytorch.images.ImageGenerator import ImageGenerator
from org.campagnelab.dl.pytorch.images.LRHelper import LearningRateHelper
from org.campagnelab.dl.pytorch.images.LossHelper import LossHelper
from org.campagnelab.dl.pytorch.images.PerformanceList import PerformanceList
from org.campagnelab.dl.pytorch.images.SplitHelper import half_images, get_random_slope
from org.campagnelab.dl.pytorch.images.datasets.SubsetDataset import SubsetDataset
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


class TrainModelDeconvolutionSplit:
    """Train a model using the unsupervised deconvolution split approach. This approach
    splits unsupervised images in two and trains to reconstruct one half from the other,
    using a deconvolution image generator. """

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
        self.best_loss = sys.maxsize
        self.start_epoch = 0
        self.use_cuda = use_cuda
        self.mini_batch_size = problem.mini_batch_size()
        self.net = None
        self.optimizer = None
        self.scheduler_train = None
        self.unsuploader = self.problem.reg_loader()
        self.trainloader = self.problem.train_loader()
        self.testloader = self.problem.test_loader()
        self.is_parallel = False
        self.best_performance_metrics = None
        self.failed_to_improve = 0
        self.confusion_matrix = None
        self.best_model_confusion_matrix = None
        self.best_acc=0

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
        self.optimizer = None
        # try loading a pre-trained model:
        encoder, generator, test_loss, epoch = self.load_pretrained()
        if not args.pretrain and encoder is None:
            print("You did not select --pretrain, and was not able to load pre-trained model with pretrained-key="+args.pretrained_key)
            exit(1)
        if hasattr(args, 'resume') and args.resume:
            # Load checkpoint.

            print('==> Resuming from checkpoint..')

            assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
            checkpoint = None

            try:
                checkpoint = torch.load('./checkpoint/ckpt_{}.t7'.format(args.checkpoint_key))
            except FileNotFoundError:
                pass
            if checkpoint is not None:

                self.net = checkpoint['net'] 
                # use the pretrained data for encoder and generator if available:
                self.image_encoder = checkpoint['encoder'] if encoder is None else encoder
                self.image_generator = checkpoint['generator']  if generator is None else generator
                self.best_loss = checkpoint['test_loss'] if not self.args.pretrain else test_loss
                self.start_epoch = checkpoint['epoch'] if not self.args.pretrain else epoch

                # rebuild a model fresh, we only
                model_built = True
            else:
                print("Could not load model checkpoint, unable to --resume.")
                model_built = False
                self.best_loss = test_loss
                self.start_epoch = epoch

        if not model_built:
            print('==> Building model {}'.format(args.model))

            self.net = create_model_function(args.model, self.problem)
            if self.use_cuda:
                self.net.cuda()

            self.image_encoder = ImageEncoder(model=self.net, number_encoder_features=self.args.num_encoder_features,
                                              number_representation_features=self.args.num_representation_features,
                                              input_shape=self.problem.example_size(), use_cuda=self.use_cuda,
                                              ngpu=self.args.n_gpus) \
                if encoder is None else encoder

            self.image_generator = ImageGenerator(number_encoded_features=self.args.num_representation_features,
                                                  number_of_generator_features=self.args.num_generator_features,
                                                  output_shape=self.problem.example_size(),
                                                  use_cuda=self.use_cuda,
                                                  ngpu=self.args.n_gpus) \
                if generator is None else generator
            print(self.image_encoder)
            if self.use_cuda:

                self.image_encoder.cuda()
                self.image_generator.cuda()
            if self.best_model is not None:
                self.best_model.cuda()
                self.best_model_confusion_matrix = self.best_model_confusion_matrix.cuda()
        cudnn.benchmark = True
        all_params = []

        all_params += list(self.image_generator.main.parameters())
        all_params += list(self.image_encoder.projection.parameters())
        all_params += list(self.image_encoder.main.parameters())

        self.optimizer = torch.optim.Adam(all_params, lr=self.args.lr, betas=(0.5, 0.999), weight_decay=self.args.L2)

        self.scheduler_train = \
            construct_scheduler(self.optimizer, 'min', factor=0.5,
                                lr_patience=self.args.lr_patience if hasattr(self.args, 'lr_patience') else 10,
                                ureg_reset_every_n_epoch=self.args.reset_lr_every_n_epochs
                                if hasattr(self.args, 'reset_lr_every_n_epochs')
                                else None)

    def train_unsup_only(self, epoch,
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
            performance_estimators += [LossHelper("train_loss")]
            performance_estimators += [FloatHelper("encoder_grad_norm")]
            performance_estimators += [FloatHelper("generator_grad_norm")]
            performance_estimators += [FloatHelper("net_grad_norm")]

        print('\nTraining, epoch: %d' % epoch)

        train_supervised_model = True
        for performance_estimator in performance_estimators:
            performance_estimator.init_performance_metrics()

        num_batches = 0
        training_dataset = SubsetDataset(self.problem.unsup_set(),
                                         range(0, self.args.num_shaving), get_label=lambda index: 1)
        length = len(training_dataset)
        train_loader_subset = torch.utils.data.DataLoader(training_dataset,
                                                          batch_size=self.problem.mini_batch_size(),
                                                          shuffle=True,
                                                          num_workers=0)

        # we use binary cross-entropy for single label with smoothing.

        criterion = MSELoss()

        self.net.train()
        self.image_generator.train()
        self.image_encoder.train()

        for batch_idx, (inputs, _) in enumerate(train_loader_subset):
            num_batches += 1

            if self.use_cuda:
                inputs = inputs.cuda()

            self.optimizer.zero_grad()

            image1, image2 = half_images(inputs, slope=get_random_slope(), cuda=self.use_cuda)
            # train the discriminator/generator pair on the first half of the image:
            encoded = self.image_encoder(image1)
            # norm_encoded=encoded.norm(p=1)
            output = self.image_generator(encoded,)
            full_image=Variable(inputs,requires_grad=False)
            optimized_loss = criterion(output, full_image)
            optimized_loss.backward()
            self.optimizer.step()

            if batch_idx == 0:
                self.save_images(epoch, image1, image2, generated_image2=output, prefix="train")

            encoder_grad_norm = grad_norm(self.image_encoder.parameters())
            generator_grad_norm = grad_norm(self.image_generator.parameters())
            net_grad_norm = grad_norm(self.net.parameters())
            performance_estimators.set_metric(batch_idx, "encoder_grad_norm", encoder_grad_norm)
            performance_estimators.set_metric(batch_idx, "generator_grad_norm", generator_grad_norm)
            performance_estimators.set_metric(batch_idx, "net_grad_norm", net_grad_norm)
            performance_estimators.set_metric(batch_idx, "train_loss", optimized_loss.data[0])
            progress_bar(batch_idx * self.mini_batch_size,
                         length,
                         performance_estimators.progress_message(["train_loss", "train_accuracy"]))

        return performance_estimators

    def test(self, epoch, performance_estimators=None):
        criterion = MSELoss()
        print('\nTesting, epoch: %d' % epoch)
        self.net.eval()
        self.image_generator.eval()
        self.image_encoder.eval()

        if performance_estimators is None:
            performance_estimators = PerformanceList()
            performance_estimators += [LossHelper("test_loss"), AccuracyHelper("test_")]

        for performance_estimator in performance_estimators:
            performance_estimator.init_performance_metrics()
        # we used unsup set to train, use training to validate:
        for batch_idx, (inputs, _) in enumerate(self.problem.train_loader_subset(range(0, self.args.num_validation))):

            if self.use_cuda:
                inputs = inputs.cuda()

            image1, image2= half_images(inputs, slope=get_random_slope(), cuda=self.use_cuda)
            # train the discriminator/generator pair on the first half of the image:
            encoded = self.image_encoder(image1)

            output = self.image_generator(encoded)

            if batch_idx == 0:
                self.save_images(epoch, image1, image2, generated_image2=output)
            full_image = Variable(inputs, requires_grad=False)
            loss = criterion(output, full_image)
            performance_estimators.set_metric(batch_idx, "test_loss", loss.data[0])

            progress_bar(batch_idx * self.mini_batch_size, self.max_validation_examples,
                         performance_estimators.progress_message(["test_loss"]))

            if ((batch_idx + 1) * self.mini_batch_size) > self.max_validation_examples:
                break
        # print()

        # Apply learning rate schedule:
        test_loss = performance_estimators.get_metric("test_loss")
        assert test_loss is not None, "test_loss must be found among estimated performance metrics"
        if not self.args.constant_learning_rates:
            self.scheduler_train.step(test_loss, epoch)
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


        if self.metric_is_better(performance_estimators):
            self.failed_to_improve = 0

            with open("best-{}-{}.tsv".format(kind, self.args.checkpoint_key), "a") as perf_file:
                perf_file.write("\t".join(map(_format_nice, metrics)))
                perf_file.write("\n")

            if self.args.pretrain:
                self.save_pretrained(epoch, performance_estimators.get_metric("test_loss"))
            else:
                self.save_checkpoint(epoch,  performance_estimators.get_metric("test_accuracy"))
            self.best_performance_metrics = performance_estimators

        if not self.metric_is_better(performance_estimators):
            self.failed_to_improve += 1
            if hasattr(self.args, "abort_when_failed_to_improve") and \
                self.failed_to_improve > self.args.abort_when_failed_to_improve:
                print("We failed to improve for {} epochs. Stopping here as requested.")
                early_stop = True  # request early stopping

        return early_stop, self.best_performance_metrics

    def save_pretrained(self, epoch, metric_value):

        # Save checkpoint.

        if metric_value < self.best_loss:
            print('Saving..')
            model = self.net
            model.eval()
            generator = self.image_generator
            generator.eval()
            encoder = self.image_encoder
            encoder.eval()
            state = {

                'generator': generator.module if self.is_parallel else generator,
                'encoder': encoder.module if self.is_parallel else encoder,
                'test_loss': metric_value,
                'epoch': epoch,
            }
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            torch.save(state, './checkpoint/pretrained_{}.t7'.format(self.args.pretrained_key))
            self.best_loss = metric_value

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

    def load_pretrained(self):
        try:
            state = torch.load('./checkpoint/pretrained_{}.t7'.format(self.args.pretrained_key))

            encoder = state['encoder']
            generator = state['generator']
            test_loss=state['test_loss']
            epoch=state['epoch']
            for m in [generator, encoder]:
                m.cpu()
                m.eval()
            print("Loaded pretrained encoder and generator for {} test_loss={} epoch={}".format( self.args.checkpoint_key,
                                                                                                  test_loss, epoch))
            return ( encoder, generator, test_loss, epoch)
        except:
            return (None,None,None,None)
        
    def epoch_is_test_epoch(self, epoch):
        epoch_is_one_of_last_ten = epoch > (self.start_epoch + self.args.num_epochs - 10)
        return (epoch % self.args.test_every_n_epochs + 1) == 1 or epoch_is_one_of_last_ten

    def training_deconvolution(self):
        """Train the model using reconstructed loss on half an image.
        :return list of performance estimators that observed performance on the last epoch run.
        """
        header_written = False

        lr_train_helper = LearningRateHelper(scheduler=self.scheduler_train, learning_rate_name="train_lr")
        previous_test_perfs = None
        perfs = PerformanceList()

        print("Training with unsupervised set..")
        for epoch in range(self.start_epoch, self.start_epoch + self.args.num_epochs):

            perfs = PerformanceList()

            perfs += self.train_unsup_only(epoch)

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

    def train_with_reconstructed_half(self):
        """Train the model using two half images: one half original from training set, the other reconstructed
         with encoder/generator trained on unsup set.
        """
        header_written = False
        self.optimizer = None
        optimizer_training=torch.optim.Adam(self.net.parameters(), lr=self.args.lr, betas=(0.5, 0.999), weight_decay=self.args.L2)


        lr_train_helper = LearningRateHelper(scheduler=self.scheduler_train, learning_rate_name="train_lr")
        previous_test_perfs = None
        perfs = PerformanceList()

        print("Training with unsupervised half..")
        for epoch in range(self.start_epoch, self.start_epoch + self.args.num_epochs):

            perfs = PerformanceList()

            perfs += self.train_with_two_halves(epoch, optimizer_training)

            perfs += [lr_train_helper]
            if previous_test_perfs is None or self.epoch_is_test_epoch(epoch):
                perfs += self.test_acc(epoch)

            if (not header_written):
                header_written = True
                self.log_performance_header(perfs)

            early_stop, perfs = self.log_performance_metrics(epoch, perfs)
            if early_stop:
                # early stopping requested.
                return perfs

        return perfs

    def train_with_two_halves(self, epoch, optimizer_training,
              performance_estimators=None,
              train_supervised_model=True,
              ):

        if performance_estimators is None:
            performance_estimators = PerformanceList()
            performance_estimators += [LossHelper("supervised_loss")]
            performance_estimators += [LossHelper("unsup_loss")]

            performance_estimators += [AccuracyHelper("train_")]
            performance_estimators += [FloatHelper("supervised_grad_norm")]
            performance_estimators += [FloatHelper("unsup_grad_norm")]
            print('\nTraining, epoch: %d' % epoch)

        self.net.train()
        supervised_grad_norm = 1.
        for performance_estimator in performance_estimators:
            performance_estimator.init_performance_metrics()

        unsupervised_loss_acc = 0
        num_batches = 0
        train_loader_subset = self.problem.train_loader_subset_range(0, self.args.num_training)
        self.image_encoder.eval()
        self.image_generator.eval()

        self.net.train()
        for batch_idx, (inputs, targets) in enumerate(train_loader_subset):
            num_batches += 1

            if self.use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            self.net.zero_grad()
            optimizer_training.zero_grad()
            image1, image2 = half_images(inputs, slope=get_random_slope(), cuda=self.use_cuda)
            encoded = self.image_encoder(image1)
            unsup_image = self.image_generator(encoded)

            if self.args.mode=="separate":
                 # train the discriminator/generator pair on the first half of the image:

                inputs, targets = Variable(inputs), Variable(targets, requires_grad=False)
                outputs = self.net(unsup_image.detach())

                unsup_loss = self.criterion(outputs, targets)
                unsup_loss.backward()
                unsup_grad_norm = grad_norm(self.net.parameters())

                optimizer_training.step()
                # outputs used to calculate the loss of the supervised model
                # must be done with the model prior to regularization:

                self.net.zero_grad()
                optimizer_training.zero_grad()


                outputs = self.net(inputs)
                supervised_loss = self.criterion(outputs, targets)

                supervised_grad_norm = grad_norm(self.net.parameters())

                supervised_loss.backward()
                unsup_grad_norm = grad_norm(self.net.parameters())

                optimizer_training.step()

            elif self.args.mode=="average":
                inputs=(unsup_image.detach()+inputs)/2
                inputs, targets = Variable(inputs), Variable(targets, requires_grad=False)

                outputs = self.net(inputs)
                supervised_loss = self.criterion(outputs, targets)
                supervised_grad_norm = grad_norm(self.net.parameters())
                supervised_loss.backward()
                optimizer_training.step()

            performance_estimators.set_metric(batch_idx, "supervised_grad_norm", supervised_grad_norm)
            performance_estimators.set_metric(batch_idx, "unsup_grad_norm", unsup_grad_norm)
            performance_estimators.set_metric_with_outputs(batch_idx, "supervised_loss", supervised_loss.data[0],
                                                           outputs, targets)
            performance_estimators.set_metric_with_outputs(batch_idx, "unsup_loss", unsup_loss.data[0],
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

    def test_acc(self, epoch, performance_estimators=None):
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
            image1, image2 = half_images(inputs, slope=get_random_slope(), cuda=self.use_cuda)
            encoded = self.image_encoder(image1)
            unsup_image = self.image_generator(encoded)

            if self.args.mode == "separate":

                inputs, targets = Variable(inputs, volatile=True), Variable(targets, volatile=True)
                outputs = self.net(inputs)
            elif self.args.mode == "average":
                inputs = (unsup_image.detach() + inputs) / 2
                inputs, targets = Variable(inputs), Variable(targets, requires_grad=False)

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


    def save_images(self, epoch, real_image1, real_image2, generated_image2, prefix="val"):
        try:
            os.stat("outputs")
        except:
            os.mkdir("outputs")

        save_image(real_image1.data,
                   '{}/{}-real_image1.png'.format("outputs", prefix),
                   normalize=True)

        save_image(real_image2.data,
                   '{}/{}-real_image2.png'.format("outputs", prefix),
                   normalize=True)

        # train the discriminator/generator pair on the first half of the image:

        save_image( generated_image2.data,
                   '{}/{}-fake_samples_split_epoch_{}.png'.format("outputs", prefix, epoch),
                   normalize=True)

    def metric_is_better(self, performance_estimators):

        if self.args.pretrain:
            metric = performance_estimators.get_metric("test_loss")
            if metric is None: return False
            return metric<= self.best_loss
        else:
            metric = performance_estimators.get_metric("test_accuracy")
            if metric is None: return False
            return metric>=self.best_acc
