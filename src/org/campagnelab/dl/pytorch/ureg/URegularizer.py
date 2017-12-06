import random
import sys

import torch
from torch.autograd import Variable

from org.campagnelab.dl.pytorch.cifar10.FloatHelper import FloatHelper
from org.campagnelab.dl.pytorch.cifar10.LossHelper import LossHelper
from org.campagnelab.dl.pytorch.cifar10.PerformanceList import PerformanceList
from org.campagnelab.dl.pytorch.cifar10.utils import progress_bar
from org.campagnelab.dl.pytorch.ureg.LRSchedules import construct_scheduler
from org.campagnelab.dl.pytorch.ureg.ModelAssembler import ModelAssembler
from org.campagnelab.dl.pytorch.ureg.MyFeatureExtractor import MyFeatureExtractor


class URegularizer:
    """
    Unsupervised regularizer. This class estimates a regularization term using
    unlabeled examples and how similar the training examples are to the unlabeled
    ones. This term can be added to the supervised loss to avoid over-fitting to
    the training set.
    """

    def __init__(self, model, mini_batch_size, num_features=64, alpha=0.5,
                 learning_rate=0.1,
                 reset_every_epochs=None, do_not_use_scheduler=False,
                 momentum=0.9, l2=0, include_output_function=None):
        """
        :param model:
        :param mini_batch_size:
        :param num_features:
        :param alpha:
        :param learning_rate:
        :param reset_every_epochs:
        :param do_not_use_scheduler:
        :param momentum:
        :param l2:
        :param threshold_activation_size do not include layer outputs that have more activations than this threshold.

        """
        self.model_assembler = ModelAssembler(num_features, layer_predicate_function=include_output_function)
        self._mini_batch_size = mini_batch_size
        self._model = model
        self._num_activations = 0
        self._learning_rate = learning_rate
        self._which_one_model = None
        self._my_feature_extractor1 = None
        self._my_feature_extractor2 = None
        self._enabled = True
        self._use_cuda = torch.cuda.is_available()
        self._scheduler = None
        self._checkpointModel = None
        self._num_features = num_features
        self._epoch_counter = 0
        self._forget_every_n_epoch = None
        self._optimizer = None
        self._eps = 1E-8
        self._epoch_counter = 0
        self._n_total = 0
        self._n_correct = 0
        self._last_epoch_accuracy = 0.5
        self._accumulator_total_which_model_loss = 0
        self._num_accumulator_updates = 0
        self.num_training = 0
        self.num_unsupervised_examples = 0
        self._use_scheduler = False
        self._reset_every_epochs = reset_every_epochs
        self.do_not_use_scheduler = do_not_use_scheduler
        self.momentum = momentum
        self.L2 = l2
        self.chosen_activations = None
        self._alpha = alpha

    def add_activations(self, num):
        self._num_activations += num
        # print("activations: " + str(self.num_activations))

    def _estimate_accuracy(self, ys, ys_true):
        if (ys.size()[0] != self._mini_batch_size):
            return
        _, predicted = torch.max(ys.data, 1)
        _, truth = torch.max(ys_true.data, 1)
        self._n_correct += predicted.eq(truth).cpu().sum()
        self._n_total += self._mini_batch_size

    def _clear_accuracy(self):
        self._n_correct = 0
        self._n_total = 0

    def estimate_accuracy(self, xs):
        # num_activations_supervised = supervised_output.size()[1]

        self.create_which_one_model(xs)

        supervised_outputs = self.extract_activation_list(xs)  # print("length of output: "+str(len(supervised_output)))

        # now we use the model:
        self._which_one_model.eval()
        # the more the activations on the supervised set deviate from the unsupervised data,
        # the more we need to regularize:
        ys = self.model_assembler.evaluate(supervised_outputs)

        self._estimate_accuracy(ys, self.ys_true)

    def ureg_accuracy(self):
        if self._n_total == 0:
            return float("nan")
        accuracy = self._n_correct / self._n_total
        # print("ureg accuracy={0:.3f} correct: {1}/{2}".format(accuracy, self._n_correct, self._n_total))
        self._last_epoch_accuracy = accuracy
        self._n_total = 0
        self._n_correct = 0
        return accuracy

    def forget_model(self, N_epoch):
        """
        Reset the weights of the which_one_model model after N epoch of training. We need to reset once in a while
         to focus on the recent history of activations, not remember the distant past.
        :param N_epoch:
        :return:
        """
        self._forget_every_n_epoch = N_epoch

    def disable(self):
        self._enabled = False

    def enable(self):
        self._enabled = True

    def create_feature_extractors(self):
        if self._my_feature_extractor1 is None or self._my_feature_extractor2 is None:
            self._my_feature_extractor1 = MyFeatureExtractor(self._model, None)
            self._my_feature_extractor2 = MyFeatureExtractor(self._model, None)

    def schedule(self, val_loss, epoch):
        """
        Apply the learning rate schedule
        :param epoch: current epoch number
        :param val_loss: the validation/test loss
        :return: None
        """
        if self._scheduler is not None:
            self._scheduler.step(val_loss, epoch)

    def reset_model(self):
        self._which_one_model = None
        self.chosen_activations = None

    def create_which_one_model(self, xs):
        num_features = self._num_features
        if self._which_one_model is None:
            if (self._checkpointModel):
                self._which_one_model = self._checkpointModel
                self._checkpointModel = None
            else:
                activation_list = self.extract_activation_list(xs)

                self._which_one_model = self.model_assembler.assemble(activation_list)
                self.chosen_activations = self.model_assembler.get_collect_output()
            # (we will use BCEWithLogitsLoss on top of linar to get the cross entropy after
            # applying a sigmoid).
            self.model_assembler.init_params()

            print("done building which_one_model:" + str(self._which_one_model))
            if self._use_cuda: self._which_one_model.cuda()
            # self.optimizer = torch.optim.Adam(self.which_one_model.parameters(),
            #                                   lr=self.learning_rate)
            self._optimizer = torch.optim.SGD(self._which_one_model.parameters(), lr=self._learning_rate,
                                              momentum=self.momentum,
                                              weight_decay=self.L2)
            if self._use_scheduler:
                self._scheduler = construct_scheduler(self._optimizer, direction="min", lr_patience=1,
                                                      extra_patience=10,
                                                      ureg_reset_every_n_epoch=self._reset_every_epochs,
                                                      factor=0.8, min_lr=[1E-7])

            self.loss_ys = torch.nn.BCELoss()  # 0 is supervised
            self.loss_yu = torch.nn.BCELoss()  # 1 is unsupervised
            if self._use_cuda:
                self.loss_ys = self.loss_ys.cuda()
                self.loss_yu = self.loss_yu.cuda()

            zeros = torch.zeros(self._mini_batch_size)
            ones = torch.ones(self._mini_batch_size)
            self.ys_true = Variable(torch.transpose(torch.stack([ones, zeros]), 0, 1), requires_grad=False)
            self.yu_true = Variable(torch.transpose(torch.stack([zeros, ones]), 0, 1), requires_grad=False)
            uncertain_values = torch.zeros(self._mini_batch_size, 2)
            uncertain_values.fill_(0.5)
            self.ys_uncertain = Variable(uncertain_values, requires_grad=False)

            if self._use_cuda:
                self.ys_true = self.ys_true.cuda()
                self.yu_true = self.yu_true.cuda()
                self.ys_uncertain = self.ys_uncertain.cuda()

    def clean_outputs(self):
        self._my_feature_extractor1.clear_outputs()
        self._my_feature_extractor2.clear_outputs()

    def install_scheduler(self):
        if not self.do_not_use_scheduler:
            self._use_scheduler = True

    def estimate_example_weights(self, xs, weight_u=None):
        """This method estimates a weight for each example in the xs minibatch. The weight
        is derived from the ureg model's ability to tell if a given sample looks more like
        a training sample or an unsupervised sample. Samples that look more like an unsupervised
        sample are given larger weights. """

        if not self._enabled:
            return

        self.create_which_one_model(xs)
        supervised_output = self.extract_activations(xs)

        # self._which_one_model.eval()
        ys = self._which_one_model(supervised_output)
        print("probabilities that samples belong to the training and test sets:" + str(ys))

        prob_from_unsupset = ys.narrow(1, 1, 1)
        # print("probabilities that samples belong to test set:"+str(prob_from_unsupset))

    def train_ureg(self, xs, xu, weight_s=None, weight_u=None):
        if not self._enabled:
            return

        if len(xu) != len(xs):
            xu, xs = self._adjust_batch_sizes(xs, xu)

        self.create_which_one_model(xs)
        self.model_assembler.model.eval()
        supervised_output_list = self.extract_activation_list(xs)
        unsupervised_output_list = self.extract_activation_list(xu)
        self._which_one_model.train()
        self._optimizer.zero_grad()
        # predict which dataset (s or u) the samples were from:

        ys = self.model_assembler.evaluate(supervised_output_list)
        yu = self.model_assembler.evaluate(unsupervised_output_list)
        if (len(ys) != len(self.ys_true)):
            print("lengths ys differ: {} !={}".format(len(ys), len(self.ys_true)))
            return None
        if (len(yu) != len(self.yu_true)):
            print("lengths yu differ: {} !={}".format(len(yu), len(self.yu_true)))
            return None
        # print("ys: {} yu: {}".format(ys.data,yu.data))
        # derive the loss of binary classifications:

        # step the whichOne model's parameters in the direction that
        # reduces the loss:

        weight_s, weight_u =(1.,1.)
        loss = torch.nn.BCELoss()
        if self._use_cuda: loss = loss.cuda()

        loss_ys = loss(ys, self.ys_true)
        loss_yu = loss(yu, self.yu_true)
        total_which_model_loss = (weight_s * loss_ys + weight_u * loss_yu)
        # print("loss_ys: {} loss_yu: {} ".format(loss_ys.data[0],loss_yu.data[0]))
        # total_which_model_loss =torch.max(loss_ys,loss_yu)
        self._accumulator_total_which_model_loss += total_which_model_loss.data[0] / self._mini_batch_size
        self._num_accumulator_updates += 1
        total_which_model_loss.backward()
        self._optimizer.step()
        # updates counts for estimation of accuracy in this minibatch:

        # Estimate accuracy on the unup set only (less likely to see these examples often then
        # those of the training set.
        self._estimate_accuracy(yu, self.yu_true)

        return total_which_model_loss

    def get_ureg_loss(self):
        """Return the average ureg _which_one_model training loss since the new_epoch method was called.
        This average loss is """
        if self._enabled:
            assert self._num_accumulator_updates > 0, "accumulator was not updated, check that you called train_ureg after new_epoch"
            return self._accumulator_total_which_model_loss / self._num_accumulator_updates
        else:
            return 0

    def train_ureg_to_convergence(self, problem, train_dataset, unsup_dataset,
                                  performance_estimators=None,
                                  epsilon=0.01,
                                  max_epochs=30, max_examples=None):
        """Train the ureg model for a number of epochs until improvements in the loss
        are minor.
        :param supervised_loader loader for supervised examples.
        :param unsupervised_loader loader for unsupervised examples.
        :param max_epochs maximum number of epochs before stopping
        :param epsilon used to determine convergence.
        :param max_examples maximum number of examples to scan per epoch.
        :return list of performance estimators
        """
        if performance_estimators is None:
            performance_estimators = PerformanceList()
            performance_estimators += [LossHelper("ureg_loss"), FloatHelper("ureg_accuracy")]
        len_supervised = len(train_dataset)
        len_unsupervised = len(unsup_dataset)
        print("Training ureg to convergence with {} training and {} unsupervised samples,"
              " using at most {} shuffled combinations of examples per training epoch".format(
            len_supervised * self._mini_batch_size,
            len_unsupervised * self._mini_batch_size, max_examples))
        self._adjust_learning_rate(self._learning_rate)
        previous_average_loss = sys.maxsize
        for ureg_epoch in range(0, max_epochs):
            # reset metric at each ureg training epoch (we use the loss average as stopping condition):
            for performance_estimator in performance_estimators:
                performance_estimator.init_performance_metrics()

            from itertools import cycle
            length = 0

            if len_supervised < len_unsupervised:

                supervised_iter = iter(cycle(self.shuffling_iter(problem, train_dataset)))
                length = len_unsupervised
            else:
                supervised_iter = iter(self.shuffling_iter(problem, train_dataset))
                length = len_supervised

            if len_unsupervised < len_supervised:
                unsupervised_iter = iter(cycle(self.shuffling_iter(problem, train_dataset)))
            else:
                unsupervised_iter = iter(self.shuffling_iter(problem, unsup_dataset))
            if max_examples is None:
                max_examples = length * self._mini_batch_size
            length = max_examples / self._mini_batch_size

            num_batches = 0

            for (batch_idx, ((s_input, s_labels), (u_input, _))) in enumerate(zip(supervised_iter, unsupervised_iter)):

                xs = Variable(s_input)
                xu = Variable(u_input)
                if self._use_cuda:
                    xs = xs.cuda()
                    xu = xu.cuda()

                weight_s, weight_u = self.loss_weights(None, None)
                loss = self.train_ureg(xs, xu, weight_s, weight_u)
                if loss is not None:
                    # print("ureg batch {} average loss={} ".format(batch_idx, loss.data[0]))
                    num_batches += 1

                    performance_estimators.set_metric_with_outputs(batch_idx, "ureg_loss", loss.data[0], None, None)
                    performance_estimators.set_metric(batch_idx, "ureg_accuracy", self.ureg_accuracy())

                epoch_ = "epoch " + str(ureg_epoch) + " "
                progress_bar(batch_idx * self._mini_batch_size, max_examples,
                                 epoch_ + " ".join(
                                     [performance_estimator.progress_message() for performance_estimator in
                                      performance_estimators]))

                if ((batch_idx + 1) * self._mini_batch_size > max_examples):
                    break
            average_loss = performance_estimators[0].estimates_of_metric()[0]
            # print("ureg epoch {} average loss={} ".format(ureg_epoch, average_loss))
            if average_loss > previous_average_loss:
                if self._scheduler is not None:
                    self.schedule(epoch=ureg_epoch, val_loss=average_loss)
                else:
                    break
            if average_loss < previous_average_loss and abs(average_loss - previous_average_loss) < epsilon:
                break

            previous_average_loss = average_loss

        return performance_estimators

    def extract_activations(self, features):
        unsupervised_outputs = self.extract_activation_list(features)
        unsupervised_output = torch.cat(unsupervised_outputs, dim=1)

        # obtain activations for supervised samples:
        if self._use_cuda:
            unsupervised_output = unsupervised_output.cuda()

        return unsupervised_output

    def extract_activation_list(self, features):
        # determine the number of activations in model:
        self.create_feature_extractors()
        # obtain activations for unsupervised samples:
        self._my_feature_extractor1.register()
        self._my_feature_extractor1.clear_outputs()
        collect_output = self.model_assembler.get_collect_output()
        unsupervised_outputs = self._my_feature_extractor1.collect_outputs(features,
                                                                           collect_output)
        self._my_feature_extractor1.cleanup()
        self._my_feature_extractor1.clear_outputs()

        return unsupervised_outputs

    def regularization_loss(self, xs, xu, weight_s=None, weight_u=None):
        """
        Calculates the regularization loss and add it to
        the provided loss (linear combination using  alpha as mixing factor).

        :param loss: loss to be optimized without regularization
        :param xs: supervised features.
        :param xu: unsupervised features.
        :return: a tuple with (loss (Variable), supervised_loss (float), regularizationLoss (float))
        """
        if not self._enabled:
            return None
        if len(xu) != len(xs):
            xu, xs = self._adjust_batch_sizes(xs, xu)

        self.create_which_one_model(xs)

        supervised_output = self.extract_activation_list(xs)  # print("length of output: "+str(len(supervised_output)))
        unsupervised_output = self.extract_activation_list( xu)  # print("length of output: "+str(len(supervised_output)))

        # now we use the model:
        self._which_one_model.eval()
        # the more the activations on the supervised set deviate from the unsupervised data,
        # the more we need to regularize:
        ys = self.model_assembler.evaluate(supervised_output)
        yu = self.model_assembler.evaluate(unsupervised_output)

        weight_s, weight_u = self.loss_weights(weight_s, weight_u)

        # self.loss_ys.weight=torch.from_numpy(numpy.array([weight_s,weight_u]))
        # self.loss_yu.weight=torch.from_numpy(numpy.array([weight_u,weight_s]))

        rLoss = weight_s * self.loss_ys(ys, self.ys_uncertain) + \
                weight_u * self.loss_yu(yu, self.ys_uncertain)
        # self._alpha = 0.5 - (0.5 - self._last_epoch_accuracy)
        # rLoss = (self.loss_ys(ys, self.ys_uncertain))
        # self.loss_yu(yu, self.ys_uncertain)) / 2
        # return the regularization loss:
        return rLoss

    def regularization_loss_unsup_similarity(self, xs):
        """
        Calculates the regularization loss whose minimum indicates maximum similarity to the
        unsupervised set.

        :param loss: loss to be optimized without regularization
        :param xs: supervised features.
        :param xu: unsupervised features.
        :return: a variable with the regularization loss.
        """
        if not self._enabled:
            return None

        self.create_which_one_model(xs)
        if len(self.yu_true) != len(xs):
            return None

        supervised_output = self.extract_activation_list(xs)  # print("length of output: "+str(len(supervised_output)))

        # now we use the model:
        self._which_one_model.eval()
        # the more the activations on the supervised set deviate from the unsupervised data,
        # the more we need to regularize:
        ys = self.model_assembler.evaluate(supervised_output)

        # rLoss is zero when the ureg model predicts the training example is part of the unsupervised set
        loss_yu = torch.nn.BCELoss()
        if self._use_cuda: loss_yu=loss_yu.cuda()

        rLoss = loss_yu(ys, self.yu_true)

        # self._alpha = 0.5 - (0.5 - self._last_epoch_accuracy)
        # rLoss = (self.loss_ys(ys, self.ys_uncertain))
        # self.loss_yu(yu, self.ys_uncertain)) / 2
        # return the regularization loss:
        return rLoss

    def loss_weights(self, weight_s, weight_u):
        if weight_s is None:
            weight_s = 1 / (self.num_training / (self.num_training + self.num_unsupervised_examples))
        if weight_u is None:
            weight_u = 1 / (self.num_unsupervised_examples / (self.num_training + self.num_unsupervised_examples))

        normalized_weight_s = weight_s / (weight_s + weight_u)
        normalized_weight_u = weight_u / (weight_s + weight_u)
        # print("using weights: T={} U={} weight_s={} weight_u={}".format(
        #     self.num_training,
        #     self.num_unsupervised_examples,
        #     normalized_weight_s, normalized_weight_u))
        return normalized_weight_s, normalized_weight_u

    def combine_losses(self, supervised_loss, regularization_loss):
        return supervised_loss * (1 - self._alpha) + self._alpha * regularization_loss

    def get_which_one_model(self):
        return self._which_one_model

    def resume(self, saved_model):
        self.checkpoint_model = saved_model

    def new_epoch(self, epoch):
        self._n_total = 0
        self._n_correct = 0
        # print("epoch {0} which_one_model loss: {1:.4f}"
        #      .format(epoch,
        #              self._accumulator_total_which_model_loss))

        self._accumulator_total_which_model_loss = 0
        self._num_accumulator_updates = 0
        if self._forget_every_n_epoch is not None:
            self._epoch_counter += 1
            # halve the learning rate for each extra epoch we don't reset:
            self._adjust_learning_rate(self._learning_rate / (pow(2, self._epoch_counter)))
            if self._epoch_counter > self._forget_every_n_epoch:
                # reset the learning rate of the which_one_model:
                self._epoch_counter = 0
                self._adjust_learning_rate(self._learning_rate)

    def _adjust_learning_rate(self, learning_rate):
        """Set learning rate of which_one_model to the parameter. """
        if self._optimizer is not None:
            self.adjust_learning_rate(self._optimizer, learning_rate, self._eps)

    def adjust_learning_rate(self, optimizer, learning_rate, epsilon=1E-8):
        """Set learning rate of which_one_model to the parameter. """
        if optimizer is not None:
            for i, param_group in enumerate(optimizer.param_groups):
                old_lr = float(param_group['lr'])
                new_lr = learning_rate
                if abs(old_lr - new_lr) > epsilon:
                    param_group['lr'] = new_lr
                    print('Adjusting learning rate to {:.4e}'
                          .format(new_lr))

    def set_num_examples(self, num_training, num_unsupervised_examples):
        self.num_training = num_training
        self.num_unsupervised_examples = num_unsupervised_examples

    def _adjust_batch_sizes(self, xs, xu):
        """
        Adjust the batch size of the smaller batch to make them compatible. Sample examples with resampling to make the smaller
        batch as large as the other one.
        :param xs: a minibatch of training data.
        :param xu: a minibatch of unsupervised examples.
        :return: xs, xu adjusted for size
        """
        assert len(xu) != len(xs), "batch size must not be equal"
        max_len = max(len(xs), len(xu))
        if (len(xs) > len(xu)):
            bag = [xu[index] for index in range(len(xu))]
            xu = torch.stack(random.choices(bag, k=max_len))
        else:
            bag = [xs[index] for index in range(len(xs))]
            xs = torch.stack(random.choices(bag, k=max_len))
        return xs, xu

    def shuffling_iter(self, problem, dataset):

        return problem.loader_for_dataset(dataset)
