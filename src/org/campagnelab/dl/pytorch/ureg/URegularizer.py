import random
import sys

import torch
from torch.autograd import Variable
from torch.nn import Sequential
from torch.optim.lr_scheduler import ReduceLROnPlateau

from org.campagnelab.dl.pytorch.cifar10.LossHelper import LossHelper
from org.campagnelab.dl.pytorch.cifar10.utils import init_params, progress_bar
from org.campagnelab.dl.pytorch.ureg.MyFeatureExtractor import MyFeatureExtractor


class URegularizer:
    """
    Unsupervised regularizer. This class estimates a regularization term using
    unlabeled examples and how similar the training examples are to the unlabeled
    ones. This term can be added to the supervised loss to avoid over-fitting to
    the training set.
    """

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

    def estimate_accuracy(self, xs):
        supervised_output = self.extract_activations(xs)  # print("length of output: "+str(len(supervised_output)))
        num_activations_supervised = supervised_output.size()[1]

        self.create_which_one_model(num_activations_supervised)

        # now we use the model:
        self._which_one_model.eval()
        # the more the activations on the supervised set deviate from the unsupervised data,
        # the more we need to regularize:
        ys = self._which_one_model(supervised_output)
        # yu = self._which_one_model(unsupervised_output)

        self._estimate_accuracy(ys, self.ys_true)

    def ureg_accuracy(self):
        if self._n_total == 0:
            return float("nan")
        accuracy = self._n_correct / self._n_total
        print("ureg accuracy={0:.3f} correct: {1}/{2}".format(accuracy, self._n_correct, self._n_total))
        self._last_epoch_accuracy = accuracy
        self._n_total = 0
        self._n_correct = 0
        return accuracy

    def __init__(self, model, mini_batch_size, num_features=64, alpha=0.5, learning_rate=0.1):
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
        self.num_training = 0
        self.num_unsupervised_examples = 0
        self._use_scheduler=False
        # def count_activations(i, o):
        #     self.add_activations(len(o))
        #
        # removeHandles = []
        # for namedModule in model.modules():
        #     print(namedModule)
        #     removeHandles.append( namedModule.register_forward_hook(count_activations))
        #
        # model(torch.zeros(num_inputs))
        # for handle in removeHandles:
        #     handle.remove()

        num_input_features = self._num_activations
        self._alpha = alpha
        # define the model tasked with predicting if activations are generated from
        # a sample in the training or unsupervised set:

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

    def create_which_one_model(self, num_activations):
        num_features = self._num_features
        if self._which_one_model is None:
            # if (self._checkpointModel):
            #     self._which_one_model = self._checkpointModel
            #     self._checkpointModel = None
            # else:
            self._which_one_model = Sequential(
                torch.nn.Linear(num_activations, num_features),
                #torch.nn.Dropout(0.5),
                torch.nn.ReLU(),
                #torch.nn.Dropout(0.5),
                torch.nn.Linear(num_features, num_features),
                #torch.nn.Dropout(0.5),
                torch.nn.ReLU(),
                torch.nn.Linear(num_features, 2),
                torch.nn.Softmax()
            )
            # (we will use BCEWithLogitsLoss on top of linar to get the cross entropy after
            # applying a sigmoid).
            init_params(self._which_one_model)

            print("done building which_one_model:" + str(self._which_one_model))
            if self._use_cuda: self._which_one_model.cuda()
            # self.optimizer = torch.optim.Adam(self.which_one_model.parameters(),
            #                                   lr=self.learning_rate)
            self._optimizer = torch.optim.SGD(self._which_one_model.parameters(), lr=self._learning_rate, momentum=0.9,
                                              weight_decay=0.01);
            if self._use_scheduler:
                self._scheduler = ReduceLROnPlateau(self._optimizer, 'min', factor=0.5, patience=0, verbose=True)

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

            for index in range(self._mini_batch_size):
                self.yu_true[index] = 1
            if self._use_cuda:
                self.ys_true = self.ys_true.cuda()
                self.yu_true = self.yu_true.cuda()
                self.ys_uncertain = self.ys_uncertain.cuda()

    def clean_outputs(self):
        self._my_feature_extractor1.clear_outputs()
        self._my_feature_extractor2.clear_outputs()

    def install_scheduler(self):

        self._use_scheduler=True

    def train_ureg(self, xs, xu, weight_s=None, weight_u=None):
        if not self._enabled:
            return

        if len(xu) != len(xs):
            xu, xs = self._adjust_batch_sizes(xs, xu)

        mini_batch_size = len(xs)

        supervised_output = self.extract_activations(xs)  # print("length of output: "+str(len(supervised_output)))
        unsupervised_output = self.extract_activations(xu)  # print("length of output: "+str(len(supervised_output)))
        num_activations_supervised = supervised_output.size()[1]
        num_activations_unsupervised = unsupervised_output.size()[1]
        # print("num_activations: {}".format(str(num_activations_supervised)))
        # print("num_activations: {}".format(str(num_activations_unsupervised)))

        # supervised_output = Variable(supervised_output.data, requires_grad = True)
        # print("has gradient: " + str(hasattr(supervised_output.grad, "data")))  # prints False


        self.create_which_one_model(num_activations_supervised)
        self._optimizer.zero_grad()
        # predict which dataset (s or u) the samples were from:
        self._which_one_model.train()
        ys = self._which_one_model(supervised_output)
        yu = self._which_one_model(unsupervised_output)
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
        weight_s, weight_u = self.loss_weights(weight_s, weight_u)

        loss_ys = self.loss_ys(ys, self.ys_true)
        loss_yu = self.loss_yu(yu, self.yu_true)
        total_which_model_loss = (weight_s * loss_ys + weight_u * loss_yu)
        # print("loss_ys: {} loss_yu: {} ".format(loss_ys.data[0],loss_yu.data[0]))
        # total_which_model_loss =torch.max(loss_ys,loss_yu)
        self._accumulator_total_which_model_loss += total_which_model_loss.data[0] / self._mini_batch_size

        total_which_model_loss.backward(retain_graph=True)
        self._optimizer.step()
        return total_which_model_loss

    def train_ureg_to_convergence(self, supervised_loader, unsupervised_loader,
                                  performance_estimators=(LossHelper("ureg_loss"),),
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
        len_supervised = len(supervised_loader)
        len_unsupervised = len(unsupervised_loader)
        print("Training ureg to convergence with {} training and {} unsupervised samples,"
              " using at most {} shuffled combinations of examples per training epoch".format(
            len_supervised*self._mini_batch_size,
            len_unsupervised*self._mini_batch_size,max_examples))
        self._adjust_learning_rate(self._learning_rate)
        previous_average_loss=sys.maxsize
        for ureg_epoch in range(0, max_epochs):
            # reset metric at each ureg training epoch (we use the loss average as stopping condition):
            for performance_estimator in performance_estimators:
                performance_estimator.init_performance_metrics()

            from itertools import cycle
            length=0
            if len_supervised < len_unsupervised:
                supervised_iter = iter(cycle(supervised_loader))
                length=len_unsupervised
            else:
                supervised_iter = iter(supervised_loader)
                length=len_supervised

            if len_unsupervised < len_supervised:
                unsupervised_iter = iter(cycle(unsupervised_loader))
            else:
                unsupervised_iter = iter(unsupervised_loader)
            if max_examples is None:
                max_examples=length*self._mini_batch_size
            num_batches=0

            for (batch_idx, ((s_input, s_labels), (u_input, _))) in enumerate(zip(supervised_iter, unsupervised_iter)):

                xs = Variable(s_input)
                xu = Variable(u_input)
                if self._use_cuda:
                    xs=xs.cuda()
                    xu=xu.cuda()
                loss=self.train_ureg(xs, xu)
                if loss is not None:
                    #print("ureg batch {} average loss={} ".format(batch_idx, loss.data[0]))
                    num_batches+=1
                    for performance_estimator in performance_estimators:
                        performance_estimator.observe_performance_metric(batch_idx, loss.data[0],
                                                                         None,
                                                                         None)
                    epoch_ = "epoch " + str(ureg_epoch) + " "
                    progress_bar(batch_idx, length,
                                 epoch_ + " ".join([performance_estimator.progress_message() for performance_estimator in
                                       performance_estimators]))

                if (batch_idx*self._mini_batch_size>max_examples):
                    break
            average_loss=performance_estimators[0].estimates_of_metric()[0]
            #print("ureg epoch {} average loss={} ".format(ureg_epoch, average_loss))
            if average_loss > previous_average_loss:
                if self._scheduler is not None:
                    self.schedule(epoch=ureg_epoch,val_loss=average_loss)
                else:
                    break
            if average_loss < previous_average_loss and abs(average_loss-previous_average_loss)<epsilon:
                break

            previous_average_loss=average_loss

        return performance_estimators

    def extract_activations(self, features):
        # determine the number of activations in model:
        self.create_feature_extractors()
        # obtain activations for unsupervised samples:
        self._my_feature_extractor1.register()
        self._my_feature_extractor1.clear_outputs()
        unsupervised_output = self._my_feature_extractor1.collect_outputs(features, [])
        self._my_feature_extractor1.cleanup()
        self._my_feature_extractor1.clear_outputs()
        unsupervised_output = torch.cat(unsupervised_output, dim=1)

        # obtain activations for supervised samples:
        if self._use_cuda:
            unsupervised_output = unsupervised_output.cuda()

        return unsupervised_output

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

        supervised_output = self.extract_activations(xs)  # print("length of output: "+str(len(supervised_output)))
        unsupervised_output = self.extract_activations(xu)  # print("length of output: "+str(len(supervised_output)))
        num_activations_supervised = supervised_output.size()[1]

        self.create_which_one_model(num_activations_supervised)

        # now we use the model:
        self._which_one_model.eval()
        # the more the activations on the supervised set deviate from the unsupervised data,
        # the more we need to regularize:
        ys = self._which_one_model(supervised_output)
        yu = self._which_one_model(unsupervised_output)

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

    def loss_weights(self, weight_s, weight_u):
        if weight_s is None:
            weight_s = 1 / (self.num_training / (self.num_training + self.num_unsupervised_examples))
        if weight_u is None:
            weight_u = 1 / (self.num_unsupervised_examples / (self.num_training + self.num_unsupervised_examples))
        # print("using weights: T={} U={} weight_s={} weight_u={}".format(
        #     self.num_training,
        #     self.num_unsupervised_examples,
        #     weight_s, weight_u))
        return weight_s, weight_u

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

    def adjust_learning_rate( self, optimizer, learning_rate, epsilon=1e-6):
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
