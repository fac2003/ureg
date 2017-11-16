import torch
from torch.autograd import Variable
from torch.nn import Sequential
from torch import cat
from torch.optim.lr_scheduler import ReduceLROnPlateau

from org.campagnelab.dl.pytorch.cifar10.utils import init_params
from org.campagnelab.dl.pytorch.ureg.MyFeatureExtractor import MyFeatureExtractor


class URegularizer:
    """
    Unsupervised regularizer. This class estimates a regularization term using
    unlabeled examples and how similar the training examples are to the unlabeled
    ones. This term can be added to the supervised loss to avoid over-fitting to
    the training set.
    """

    def add_activations(self, num):
        self.num_activations += num
        # print("activations: " + str(self.num_activations))

    def __init__(self, model, mini_batch_size, num_features=64, alpha=0.5, learning_rate=0.1):
        self.mini_batch_size = mini_batch_size
        self.model = model
        self.num_activations = 0
        self.learning_rate = learning_rate
        self.which_one_model = None
        self.my_feature_extractor1 = None
        self.my_feature_extractor2 = None
        self.enabled = True
        self.use_cuda = torch.cuda.is_available()
        self.scheduler = None
        self.checkpointModel = None
        self.num_features = num_features
        self.epoch_counter=0
        self._forget_every_n_epoch =None
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

        num_input_features = self.num_activations
        self.alpha = alpha
        # define the model tasked with predicting if activations are generated from
        # a sample in the training or unsupervised set:

    def forget_model(self, N_epoch):
        """
        Reset the weights of the which_one_model model after N epoch of training. We need to reset once in a while
         to focus on the recent history of activations, not remember the distant past.
        :param N_epoch:
        :return:
        """
        self._forget_every_n_epoch=N_epoch

    def disable(self):
        self.enabled = False

    def enable(self):
        self.enabled = True

    def create_feature_extractors(self):
        if self.my_feature_extractor1 is None or self.my_feature_extractor2 is None:
            self.my_feature_extractor1 = MyFeatureExtractor(self.model, None)
            self.my_feature_extractor2 = MyFeatureExtractor(self.model, None)

    def schedule(self, val_loss, epoch):
        """
        Apply the learning rate schedule
        :param epoch: current epoch number
        :param val_loss: the validation/test loss
        :return: None
        """
        if self.scheduler is not None:
            self.scheduler.step(val_loss, epoch)

    def create_which_one_model(self, num_activations):
        num_features = self.num_features
        if self.which_one_model is None:

            if self.checkpointModel is not None:
                self.which_one_model = self.checkpointModel
            else:
                self.which_one_model = Sequential(
                    torch.nn.Dropout(p=0.5),
                    torch.nn.Linear(num_activations, num_features),
                    torch.nn.Dropout(p=0.5),
                    torch.nn.ReLU(),
                    torch.nn.Linear(num_features, num_features),
                    torch.nn.Dropout(p=0.5),
                    torch.nn.ReLU(),
                    torch.nn.Linear(num_features, 2),
                    torch.nn.Dropout(p=0.5),
                    torch.nn.ReLU(),
                    torch.nn.Softmax())
            print("done building which_one_model:" + str(self.which_one_model))
            if self.use_cuda: self.which_one_model.cuda()
            # self.optimizer = torch.optim.Adam(self.which_one_model.parameters(),
            #                                   lr=self.learning_rate)
            self.optimizer = torch.optim.SGD(self.which_one_model.parameters(), lr=0.1, momentum=0.9);
            self.scheduler = ReduceLROnPlateau(self.optimizer, 'min', factor=0.5, patience=0, verbose=True)
            self.loss_ys = torch.nn.CrossEntropyLoss()  # 0 is supervised
            self.loss_yu = torch.nn.CrossEntropyLoss()  # (yu, torch.ones(mini_batch_size))  # 1 is unsupervised
            if self.use_cuda:
                self.loss_ys = self.loss_ys.cuda()
                self.loss_yu = self.loss_yu.cuda()

            self.ys_true = Variable(torch.LongTensor([0] * self.mini_batch_size), requires_grad=False)
            self.yu_true = Variable(torch.LongTensor([1] * self.mini_batch_size), requires_grad=False)
            for index in range(self.mini_batch_size):
                self.yu_true[index] = 1
            if self.use_cuda:
                self.ys_true = self.ys_true.cuda()
                self.yu_true = self.yu_true.cuda()

    def clean_outputs(self):
        self.my_feature_extractor1.clear_outputs()
        self.my_feature_extractor2.clear_outputs()

    def regularization_loss(self, loss, xs, xu):
        if not self.enabled:
            return (loss, loss.data[0], 0.0)

        if len(xu) != len(xs):
            print("mismatch between inputs (sizes={} != {}), ignoring this regularization step"
                  .format(xs.size(), xu.size()))
            return (loss, loss.data[0], 0.0)

        mini_batch_size = len(xs)

        # determine the number of activations in model:
        self.create_feature_extractors()
        # obtain activations for unsupervised samples:
        self.my_feature_extractor1.register()
        self.my_feature_extractor1.clear_outputs()
        unsupervised_output = self.my_feature_extractor1.collect_outputs(xu, [])
        self.my_feature_extractor1.cleanup()
        self.my_feature_extractor1.clear_outputs()

        self.my_feature_extractor2.register()
        self.my_feature_extractor2.clear_outputs()
        supervised_output = self.my_feature_extractor2.collect_outputs(xs, [])
        self.my_feature_extractor2.cleanup()
        self.my_feature_extractor2.clear_outputs()

        # obtain activations for supervised samples:
        supervised_output = torch.cat(supervised_output, dim=1)
        unsupervised_output = torch.cat(unsupervised_output, dim=1)

        if self.use_cuda:
            supervised_output = supervised_output.cuda()
            unsupervised_output = unsupervised_output.cuda()

        # print("length of output: "+str(len(supervised_output)))
        num_activations_supervised = supervised_output.size()[1]
        num_activations_unsupervised = unsupervised_output.size()[1]
        # print("num_activations: {}".format(str(num_activations_supervised)))
        # print("num_activations: {}".format(str(num_activations_unsupervised)))

        # supervised_output = Variable(supervised_output.data, requires_grad = True)
        # print("has gradient: " + str(hasattr(supervised_output.grad, "data")))  # prints False


        self.create_which_one_model(num_activations_supervised)

        # predict which dataset (s or u) the samples were from:

        ys = self.which_one_model(supervised_output)
        yu = self.which_one_model(unsupervised_output)

        # derive the loss of binary classifications:

        # step the whichOne model's parameters in the direction that
        # reduces the loss:
        # zeroes = Variable(, requires_grad=False)

        total_which_model_loss = self.loss_ys(ys, self.ys_true) + \
                                 self.loss_yu(yu, self.yu_true)
        # print("total_which_model_loss: "+str(total_which_model_loss.data[0]))
        self.optimizer.zero_grad()
        total_which_model_loss.backward(retain_graph=True)
        self.optimizer.step()

        # the more the activations on the supervised set deviate from the unsupervised data,
        # the more we need to regularize:
        self.regularizationLoss = -self.loss_ys(ys, self.ys_true)
        #       self.regularizationLoss=0
        # print("loss {} regularizationLoss: {}".format(loss.data, str(self.regularizationLoss.data[0])))
        # return the output on the supervised sample:
        supervised_loss = loss
        loss = supervised_loss * (1 - self.alpha) + self.alpha * self.regularizationLoss
        # print("loss: {0:.2f} supervised: {1:.2f} unsupervised: {2:.2f} ".format(loss.data[0], supervised_loss.data[0], self.regularizationLoss.data[0]))
        return (loss, supervised_loss.data[0], self.regularizationLoss.data[0])

    def resume(self, saved_model):
        self.checkpoint_model = saved_model

    def new_epoch(self, epoch):

            if self._forget_every_n_epoch is not None:
                self.epoch_counter+=1
                if self.epoch_counter>self._forget_every_n_epoch:
                    # reset the weights of the which_one_model:
                    self.epoch_counter=0
                    init_params(self.which_one_model)
