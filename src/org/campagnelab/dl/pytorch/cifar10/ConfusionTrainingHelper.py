import os
import torch
from torch.autograd import Variable
from torch.nn import CrossEntropyLoss
from torch.optim.lr_scheduler import ReduceLROnPlateau

from org.campagnelab.dl.pytorch.cifar10.FloatHelper import FloatHelper
from org.campagnelab.dl.pytorch.cifar10.LossHelper import LossHelper
from org.campagnelab.dl.pytorch.cifar10.PerformanceList import PerformanceList
from org.campagnelab.dl.pytorch.cifar10.Problems import create_model
from org.campagnelab.dl.pytorch.cifar10.confusion.ConfusionModel import ConfusionModel
from org.campagnelab.dl.pytorch.cifar10.utils import batch, progress_bar


def class_label(num_classes, predicted_index, true_index):
    return predicted_index * num_classes + true_index

class ConfusionTrainingHelper:

    def __init__(self, model, problem, args, use_cuda, checkpoint_key=None):

        self.use_cuda = use_cuda
        self.problem = problem
        self.args = args
        if checkpoint_key is not None:
            self.model=self.load_confusion_model( self.args.checkpoint_key)
            if use_cuda:
                self.model.cuda()

        else:
            self.model = ConfusionModel(model, problem)
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=args.lr, momentum=0.9,
                                             weight_decay=args.L2)
            self.lr_scheduler = ReduceLROnPlateau(self.optimizer, "min", factor=0.1,
                                                   patience=5,
                                                   verbose=True)
            self.criterion = CrossEntropyLoss()
            if use_cuda:
                self.model.cuda()
                self.criterion = self.criterion.cuda()

    def train(self, epoch, confusion_data):
        args=self.args
        optimizer=self.optimizer
        problem=self.problem
        performance_estimators = PerformanceList()
        performance_estimators += [FloatHelper("train_loss")]

        for performance_estimator in performance_estimators:
            performance_estimator.init_performance_metrics()
        self.model.train()
        for batch_idx, confusion_list in enumerate(batch(confusion_data, args.mini_batch_size)):
            batch_size = min(len(confusion_list), args.mini_batch_size)
            images = [None] * batch_size
            targets = torch.zeros(batch_size)

            optimizer.zero_grad()
            training_loss_input = torch.zeros(batch_size, 1)
            trained_with_input = torch.zeros(batch_size, 1)

            for index, confusion in enumerate(confusion_list):

                num_classes = problem.num_classes()
                targets[index]=class_label(num_classes,
                                           confusion.predicted_label,confusion.true_label)
                dataset=problem.train_set() if confusion.trained_with  else problem.test_set()
                images[index], _ = dataset[confusion.example_index]

                training_loss_input[index]=confusion.train_loss
                trained_with_input[index]=1.0 if confusion.trained_with else 0.0

            image_input = Variable(torch.stack(images, dim=0), requires_grad=True)
            training_loss_input = Variable(training_loss_input, requires_grad=True)
            trained_with_input = Variable(trained_with_input, requires_grad=True)
            targets = Variable(targets, requires_grad=False).type(torch.LongTensor)
            if self.use_cuda:
                image_input = image_input.cuda()
                training_loss_input = training_loss_input.cuda()
                trained_with_input = trained_with_input.cuda()
                targets = targets.cuda()

            outputs = self.model(training_loss_input, trained_with_input, image_input)
            loss = self.criterion(outputs, targets)

            loss.backward()
            optimizer.step()

            performance_estimators.set_metric(batch_idx, "train_loss", loss.data[0])
            #progress_bar(batch_idx * batch_size,
            #             len(confusion_data),
            #             " ".join([performance_estimator.progress_message() for performance_estimator in
            #                       performance_estimators]))
        return performance_estimators

    def test(self, epoch, confusion_data):
        args = self.args
        problem = self.problem
        performance_estimators = PerformanceList()
        performance_estimators += [FloatHelper("test_loss")]
        self.model.eval()
        for performance_estimator in performance_estimators:
            performance_estimator.init_performance_metrics()

        for batch_idx, confusion_list in enumerate(batch(confusion_data, args.mini_batch_size)):
            batch_size = min(len(confusion_list), args.mini_batch_size)
            images = [None] * batch_size
            targets = torch.zeros(batch_size)
            training_loss_input = torch.zeros(batch_size, 1)
            trained_with_input = torch.zeros(batch_size, 1)

            for index, confusion in enumerate(confusion_list):
                num_classes = problem.num_classes()
                targets[index] = class_label(num_classes,
                                             confusion.predicted_label, confusion.true_label)
                dataset = problem.train_set() if confusion.trained_with else problem.test_set()
                images[index], _ = dataset[confusion.example_index]

                training_loss_input[index] = confusion.train_loss
                trained_with_input[index] = 1.0 if confusion.trained_with else 0.0

            image_input = Variable(torch.stack(images, dim=0), requires_grad=True)
            training_loss_input = Variable(training_loss_input, requires_grad=True)
            trained_with_input = Variable(trained_with_input, requires_grad=True)
            targets = Variable(targets, requires_grad=False).type(torch.LongTensor)
            if self.use_cuda:
                image_input = image_input.cuda()
                training_loss_input = training_loss_input.cuda()
                trained_with_input = trained_with_input.cuda()
                targets = targets.cuda()

            outputs = self.model(training_loss_input, trained_with_input, image_input)
            loss = self.criterion(outputs, targets)

            performance_estimators.set_metric(batch_idx, "test_loss", loss.data[0])
            #progress_bar(batch_idx * batch_size,
            #             len(confusion_data),
            #             " ".join([performance_estimator.progress_message() for performance_estimator in
            #                       performance_estimators]))
        self.lr_scheduler.step(performance_estimators.get_metric("test_loss"), epoch=epoch)
        return performance_estimators


    def predict(self, training_losses):
        args = self.args
        problem = self.problem
        performance_estimators = PerformanceList()
        performance_estimators += [FloatHelper("test_loss")]
        self.model.eval()
        for performance_estimator in performance_estimators:
            performance_estimator.init_performance_metrics()

        for batch_idx, tensor_images in enumerate(batch(problem.unsup_set(), args.mini_batch_size)):
            batch_size = min(len(tensor_images), args.mini_batch_size)
            targets = torch.zeros(batch_size)
            training_loss_input = torch.zeros(batch_size, 1)
            trained_with_input = torch.zeros(batch_size, 1)

            for index, training_loss in enumerate(training_losses):
                num_classes = problem.num_classes()
                training_loss_input[index] = training_loss
                trained_with_input[index] =0 # we are predicting on a set never seen by the model

            image_input = Variable(tensor_images, requires_grad=False)
            training_loss_input = Variable(training_loss_input, requires_grad=False)
            trained_with_input = Variable(trained_with_input, requires_grad=False)
            targets = Variable(targets, requires_grad=False).type(torch.LongTensor)
            if self.use_cuda:
                image_input = image_input.cuda()
                training_loss_input = training_loss_input.cuda()
                trained_with_input = trained_with_input.cuda()
                targets = targets.cuda()

            outputs = self.model(training_loss_input, trained_with_input, image_input)
            print(outputs.data)
            #progress_bar(batch_idx * batch_size,
            #             len(confusion_data),
            #             " ".join([performance_estimator.progress_message() for performance_estimator in
            #                       performance_estimators]))

        return performance_estimators

    def load_confusion_model(self, checkpoint_key):
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        state = torch.load('./checkpoint/ckpt_{}.t7'.format(self.args.checkpoint_key))
        model = state['confusion-model']
        model.cpu()
        model.eval()
        return model

    def save_confusion_model(self, epoch, test_loss, training_losses):

        # Save checkpoint.

        #print('Saving confusion model..')
        model = self.model
        model.eval()

        state = {
            'confusion-model': self.model,
            'test_loss': test_loss,
            'epoch': epoch,
            'training_losses': training_losses
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/confusionmodel_{}.t7'.format(self.args.checkpoint_key))
