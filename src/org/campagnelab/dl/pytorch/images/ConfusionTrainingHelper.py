import os
from queue import PriorityQueue

import sys
from random import shuffle

import torch
from torch.autograd import Variable
from torch.nn import CrossEntropyLoss
from torch.optim.lr_scheduler import ReduceLROnPlateau

from org.campagnelab.dl.pytorch.images.FloatHelper import FloatHelper
from org.campagnelab.dl.pytorch.images.LossHelper import LossHelper
from org.campagnelab.dl.pytorch.images.PerformanceList import PerformanceList
from org.campagnelab.dl.pytorch.images.PriorityQueues import PriorityQueues
from org.campagnelab.dl.pytorch.images.Problems import create_model
from org.campagnelab.dl.pytorch.images.confusion.ConfusionModel import ConfusionModel
from org.campagnelab.dl.pytorch.images.utils import batch, progress_bar


def class_label(num_classes, predicted_index, true_index):
    return predicted_index * num_classes + true_index


class ConfusionTrainingHelper:
    def __init__(self, model, problem, args, use_cuda, checkpoint_key=None):

        self.use_cuda = use_cuda
        self.problem = problem
        self.args = args
        if checkpoint_key is not None:
            self.model, self.training_losses, self.validation_losses = \
                self.load_confusion_model(self.args.checkpoint_key)
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
        args = self.args
        optimizer = self.optimizer
        problem = self.problem
        performance_estimators = PerformanceList()
        performance_estimators += [FloatHelper("train_loss")]

        for performance_estimator in performance_estimators:
            performance_estimator.init_performance_metrics()
        self.model.train()
        shuffle(confusion_data)
        max = min(args.max_training, len(confusion_data))
        confusion_data = confusion_data[0:max]
        for batch_idx, confusion_list in enumerate(batch(confusion_data, args.mini_batch_size)):
            batch_size = min(len(confusion_list), args.mini_batch_size)
            images = [None] * batch_size
            targets = torch.zeros(batch_size)

            optimizer.zero_grad()
            training_loss_input = torch.zeros(batch_size, 1)
            trained_with_input = torch.zeros(batch_size, 1)

            for index, confusion in enumerate(confusion_list):
                num_classes = problem.num_classes()
                targets[index] = class_label(num_classes,
                                             confusion.predicted_label, confusion.true_label)
                dataset = problem.train_set() if confusion.trained_with  else problem.test_set()
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

            loss.backward()
            optimizer.step()

            performance_estimators.set_metric(batch_idx, "train_loss", loss.data[0])
            if args.progress_bar:
                progress_bar(batch_idx * batch_size,
                             len(confusion_data),
                             " ".join([performance_estimator.progress_message() for performance_estimator in
                                       performance_estimators]))
        return performance_estimators

    def test(self, epoch, confusion_data):
        args = self.args
        problem = self.problem
        performance_estimators = PerformanceList()
        performance_estimators += [FloatHelper("test_loss")]
        self.model.eval()
        for performance_estimator in performance_estimators:
            performance_estimator.init_performance_metrics()

        shuffle(confusion_data)
        max = min(args.max_training, len(confusion_data))
        confusion_data = confusion_data[0:max]

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

            image_input = Variable(torch.stack(images, dim=0), volatile=True)
            training_loss_input = Variable(training_loss_input, volatile=True)
            trained_with_input = Variable(trained_with_input, volatile=True)
            targets = Variable(targets, volatile=True).type(torch.LongTensor)
            if self.use_cuda:
                image_input = image_input.cuda()
                training_loss_input = training_loss_input.cuda()
                trained_with_input = trained_with_input.cuda()
                targets = targets.cuda()

            outputs = self.model(training_loss_input, trained_with_input, image_input)
            loss = self.criterion(outputs, targets)

            performance_estimators.set_metric(batch_idx, "test_loss", loss.data[0])
            if args.progress_bar:
                progress_bar(batch_idx * batch_size,
                             len(confusion_data),
                             " ".join([performance_estimator.progress_message() for performance_estimator in
                                       performance_estimators]))
        self.lr_scheduler.step(performance_estimators.get_metric("test_loss"), epoch=epoch)
        return performance_estimators

    def predict(self, max_examples=sys.maxsize, max_queue_size=10):

        val_losses = self.validation_losses
        pq = PriorityQueues(val_losses, max_queue_size=max_queue_size)

        args = self.args
        problem = self.problem
        self.model.eval()
        decoder = {}
        num_classes = problem.num_classes()
        for i in range(num_classes):
            for j in range(num_classes):
                decoder[class_label(num_classes, i, j)] = (i, j)
        unsup_set_length = len(problem.unsup_set())
        image_index = 0
        for batch_idx, tensors in enumerate(batch(problem.unsup_set(), args.mini_batch_size)):

            batch_size = min(len(tensors), args.mini_batch_size)
            image_index = batch_idx * len(tensors)
            tensor_images = (torch.stack([ti for ti, _ in tensors], dim=0))
            image_input = Variable(torch.stack(tensor_images, dim=0), volatile=True)

            if self.use_cuda:
                image_input = image_input.cuda()

            for training_loss in val_losses:
                training_loss_input = torch.zeros(batch_size, 1)
                trained_with_input = torch.zeros(batch_size, 1)

                for index in range(batch_size):
                    training_loss_input[index] = training_loss
                    trained_with_input[index] = 0  # we are predicting on a set never seen by the model

                trained_with_input = Variable(trained_with_input, volatile=True)
                training_loss_input = Variable(training_loss_input, volatile=True)

                if self.use_cuda:

                    training_loss_input = training_loss_input.cuda()
                    trained_with_input = trained_with_input.cuda()

                outputs = self.model(training_loss_input, trained_with_input, image_input)
                max_values, indices = torch.max(outputs.data, dim=1)
                for index in range(batch_size):
                    (predicted_index, true_index) = decoder[indices[index]]
                    probability = max_values[index]
                    if predicted_index != true_index:
                        # off diagonal prediction, predicting an error on this image:
                        unsup_index = image_index + index
                        # print("training_loss={} probability={} predicted_index={}, true_index={} unsup_index={}".format(
                        #       training_loss, probability, predicted_index, true_index, unsup_index))
                        pq.put(training_loss, probability, (unsup_index, predicted_index, true_index))
            if args.progress_bar:
                progress_bar(batch_idx * batch_size,
                             unsup_set_length)

            if batch_idx * args.mini_batch_size > max_examples: break
        return pq

    def load_confusion_model(self, checkpoint_key):
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        state = torch.load('./checkpoint/confusionmodel_{}.t7'.format(self.args.checkpoint_key))
        model = state['confusion-model']
        training_losses = state['training_losses']
        validation_losses = state['validation_losses']
        model.cpu()
        model.eval()
        return (model, training_losses, validation_losses)

    def save_confusion_model(self, epoch, test_loss, training_losses, validation_losses):

        # Save checkpoint.

        # print('Saving confusion model..')
        model = self.model
        model.eval()

        state = {
            'confusion-model': self.model,
            'test_loss': test_loss,
            'epoch': epoch,
            'training_losses': training_losses,
            'validation_losses': validation_losses
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/confusionmodel_{}.t7'.format(self.args.checkpoint_key))
