import torch
from torch.autograd import Variable
from torch.nn import CrossEntropyLoss

from org.campagnelab.dl.pytorch.cifar10.Problems import create_model
from org.campagnelab.dl.pytorch.cifar10.confusion.ConfusionModel import ConfusionModel
from org.campagnelab.dl.pytorch.cifar10.utils import batch


def class_label(num_classes, predicted_index, true_index):
    return predicted_index * num_classes + true_index

class ConfusionTrainingHelper:

    def __init__(self, model, problem, args):
        self.model = ConfusionModel(create_model(model, problem), problem)
        self.problem=problem
        self.args=args
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=args.lr, momentum=0.9,
                                         weight_decay=args.L2)
        self.criterion = CrossEntropyLoss()

    def train(self, epoch, confusion_data):
        args=self.args
        optimizer=self.optimizer
        problem=self.problem
        for batch_idx, confusion_list in enumerate(batch(confusion_data, args.mini_batch_size)):
            images=[None]*args.mini_batch_size
            targets=torch.zeros(args.mini_batch_size)
            optimizer.zero_grad()
            training_loss_input = torch.zeros(args.mini_batch_size, 1)

            trained_with_input =torch.zeros(args.mini_batch_size, 1)

            for index, confusion in enumerate(confusion_list):

                num_classes = problem.num_classes()
                targets[index]=class_label(num_classes,
                                           confusion.predicted_label,confusion.true_label)
                dataset=problem.train_set() if confusion.trained_with  else problem.test_set()
                images[index], _ = dataset[confusion.example_index]
                training_loss_input[index]=confusion.train_loss
                training_loss_input[index]=1.0 if confusion.trained_with else 0.0

            if len(images)==args.mini_batch_size:
                image_input=Variable(torch.stack(images,dim=0), requires_grad=True)
                training_loss_input=Variable(training_loss_input, requires_grad=True)
                trained_with_input=Variable(trained_with_input, requires_grad=True)
                targets=Variable(targets, requires_grad=False).type(torch.LongTensor)

                outputs=self.model( training_loss_input, trained_with_input,image_input)
                loss=self.criterion(outputs,targets)
                print("epoch {} batch {} training loss {:.3f} ".format(epoch,batch_idx, loss.data[0]))
                loss.backward()
                optimizer.step()
            else:
                print("Skipping incomplete mini-batch.")