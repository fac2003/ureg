from org.campagnelab.dl.pytorch.cifar10.models \
    import VGG, ResNet18, PreActResNet18, GoogLeNet, DPN92, MobileNet, \
    ResNeXt29_2x64d, ShuffleNetG2, SENet18, DenseNet121


def vgg16(problem):
    return VGG('VGG16', problem.example_size())


def vgg19(problem):
    return VGG('VGG19', problem.example_size())


def resnet18(problem):
    return ResNet18(problem.example_size())


def preactresnet18(problem):
    return PreActResNet18(problem.example_size())


def googlenet(problem):
    return GoogLeNet(problem.example_size())


def densenet121(problem):
    return DenseNet121(problem.example_size())


def resnetx29(problem):
    return ResNeXt29_2x64d(problem.example_size())


def mobilenet(problem):
    return MobileNet(input_shape=problem.example_size())


def dpn92(problem):
    return DPN92(problem.example_size())

# not converted to STL10, only works with CIFAR10:
def shufflenetg2():
    return ShuffleNetG2()

def senet18(problem):
    return SENet18(problem.example_size())


models = {
    "VGG16": vgg16,
    "VGG19": vgg19,
    "ResNet18": resnet18,
    "PreActResNet18": preactresnet18,
    "GoogLeNet": googlenet,
    "DenseNet121": densenet121,
    "ResNeXt29": resnetx29,
    "MobileNet": mobilenet,
    "DPN92": dpn92,
    "ShuffleNetG2": shufflenetg2,
    "SENet18": senet18
}


def create_model(modelName,problem):
    function = models[modelName]
    if function is None:
        print("Wrong model name: " + modelName)
        exit(1)
    # construct the model specified on the command line:
    net = function(problem)
    return net
