from org.campagnelab.dl.pytorch.images.models \
    import VGG, ResNet18, PreActResNet18, GoogLeNet, DPN92, MobileNet, \
    ResNeXt29_2x64d, ShuffleNetG2, SENet18, DenseNet121, PreActResNet34, PreActResNet50, PreActResNet152, \
    PreActResNet101, torch
from org.campagnelab.dl.pytorch.images.models.capsules.capsule_net import CapsNet3
from org.campagnelab.dl.pytorch.images.models.dual import LossEstimator_sim
from org.campagnelab.dl.pytorch.images.models.preact_resnet_dual import PreActResNet18Dual
from org.campagnelab.dl.pytorch.images.models.preact_resnet_recursive import PreActResNet18Recursive
from org.campagnelab.dl.pytorch.images.models.recursive import VGGRecursive
from org.campagnelab.dl.pytorch.images.models.vgg_dual import VGGDual
from org.campagnelab.dl.pytorch.images.utils import init_params


def capsnet3_8_64(problem):

    return CapsNet3(example_size=problem.example_size(), num_conv_in_channel=3, num_conv_out_channel=256,
                    num_primary_unit=8,
                    num_classes=problem.num_classes(), output_unit_size=16, num_routing=3,
                    use_reconstruction_loss=True, cuda_enabled=torch.cuda.is_available(),
                    capsule_out_size=64)


def capsnet3_16_64(problem):

    return CapsNet3(example_size=problem.example_size(), num_conv_in_channel=3, num_conv_out_channel=256,
                    num_primary_unit=16,
                    num_classes=problem.num_classes(), output_unit_size=16, num_routing=3,
                    use_reconstruction_loss=True, cuda_enabled=torch.cuda.is_available(),
                    capsule_out_size=64)

def capsnet3_24_8(problem):

    return CapsNet3(example_size=problem.example_size(), num_conv_in_channel=3, num_conv_out_channel=256,
                    num_primary_unit=24,
                    num_classes=problem.num_classes(), output_unit_size=16, num_routing=3,
                    use_reconstruction_loss=True, cuda_enabled=torch.cuda.is_available(),
                    capsule_out_size=8)


def vgg16(problem):
    return VGG('VGG16', problem.example_size())

def vgg11_recursive(problem):
    return VGGRecursive('VGG11', problem.example_size())



def vgg13_recursive(problem):
    return VGGRecursive('VGG13', problem.example_size())


def vgg16_recursive(problem):
    return VGGRecursive('VGG16', problem.example_size())

def vgg19_recursive(problem):
    return VGGRecursive('VGG19', problem.example_size())


def vgg256_recursive(problem):
    return VGGRecursive('VGG256', problem.example_size())


def vgg16dual(problem):
    return VGGDual('VGG16', problem.example_size(), loss_estimator=LossEstimator_sim)


def vgg19(problem):
    return VGG('VGG19', problem.example_size())


def resnet18(problem):
    return ResNet18(problem.example_size())

def preactresnet18_recursive(problem):
    return PreActResNet18Recursive(input_shape=problem.example_size())

def preactresnet18(problem):
    return PreActResNet18(problem.example_size())


def preactresnet18dual(problem):
    return PreActResNet18Dual(problem.example_size(), loss_estimator=LossEstimator_sim)


def preactresnet34(problem):
    return PreActResNet34(problem.example_size())


def preactresnet50(problem):
    return PreActResNet50(problem.example_size())


def preactresnet101(problem):
    return PreActResNet101(problem.example_size())


def preactresnet152(problem):
    return PreActResNet152(problem.example_size())


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
    "VGG11Recursive": vgg11_recursive,
    "VGG13Recursive": vgg13_recursive,
    "VGG16Recursive": vgg16_recursive,
    "VGG19Recursive": vgg19_recursive,
    "VGG256Recursive": vgg256_recursive,
    "VGG16Dual": vgg16dual,
    "VGG19": vgg19,
    "ResNet18": resnet18,
    "PreActResNet18": preactresnet18,
    "PreActResNet18Recursive": preactresnet18_recursive,
    "PreActResNet18Dual": preactresnet18dual,
    "PreActResNet34": preactresnet34,
    "PreActResNet50": preactresnet50,
    "PreActResNet101": preactresnet101,
    "PreActResNet152": preactresnet152,
    "GoogLeNet": googlenet,
    "DenseNet121": densenet121,
    "ResNeXt29": resnetx29,
    "MobileNet": mobilenet,
    "DPN92": dpn92,
    "ShuffleNetG2": shufflenetg2,
    "SENet18": senet18,
    "CapsNet3_8_64": capsnet3_8_64,
    "CapsNet3_16_64": capsnet3_16_64,
    "CapsNet3_24_8": capsnet3_24_8,

}


def create_model(modelName, problem, dual=False):
    modelName += ("Dual" if dual else "")
    function = models[modelName]
    if function is None:
        print("Wrong model name: " + modelName)
        exit(1)
    # construct the model specified on the command line:
    net = function(problem)
    net.apply(init_params)
    return net
