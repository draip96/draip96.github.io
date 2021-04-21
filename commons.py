import io

from PIL import Image
from torchvision import models
import torchvision.transforms as transforms



def get_model(model_name):
    if model_name == "DenseNet":
        model = models.densenet121(pretrained=True)
    elif model_name == "AlexNet":
        model = models.alexnet(pretrained=True)
    elif model_name == "VGG":
        model = models.vgg16(pretrained=True)
    elif model_name == "ResNet":
        model = models.wide_resnet50_2(pretrained=True)
    elif model_name == "GoogleNet":
        model = models.googlenet(pretrained=True)
    elif model_name == "Inception":
        model = models.inception_v3(pretrained=True)
    model.eval()
    return model

def transform_image(image_bytes):
    my_transforms = transforms.Compose([transforms.Resize(255),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize(
                                            [0.485, 0.456, 0.406],
                                            [0.229, 0.224, 0.225])])
    image = Image.open(io.BytesIO(image_bytes))
    return my_transforms(image).unsqueeze(0)


def format_class_name(class_name):
    class_name = class_name.replace('_', ' ')
    class_name = class_name.title()
    return class_name
