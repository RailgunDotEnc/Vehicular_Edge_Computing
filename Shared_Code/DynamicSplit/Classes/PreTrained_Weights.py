from torchvision.models import resnet50, ResNet50_Weights

# Initialize model
weights = ResNet50_Weights.DEFAULT

hold=weights.get_state_dict(True).keys()
print(hold)
model = resnet50(weights=weights)


# Set model to eval mode
model.eval()