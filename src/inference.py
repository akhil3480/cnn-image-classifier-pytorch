import torch
from PIL import Image
import torchvision.transforms as transforms
from model import FashionCNN

class_names = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot"
]

def load_model(weights_path, device):
    model = FashionCNN().to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    return model

def preprocess_image(image_path):
    img = Image.open(image_path).convert("L")
    img = img.resize((28, 28))

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    img_tensor = transform(img).unsqueeze(0)
    return img_tensor

def predict(image_path, model, device):
    img_tensor = preprocess_image(image_path).to(device)

    with torch.no_grad():
        outputs = model(img_tensor)
        _, pred = torch.max(outputs, 1)

    return class_names[pred.item()]

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model("models/fashion_cnn_best.pth", device)

    # Change this to the path of an image you want to test
    test_image = "test_sample.png"
    result = predict(test_image, model, device)
    print("Predicted class:", result)
