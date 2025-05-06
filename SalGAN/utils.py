import os
import cv2
from torchvision import transforms

IMG_SIZE = (256,192)

#Create path if it doesn't exist
def createPath(path):
    if not os.path.exists(path):
        os.makedirs(path)

#Resize Image
def resizeImage(path, size=IMG_SIZE):
    img = cv2.imread(path)
    return cv2.resize(img, size, interpolation=cv2.INTER_AREA)

#Predict and Save per epoch
def predict(model, image, epoch, path, device):
    image = transforms.ToTensor()(image).unsqueeze(0).to(device)
    result = model(image).cpu().data.squeeze()
    result = result.numpy()*255
    cv2.imwrite(path+'ep'+str(epoch)+'.png',result)

#Predict and Save
def predict_single(model, image, path, device):
    image = transforms.ToTensor()(image).unsqueeze(0).to(device)
    result = model(image).cpu().data.squeeze()
    result = result.numpy()*255
    cv2.imwrite(path,result)

    return result

#Predict and use for plotting
def predict_plot(model, image, device):
    image = transforms.ToTensor()(image).unsqueeze(0).to(device)
    result = model(image).cpu().data.squeeze(0)

    return result



