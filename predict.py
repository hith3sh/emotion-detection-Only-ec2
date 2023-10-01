import cv2
import torch
from FaceCNNModel import FaceCNN
import torchvision.transforms as transforms
import torch.nn.functional as F
import numpy as np
from PIL import Image

nb_classes = 7
model = FaceCNN(nb_classes)

state_dict  = torch.load('pytorch_weights/best weights.pth',map_location=torch.device('cpu'))
model.load_state_dict(state_dict)

#haar_file=cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
haar_file = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
face_cascade=cv2.CascadeClassifier(haar_file)

# main function which does the prediction and bounding box part
def detect_emotions(frame):

    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces=face_cascade.detectMultiScale(frame,1.3,5)

    for (x,y,w,h) in faces:

        #obtaining the face coordinates
        image = gray[y:y+h,x:x+w]
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),5)
        
        #resizing the face image
        transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((48, 48)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
        ])

        image = transform(image).unsqueeze(0)

        labels = {0 : 'angry', 1 : 'disgust', 2 : 'fear', 3 : 'happy', 4 : 'neutral', 5 : 'sad', 6 : 'surprise'}

        #setting to evaluation mode
        model.eval() 

        #disabling gradient calculation
        with torch.no_grad(): 
            feature = image
            logits = model(feature)
            probs = F.softmax(logits, dim=1)
            # argmax returns a tensor we convert it to a integer
            prediction_label = labels[probs.argmax().item()]

        print("Predicted Output:", prediction_label)

        cv2.putText(frame, '% s' %(prediction_label), (x-10, y-10),cv2.FONT_HERSHEY_SIMPLEX, 3, (0,0,255), 5)


        return frame
    return frame