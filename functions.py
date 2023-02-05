import cv2
import torch
import torch.backends.cudnn as cudnn
from models.experimental import attempt_load
from utils.general import non_max_suppression
from torchvision import models
from torchvision import transforms
from PIL import Image
import time

# Load the YOLOv5 model
yolov5_weight_file = 'model100e.pt'


#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#yolov5_model = attempt_load(yolov5_weight_file, yolov5_model)
#cudnn.benchmark = True
#names = yolov5_model.module.names if hasattr(yolov5_model, 'module') else yolov5_model.names

#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#yolov5_model = torch.load(yolov5_weight_file, device)
#cudnn.benchmark = True
#names = yolov5_model.module.names if hasattr(yolov5_model, 'module') else yolov5_model.names


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
yolov5_model = attempt_load(yolov5_weight_file, device=device, inplace=True, fuse=True)
#model = attempt_load(yolov5_weight_file, device=device, inplace=True, fuse=True)
cudnn.benchmark = True 
names = yolov5_model.module.names if hasattr(yolov5_model, 'module') else yolov5_model.names



# Set variables
conf_set = 0.1
frame_size = (800, 480)

colors = {
    'helmet': (255, 0, 0),
    'rider': (0, 255, 0),
    'number': (0, 0, 255),
    'no_helmet': (0, 100, 255),
    # add more classes and colors as needed
}

def detect_objects(frame):
    """
    Use the YOLOv5 model to detect objects in a frame, and draw rectangles of different colors for different classes.
    """
    # Convert the frame to a tensor and pass it through the model
    img = torch.from_numpy(frame)
    img = img.permute(2, 0, 1).float().to(device)
    img /= 255.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # Use torch.no_grad to improve performance
    with torch.no_grad():
        pred = yolov5_model(img, augment=False)[0]
        pred = non_max_suppression(pred, conf_set, 0.30)

        # Create a list of detections
        detections = []
        for det in pred:
            if len(det):
                for d in det:  # d = (x1, y1, x2, y2, conf, cls)
                    x1 = int(d[0].item())
                    y1 = int(d[1].item())
                    x2 = int(d[2].item())
                    y2 = int(d[3].item())
                    conf = round(d[4].item(), 2)
                    c = int(d[5].item())
                    detected_name = names[c]
                    detections.append((x1, y1, x2, y2, conf, detected_name))
                    
                    # Draw rectangle of the corresponding color
                    color = colors.get(detected_name, (255, 255, 255))
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, detected_name, (x1, y1), cv2.FONT_HERSHEY_DUPLEX, 1, color, 2)
                    
        return detections

