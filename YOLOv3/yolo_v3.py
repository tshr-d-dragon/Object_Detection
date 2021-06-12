
import cv2
import numpy as np
from numpy import random
import os
import argparse


global colors, wdT, confThreshold, nmsThreshold

# Colors for bboxes
np.random.seed(17) 
colors = random.randint(0, 255, size=(80, 3))

# since yolo-320 is used
wdT = 320

confThreshold = 0.3
nmsThreshold = 0.2


def set_cwd():
    
    parser = argparse.ArgumentParser(description = 'path of project directory')
    parser.add_argument("-p", "--path", default='/YOLOv3/', 
                    required = True, type = str, help = 'Give path of project directory')
    args = parser.parse_args()
    os.chdir(args.path)


def load_model():
        
    ## Coco Names
    classesFile = r'coco.names'
    classNames = []
    with open(classesFile, 'rt') as f:
        classNames = f.read().rstrip('\n').split('\n')
        
    ## Model Files
    modelConfiguration = "yolov3.cfg" # yolo-320
    modelWeights = "yolov3.weights" # yolo-320
    
    net = cv2.dnn.readNet(modelConfiguration, modelWeights, framework = 'Darknet')
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    # # For GPU
    # net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    # net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA) ###cv.dnn.DNN_TARGET_OPENCL
    return classNames, net
    
    
def findObjects(outputs, img, classNames):
    
    hT, wT, cT = img.shape
    bbox = []
    classIds = []
    confs = []
    for output in outputs:
        for det in output:
            scores = det[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > confThreshold:
                w,h = int(det[2]*wT) , int(det[3]*hT)
                x,y = int((det[0]*wT)-w/2) , int((det[1]*hT)-h/2)
                bbox.append([x,y,w,h])
                classIds.append(classId)
                confs.append(float(confidence))
 
    indices = cv2.dnn.NMSBoxes(bbox, confs, confThreshold, nmsThreshold)
    
    for i in indices:
        
        text = f'YOLOv3-{wdT} (CPU)'
        text_width, text_height = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        cv2.rectangle(img, (0, 0), (text_width+3+5, text_height+3+5), (51, 68, 255), -1)
        cv2.putText(img, text, (3, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        
        i = i[0]
        box = bbox[i]
        x, y, w, h = box[0], box[1], box[2], box[3]
        c = classIds[i]
        color = (int(colors[c][0]), int(colors[c][1]), int(colors[c][2]))
        
        cv2.rectangle(img, (x, y), (x+w, y+h), color, 1)
        
        label = f'{classNames[classIds[i]].lower()}' # ': {int(confs[i]*100)}%' # for confidence
        FONT = cv2.FONT_HERSHEY_SIMPLEX
        FONT_SCALE = 0.3
        FONT_THICKNESS = 1 
        label_width, label_height = cv2.getTextSize(label, FONT, FONT_SCALE, FONT_THICKNESS)[0]
        
        cv2.rectangle(img, (x, y-label_height-6+3), (x+label_width+3+5, y), color, -1)
        
        cv2.putText(img, label, (x+3, y-3), FONT, FONT_SCALE, (0, 0, 0), FONT_THICKNESS, cv2.LINE_AA)
        
        return img


def main():
    
    classNames, net = load_model()
    
    video_file = r'Video.mp4'
    # video_file = 0  ## for webcam
    cap = cv2.VideoCapture(video_file)
    
    No_of_frames = int(cap.get(7))
    frame_width = int(cap.get(3)) 
    frame_height = int(cap.get(4)) 
    size = (frame_width, frame_height) 
    vid_fps = cap.get(5)
    out = cv2.VideoWriter(r'Video_YOLOv3_pred.mp4', cv2.VideoWriter_fourcc(*'MPEG'), vid_fps, size)
    ret = True
    
    while ret:
        
        frameId = int(round(cap.get(1)))    
        ret, img = cap.read() 
        
        # if frameId != No_of_frames - 10:
        #     continue
        
        # preprocessing
        blob = cv2.dnn.blobFromImage(img, 1 / 255.0, (wdT, wdT), [0, 0, 0], 1, crop=False)
        
        net.setInput(blob)
        layersNames = net.getLayerNames()
        outputNames = [(layersNames[i[0] - 1]) for i in net.getUnconnectedOutLayers()]
	        outputs = net.forward(outputNames)
        
        img = findObjects(outputs, img, classNames)
        
        cv2.imshow('Image', img)
        
        out.write(img)
        
        print(frameId)
        
        if frameId == No_of_frames - 1:
            break
        
        print(frameId)
        if cv2.waitKey(1) & 0xFF == ord('q'): 
            break
    
    cap.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    set_cwd()
    main()