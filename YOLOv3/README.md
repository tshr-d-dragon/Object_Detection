# Object Detection using YOLOv3 with pretrained (MS COCO) weights
 
In this project, with the help of pretrained weights of YOLOv3 on MS COCO 2017 dataset, we can detect 80 different objects. Pretrained weights and config files can be found at https://pjreddie.com/darknet/yolo/

## Project Structure
1. The file coco.names contains names of all the 80 classes present in MS COCO dataset
2. The file yolo_v3.py contains all the code for object detection using yolov3 model. The value of variable wdt will change as per the type of yolo model is used. For example, here, I have used yolo-320 and hence, wdT = 320. For yolo-tiny and yolo-spp it will be 320 and 608, respectively. Also, for webcam change the value of vairable video_file = 0 in the main function.
3. Ensure that video to be used for prediction has name Video.mp4, otherwise change the value of vairable video_file in the main function accordingly.
4. I have uploaded prediction videos for yolov3-320 with the same script stored in prediction folder.

## To run the prject, follow below steps
1. Ensure that you are in the project home directory
2. Create anaconda environment
3. Activate environment
4. >pip install -r requirement.txt
5. >python Face_Mask_Detection.py -p path_of_preject_directory

## Please feel free to connect for any suggestions or doubts!!!

## Credits
The credits for videos used for prediction goes to:
1. 
