# Object Detection using YOLOv3 with pretrained (MS COCO) weights
 
In this project, with the help of pretrained weights of YOLOv3 on MS COCO 2017 dataset, we can detect 80 different objects. Pretrained weights and config files can be found at https://pjreddie.com/darknet/yolo/

![Demo1](https://github.com/tshr-d-dragon/Object_Detection/blob/main/YOLOv3/prediction/People_pred_YOLOv3.gif)
![Demo2](https://github.com/tshr-d-dragon/Object_Detection/blob/main/YOLOv3/prediction/Traffic_pred_YOLOv3.gif)
![Demo3](https://github.com/tshr-d-dragon/Object_Detection/blob/main/YOLOv3/prediction/Los_Angeles_pred_YOLOv3.gif)

## Project Structure
1. The file coco.names contains names of all the 80 classes present in MS COCO dataset
2. The file yolo_v3.py contains all the code for object detection using yolov3 model. The value of variable wdt will change as per the type of yolo model is used. For example, here, I have used yolo-320 and hence, wdT = 320. For yolo-tiny and yolo-spp it will be 320 and 608, respectively. Also, for webcam change the value of vairable video_file = 0 in the main function.
3. Ensure that video to be used for prediction has name Video.mp4, otherwise change the value of vairable video_file in the main function accordingly.
4. Also ensure along weights, config and coco.names files are in same project directory.
5. I have uploaded prediction videos for yolov3-320 with the same script stored in prediction folder.

## To run the prject, follow below steps
1. Ensure that you are in the project home directory
2. Create anaconda environment
3. Activate environment
4. >pip install -r requirement.txt
5. >python Face_Mask_Detection.py -p path_of_preject_directory

## Please feel free to connect for any suggestions or doubts!!!

### More predicted videos can be found [here](https://www.youtube.com/watch?v=SZZ4ozyXMls&list=PLpNcmMJb4QJIXbkmfxw0plnfpS3dWSrHZ&index=1)

## Credits
The credits for videos used for prediction goes to:
1. Video by <a href="https://pixabay.com/users/preditorcuts-4627334/?utm_source=link-attribution&amp;utm_medium=referral&amp;utm_campaign=image&amp;utm_content=32851">William Sevilla</a> from <a href="https://pixabay.com/?utm_source=link-attribution&amp;utm_medium=referral&amp;utm_campaign=image&amp;utm_content=32851">Pixabay</a>
2. Video by <a href="https://pixabay.com/users/coverr-free-footage-1281706/?utm_source=link-attribution&amp;utm_medium=referral&amp;utm_campaign=image&amp;utm_content=6387">Coverr-Free-Footage</a> from <a href="https://pixabay.com/?utm_source=link-attribution&amp;utm_medium=referral&amp;utm_campaign=image&amp;utm_content=6387">Pixabay</a>
3. Video by <a href="https://pixabay.com/users/lizmavor-11592643/?utm_source=link-attribution&amp;utm_medium=referral&amp;utm_campaign=image&amp;utm_content=21437">Elizabeth Mavor</a> from <a href="https://pixabay.com/?utm_source=link-attribution&amp;utm_medium=referral&amp;utm_campaign=image&amp;utm_content=21437">Pixabay</a>

**For more accurate results, we can use YOLOv3-608 or YOLOv3-spp. Because of infrastucture issue, I have used YOLOv3-320.**
