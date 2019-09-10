from moviepy.editor import VideoFileClip
from svm_pipeline import *
from yolo_pipeline import *
from lane import *
import matplotlib.pyplot as plt
import cv2

def pipeline_yolo(img):

    img_undist, img_lane_augmented, lane_info = lane_process(img)
    output = vehicle_detection_yolo(img_undist, img_lane_augmented, lane_info)

    return output

def pipeline_svm(img):

    img_undist, img_lane_augmented, lane_info = lane_process(img)
    output,boxes = vehicle_detection_svm(img_undist, img_lane_augmented, lane_info)
    # text_file = open("detection.txt", "a")
    # for i in range(len(boxes)):
    #     text_file.write('Detected Object:')
    #     box=boxes[i]
    #     for j in range(len(box)):
    #         rect=box[j]
    #         print(rect[0])
    #         text_file.write("(%s,%s)" %(rect[0], rect[1]))
    #     text_file.write('\n')
    # text_file.close()
    return output


if __name__ == "__main__":
    filename = 'examples/Vehile_data_1.m2ts'
    cap= cv2.VideoCapture(filename)
    while(cap.isOpened()):
        ret, frame = cap.read()
        frame=cv2.resize(frame,(1280,720))
        yolo_result = pipeline_yolo(frame)
        plt.figure()
        plt.imshow(yolo_result)
        plt.title('yolo pipeline', fontsize=30)
        plt.show()
        draw_img = pipeline_svm(frame)
        fig = plt.figure()
        plt.imshow(draw_img)
        plt.title('svm pipeline', fontsize=30)
        plt.show()

    # demo = 3  # 1:image (YOLO and SVM), 2: video (YOLO Pipeline), 3: video (SVM pipeline)
    #
    # if demo == 1:
    #     filename = 'examples/test1.mp4'
    #     image = cv2.imread(filename)
    #
    #     #(1) Yolo pipeline
    #     yolo_result = pipeline_yolo(image)
    #     plt.figure()
    #     plt.imshow(yolo_result)
    #     plt.title('yolo pipeline', fontsize=30)
    #
    #     #(2) SVM pipeline
    #     draw_img = pipeline_svm(image)
    #     fig = plt.figure()
    #     plt.imshow(draw_img)
    #     plt.title('svm pipeline', fontsize=30)
    #     plt.show()
    #
    # elif demo == 2:
    #     # YOLO Pipeline
    #     video_output = 'examples/project_YOLO.mp4'
    #     clip1 = VideoFileClip("examples/MAH00134.MP4").subclip(0,150)
    #     clip = clip1.fl_image(pipeline_yolo)
    #     clip.write_videofile(video_output, audio=False)
    #
    # else:
    #     # SVM pipeline
    #     video_output = 'examples/output.mp4'
    #     clip1 = VideoFileClip("examples/test1.mp4").subclip(1,300)
    #     clip = clip1.fl_image(pipeline_svm)
    #     clip.write_videofile(video_output, audio=False)


