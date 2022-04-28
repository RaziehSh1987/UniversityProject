
#Bishop's Universirty
#Image Processing (Winter2022)-Dr Dorra Riahi
#                                Members
#Razieh Shahsavar(002341606)   -    Maryam Bayatzadeh(002338161)  -   Salar Rezaei

# importing libraries
import cv2

# Create a VideoCapture object and read from input file
cap = cv2.VideoCapture('D:\\Razieh\\ImageProcessing2022\\TrackingVideo.avi')

# Capture frame-by-frame in count of 10 frame
for i in range(10):
    # [return `false` to 'success' parameter ,if no frames has been grabbed] and  [image the video frame is returned here to 'frame' parameter]
    success,frame = cap.read()

#if there is no frame,exit
if not success:
    exit(1)
#return number of pixel of height and width of frame of video (480,640)
frame_h,frame_w = frame.shape[:2]
size = (frame_h,frame_w)

#define area of window track base on the frame_width and frame_height and the start point of detection(x,y) in the video
w =frame_w//5
h =frame_h//5
x =600
y =320
# I have manually calculated the initial position as meanshift itself would be unable to do.
track_window = (x,y,w,h)

#We use color histograms for object tracking in images, such as with the CamShift algorithm.
#"roi" is the image that we want to compute a histogram for.
roi =frame[y:y+h, x-w:x]

#Hence we would like to convert the RGB colorspace to HSV. The hue component of HSV model helps
# us in understanding the color of objects,so we  use ".cvtColor()"
hsv_roi = cv2.cvtColor(roi,cv2.COLOR_BGR2HSV)

#If a mask is provided, a histogram will be computed for masked pixels only. If we do not have a
# mask or do not want to apply one, we can just provide a value of None.
mask = None

#We us the cv2.calcHist function to build our histograms
#{"[0]" shows channels: A list of indexes, where we specify the
# index of the channel we want to compute a histogram for. To
# compute a histogram of a grayscale image, the list would be [0]}
#{"[180]"This is the number of bins we want to use when computing a histogram. Again,this
# is a list, one for each channel we are computing a histogram for.}
#"[0,180]" are The range of possible pixel values, hence we are using color space (HSV) the range is between 0,180
roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0,180])

#we are simply dividing the raw frequency counts for each bin of the histogram by the Normalizing
cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)

# Setup the termination criteria, either 10 iteration or move by atleast 1 pt
term_criteria =(cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS,10,1)

# "Forcc" set the  code of codec(example:mp4v) used to compress the frames.
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

#set name of the output video file.To save image sequence use a proper filename
#"10" is Framerate of the created video stream.
#"cap.get(3)" is Size of the video frames.
#"ap.get(4)" isColor
result = cv2.VideoWriter('BoyMotion(RaziehSahsavar-MaryamBayatzadeh-SalarRezaei).avi',fourcc,10,(int(cap.get(3)),int(cap.get(4))))

true,frame = cap.read()
while true:
    hsv =cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)

    #Back Projection is a way of recording how well the pixels of a given image fit the distribution of pixels in a histogram model.
    #To make it simpler: For Back Projection, you calculate the histogram model of a feature and then use it to find this feature in an image.
    back_project = cv2.calcBackProject([hsv], [0],roi_hist, [0,180],1)

    #o use meanshift in OpenCV, first we need to setup the target, find its histogram so that we can backproject the target on each frame for
    # calculation of meanshift. We also need to provide an initial location of window. For histogram, only Hue is considered here(that we implemented in the befor lines)
    num_iters,track_window = cv2.meanShift(back_project,track_window,term_criteria)

    x,y,w,h = track_window

    #draw circle around the image_tracker
    cv2.circle(frame,(x,y+5),60,(0,0,255),6)
    # cv2.imshow('back-projection',back_project)

    #show the output image that is tracked
    cv2.imshow('meanshift', frame)
    result.write(frame)

    #condition for exit the run
    k = cv2.waitKey(50)
    if k == 27:
        break
    if k == ord('p'):
        cv2.waitKey(-1)

    true,frame = cap.read()
