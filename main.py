import blobconverter
import cv2
import depthai as dai
import numpy as np
import time

pipeline = dai.Pipeline()

print("Creating Color Camera...")
cam = pipeline.create(dai.node.ColorCamera)
cam.setPreviewSize(1080, 1080)
cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
cam.setInterleaved(False)
cam.setBoardSocket(dai.CameraBoardSocket.RGB)

cam_xout = pipeline.create(dai.node.XLinkOut)
cam_xout.setStreamName("color")
cam.preview.link(cam_xout.input)

# ImageManip that will crop the frame before sending it to the Face detection NN node
face_det_manip = pipeline.create(dai.node.ImageManip)
face_det_manip.initialConfig.setResize(300, 300)
face_det_manip.initialConfig.setFrameType(dai.RawImgFrame.Type.RGB888p)
cam.preview.link(face_det_manip.inputImage)

# create face detection Nnet Manager
face_det_nn = pipeline.create(dai.node.MobileNetDetectionNetwork)

face_det_nn.setConfidenceThreshold(0.5)
face_det_nn.setBlobPath(blobconverter.from_zoo(name="face-detection-retail-0004", shaves=6))
face_det_manip.out.link(face_det_nn.input)

# send output to host for testing
face_det_xout = pipeline.create(dai.node.XLinkOut)
face_det_xout.setStreamName("detection")
face_det_nn.out.link(face_det_xout.input)

# ImageManip that will crop the frame before sending it to the gaze detection (manip scrip) NN node
image_gaze = pipeline.create(dai.node.ImageManip)
image_gaze.initialConfig.setFrameType(dai.RawImgFrame.Type.RGB888p)
cam.preview.link(image_gaze.inputImage)

# Script node will take the output from the face detection NN as an input and set ImageManipConfig
# to the 'gaze detection' to crop the initial frame
image_manip_script = pipeline.create(dai.node.Script)
face_det_nn.out.link(image_manip_script.inputs['face_det_in'])
image_gaze.out.link(image_manip_script.inputs['image'])


# face_det_nn.passthrough.link(image_manip_script.inputs['passthrough'])

image_manip_script.setScript("""
    import numpy as np
    import cv2

    def bbox_head_np(bb):
        bb.xmin = xmin * 0.7
        bb.ymin = ymin *0.4
        bb.xmax = xmax *1.25
        bb.ymax = ymax *1.25
        return bb

    def extract_head(image, bbox_head):
        extracted_head = image[int(bbox_head[1]):int(bbox_head[3]), int(bbox_head[0]):int(bbox_head[2])]
        # print(extracted_head)
        input_arr = cv2.resize(image, (224, 224))
        input_arr = cv2.cvtColor(input_arr, cv2.COLOR_BGR2RGB)
        input_arr = np.array([input_arr]).astype(np.float16)
        return extracted_head
       
    bbox_head = bbox_head_np(face_det_in)  
    image1 = extract_head(image, bbox_head)
    node.io['out'].send(image1)    
    """)

img_xout = pipeline.create(dai.node.XLinkOut)
img_xout.setStreamName("gaze")
image_manip_script.outputs['out'].link(img_xout.input)

gaze_nn = pipeline.create(dai.node.NeuralNetwork)
gaze_nn.setBlobPath('aze_detection_regression_001_MyriadX_FP16.blob')
image_manip_script.outputs['out'].link(gaze_nn.input)

gaze_xout = pipeline.create(dai.node.XLinkOut)
gaze_xout.setStreamName("gaze")
gaze_nn.out.link(gaze_xout.input)

# Connect to device and start pipeline
with dai.Device(pipeline) as device:

    print('Connected cameras: ', device.getConnectedCameras())
    # Print out usb speed
    print('Usb speed: ', device.getUsbSpeed().name)

    # Output queues will be used to get the rgb frames and nn data from the outputs defined above
    gazeNNQueue = device.getOutputQueue(name="out")

    startTime = time.monotonic()
    counter = 0
    fps = 0

    while True:
        inDet = gazeNNQueue.get()
        print('test: ', inDet)
        


        counter+=1
        current_time = time.monotonic()
        if (current_time - startTime) > 1 :
            fps = counter / (current_time - startTime)
            counter = 0
            startTime = current_time
            print(fps)
        
        print(inDet)

        # if inDet is not None:
        #     detections = inDet.detections

        # for detection in detections:
        #     print(detection)
