# public_uasproject
Drone Detection with Waggle Sensor Nodes

## Investigating Drone Detection Capabilities with the Argonne Waggle Platform

The main research question for this project is: how effective can a low-cost distributed visual based edge computed machine learning algorithm be at autonomously detecting flying objects over a geographic region?  The project will use the Waggle based sensor nodes with a slightly upgraded camera to run custom detection algorithms in real-time.  These detection algorithms will use state-of-the-art computer vision and neural network based algorithms to detect moving objects in the sky and classify the object type.  Part of the research will include using different types of neural network modeling techniques to create the most effective classification algorithm.  It is hypothesized that using a combination of static image classification models, motion analysis and temporal based classification models will improve accuracy of classification in this “low-power” camera type node.  

## Test Deployment for Data Collection and Algorithm Demonstration

An experimental set of six Argonne Waggle-based nodes will be deployed around the Chicago Soldier Field Stadium for the purposes of research in detecting drones.  These units will be housed in generic enclosures rather than the branded AoT enclosures.  The units will contain no sensors or microphones or downward-facing cameras—only a single upward facing camera.  The underlying Waggle control systems will be managed by the Argonne Waggle team, with authorized user access only to the edge computing device for software development and test. By deploying these test nodes around Soldier Field we will be able to not only test a number of different algorithms in real-world situations, but also continually collect data that will ultimately improve the algorithms over time.

## Data Collection

The project will be both an effort in data collection and algorithm testing and development.  Over the course of the project we will continually collect data that can be used to train (re-train) the classification algorithms which will ultimately improve accuracy of the model. Toward the start of the project (Phase 1) we will record and collect raw video data in the form of 5 minute mp4 video files.  These can be used to both extract training data (images of moving objects and their motion) and test the object detection and classification algorithm offline prior to deployment.  Once the object detection algorithm has been refined (Phase 2) a special "data collector" version of that code will be used to save the images of moving objects and their motion tracks to disk and transfer back to Argonne servers.  This will aleviate the need to save and transfer large video files over the data link. When the object detection and classification algorithms are being tested and demonstrated (Phase 3) the only data saved and transfered back to Argonne servers will be object classification "alerts".  The alerts will contain the type of object, the classification probabilities, the cropped image of the object and information about the object trajectory.

The video data will be transferred via scp to project personnel computers at Argonne and will be stored on a central GSS server. (Phase 1) Initially the cropped images of detected objects and motion path data will be collected on the Waggle node, zipped up and transferred via scp to project personnel computers at Argonne. (Phase 2a)  At some point the transfer of Phase 2 data will transition from raw scp transfer to using the Waggle Beehive communication protocols.  (Phase 2b)  This will mean that the data will be transferred to the Beehive server and then moved to GSS servers.  Similarly in Phase 3 the classification "alerts" data will sent via a custom communications protocol initially and then will be changed to use the Waggle Beehive protocols. 

## Code and Algorithms

In Phase 1 we will use the standard ffmpeg program to capture and record raw video from the camera on the Waggle node.  As mentioned in the previous section ffmpeg will be configured to save the videos in 5 minute increments using the mp4 format.  No custom code is needed for this operation.

In Phase 2a we will execute custom C++ code that will utilize the video4linux drivers and the OpenCV computer vision libraries.  This code is provided in the XXX directory of this repository.  The code will be run as a standalone linux executable and will not interact with the Waggle framework.  The code uses a number of computer vision techniques to detect moving objects against a potentially moving background along with some tracking and motion prediction techniques to reject clutter.  A combination of MOG differencing, Kalmann filters, thresholding, contouring, motion trajectory prediction, and ... are used.

In Phase 2b we will update the code from Phase 2a to utilize the Waggle Beehive communication protocols to transfer the collected data back to the Beehive server automatically.

In Phsae 3a we will execute custom C++ code that utilizes the algorithms from 2a and adds a trained neural network model for classification of the detected objects based on the object static image and the object motion path.  This code is provided in the XXX directory of this repository.  The neural network code uses the Caffe library framework for the forwarrd-pass classification model.  

### Dependencies are: ffmpeg, OpenCV compiled with OpenCL, Caffe

