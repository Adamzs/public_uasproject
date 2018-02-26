// notes
/*
Rights:
This software is property of Argonne National Laboratory.
This software shall not be used or distributed without the
express consent of Argonne National Laboratory.
Date:
Last updated Feb 2018
Author:
Sean R. Richardson (except as noted in section comments below)
United States Air Force National Laboratory Technical Fellow
Argonne National Laboratory, Illinois
(This software was written as part of a project led by Adam Szymanski.)
Purpose:
This software is meant to use upward-looking video to detect, track, and categorize
airborne objects, then send that track information to a network for further processing.
Specifically, it determines which airborne objects are drones, as opposed to
birds or aircraft. It is optimized to run at ~30 fps on an Odroid XU4 computer.
Compilation command for Odroid:
clear && g++-5 -O3 ~/ddd/caffe/tracker.cpp \
-o ~/ddd/caffe/tracker \
-L /usr/local/lib \
-lexiv2 -lboost_system -lcaffe -lglog -lpthread \
`pkg-config --cflags --libs opencv` \
-std=c++14 && \
~/ddd/caffe/tracker
Program input:
Input to this program is Withrobot 5 megapixel oCam
set to 1280x720 resolution and 30 fps. 
Can also accept .mp4 videos with the same resolution and fps.
If program is compiled as "tracker", you can view help 
with the following terminal command: 
./tracker h
Program output:
Outputs from this program are the last updated still images 
of tracks, with associated track info encoded as metadata. 
Since custom metadata tags could not be used, the following 
standard "ascii"-type metadatatags were used.
Exif.Image.ImageID
	Associated track data: track name
	Example: t20180123123456123456 (format tYYYYMMDDhhmmssdddddd)
	Explanation: format is a 't' followed by date and time to the microsecond 
		that track was first detected
Exif.Image.TargetPrinter 
	Associated track data: y position of last track update
	Example: 123 (should be read into int variable)
	Explanation: y position is number of pixels down from top 
		of camera's 1280x720 pixel image, where the first pixel is pixel "0"
Exif.Image.ImageHistory 
	Associated track data: x position of last track update
	Example: 123 (should be read into int variable)
	Explanation: x position is number of pixels right from left side 
		of camera's 1280x720 pixel image, where the first pixel is pixel "0"
Exif.Image.Make 
	Associated track data: y speed of last track update
	Example: 1.23 (should be read into float variable)
	Explanation: y speed is pixels per 1/30th of a second, 
		with positive being downward in the camera's image
Exif.Image.Model 
	Associated track data: x speed of last track update
	Example: 1.23 (should be read into float variable)
	Explanation: x speed is pixels per 1/30th of a second, 
		with positive being rightward in the camera's image
Exif.Image.Software
	Associated track data: overall speed of last track update
	Example: 1.74 (should be read into float variable)
	Explanation: overall speed is pixels per 1/30th of a second
		(will always be positive, as it is a magnitude combining
		the y and x speed values)
Exif.Image.ImageDescription 
	Associated track data: machine learning category label
	Example: bird 3 (format: plain-english label and category number)
	Explanation: most likely result from machine learning
Exif.Image.Artist 
	Associated track data: machine learning most likely category probability
	Example: 0.95 (should be read into float variable)
	Explanation: 1.00 would mean machine learning algorithm is 100% certain
		that the track category label is correct.
Exif.Image.SpectralSensitivity 
	Associated track data: coasted frames
	Example: 2 (should be read into int variable)
	Explanation: how many frames track was coasted immediately before
		track was dropped and this image was created; more coasted frames may
		indicate that metadata track information is less accurate
Exif.Image.CameraSerialNumber 
	Associated track data: box number
	Example: 12 (should be read into string or int variable)
	Explanation: number that should be unique to the box 
		that generated this track (i.e., the camera and Odroid combination);
		number is saved in file at /home/odroid/boxnumber.txt.
*/

// for Odroid, where Caffe was build with CPU_ONLY 
//(but C++ compiler doesn't know that and tries to find cublas_v2.h)
#define CPU_ONLY

//includes
#include <iostream>
#include <fstream>
#include <string>
#include <stdio.h>
#include <sstream>
#include <sys/time.h>
#include <vector>
#include <memory>
#include <queue>
#include <chrono>
#include <thread>
#include <atomic>
#include <mutex>
#include <iomanip>
#include <caffe/caffe.hpp>
#include <algorithm>
#include <iosfwd>
#include <utility>

// for OpenCV 3
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/video/background_segm.hpp>
#include <opencv2/core/cuda.hpp>

// for oCam
#include <queue>
#include "withrobot_platform.hpp"

// for saving image metadata
#include <exiv2/exiv2.hpp>
#include <cassert>

using namespace std;
using namespace cv;

auto globalHack = chrono::high_resolution_clock::now();

// FOR TEST
chrono::duration<double> timeCamSuccess;
auto globalCamHack = chrono::high_resolution_clock::now();
auto timeLastCamSuccess = chrono::high_resolution_clock::now();

// true means video input from oCam, false means video input from file
bool fromCameraFlag;    // value set in main()

// FOR TEST
bool programTimingFlag {false}; // records 26 events per frame
int programTimingStartFrame {110}, programTimingEndFrame {};

bool savePaths;         	 // value set in main()
bool saveProbabilities; 	 // value set in main()
bool saveDroppedTrackStills; // value set in main()

string boxNumber {""}; // value set in main()

ofstream probabilitiesFile; // file to hold machine learning probabilities .csv
ifstream boxNumberFile("/home/odroid/boxnumber.txt"); // file to hold box number unique to this physical computer (for outgoing image metadata)

// set 1 to print time hacks, 0 to hide (will speed up program)
// set >1 to print time hacks every X frames
int printTimeHacksFlag {50};
bool printDetailedTimeHacksFlag {false}; // print time hacks for the OpenCV work done within threads (printing number of tracks and fps are not affected by this flag)

// how many auxiliary threads will do parallel image processing aside from main() thread and camera capture thread
int numberOfThreads;

atomic<bool> releaseVideoWriterNow {false};

mutex mutexCapture;	// mutex to lock camera or video file capture
mutex mutexCout; 	// FOR TEST timing

atomic<int> nextFrameNumForCapture {2}; // to ensure threads capture frames in sequence

int mainFrameNum {2}; // means main() will be processing Frame 2 next (Frame 1 was throw-away from above code)

int frameHeight, frameWidth;

int blurSize {3};				// how much each frame is blurred before MOG is applied (higher number erases bigger noise but may erase small targets; must be odd number)
double mogLearnRate {0.035};	// rate at which background Gaussians are updated (higher updates in shorter time)

// how much final targets are enlarged to fill in holes to make them single whole shapes
// note: dilating with kernel larger than 1 causes OpenCL resource error,
//       so to increase dilation, paste an additional dilation line in processFrame() and add an fgMaskD_U Mat;
//       this dilation variable should match number of dilation lines in processFrame(),
//       and its only purpose is to set a corresponding noise size for rejection
// note: larger dilation sizes will dramatically increase findContours time, lowering fps
int dilation {1};
Mat kernelDilate;

// FOR TEST track color identifier
int trackColor {1};

float globalMaxDistDetToTrack {150};//50 WORKED WELL // max allowable distance between detection and past track to allow that detection to be matched to that track
float minDistanceFromOrigin {7}; 	// minimum distance a track must be from its origin coordinates before it's declared "real" (as opposed to noise)
float minDistanceFromOriginSquared; // square of above number to save square root calculations later
int framesUntilRejected {6};		// Frames until a Track that isn't declared "real" is changed to Rejection area and dropped
int framesRemaining {300};			// Frames until a Rejection area is erased (i.e., area is allowed to form Detections again)
int updatesUntilRealTrack {3}; 		// number of updates required before track can be declared "real" (as opposed to noise)

// list of noise rejection areas that will be ignored
vector<vector<int>> rejectionList; 

// Caffe code copied almost entirely from caffe.berkeleyvision.org
class Classifier 
{
	public:
		Classifier( const string& model_file,
					const string& trained_file,
					const string& mean_file,
					const string& label_file);
		vector<float> Predict(const Mat& img);
		vector<string> labels_;
		
	private:
		void SetMean(const string& mean_file);
		void WrapInputLayer(vector<Mat>* input_channels);
		void Preprocess(const Mat& img, vector<Mat>* input_channels);
		shared_ptr<caffe::Net<float>> net_;
		Size input_geometry_;
		int num_channels_;
		Mat mean_;
};

Classifier::Classifier(const string& model_file, const string& trained_file, const string& mean_file, const string& label_file)
{
	caffe::Caffe::set_mode(caffe::Caffe::CPU);
	
	net_.reset(new caffe::Net<float>(model_file, caffe::TEST));
	net_->CopyTrainedLayersFrom(trained_file);
	
	CHECK_EQ(net_->num_inputs(), 1) << "Network should have exactly one input.";
	CHECK_EQ(net_->num_outputs(), 1) << "Network should have exactly one output.";
	
	caffe::Blob<float>* input_layer = net_->input_blobs()[0];
	num_channels_ = input_layer->channels();
	CHECK(num_channels_ == 3 || num_channels_ == 1) << "Input layer should have 1 or 3 channels.";
	input_geometry_ = Size(input_layer->width(), input_layer->height());
	
	SetMean(mean_file);
	
	ifstream labels(label_file.c_str());
	CHECK(labels) << "Unable to open labels file " << label_file;
	string line;
	while (getline(labels, line))
		labels_.push_back(string(line));
	
	caffe::Blob<float>* output_layer = net_->output_blobs()[0];
	CHECK_EQ(labels_.size(), output_layer->channels()) << "Number of labels is different from the output layer dimension.";
}

static bool PairCompare(const pair<float, int>& lhs, const pair<float, int>& rhs) {return lhs.first > rhs.first;}

// load mean file in binaryproto format
void Classifier::SetMean(const string& mean_file)
{
	caffe::BlobProto blob_proto;
	caffe::ReadProtoFromBinaryFileOrDie(mean_file.c_str(), &blob_proto);
	
	// convert from BlobProto to Blob<float>
	caffe::Blob<float> mean_blob;
	mean_blob.FromProto(blob_proto);
	CHECK_EQ(mean_blob.channels(), num_channels_) << "Number of channels of mean file doesn't match input layer.";
	
	// format of mean file is planar 32-bit float BGR or grayscale
	vector<Mat> channels;
	float* data = mean_blob.mutable_cpu_data();
	for (int i = 0; i < num_channels_; ++i)
	{
		// extract an individual channel
		Mat channel(mean_blob.height(), mean_blob.width(), CV_32FC1, data);
		channels.push_back(channel);
		data += mean_blob.height() * mean_blob.width();
	}
	
	// merge separate channels into single image
	Mat mean;
	merge(channels, mean);
	
	// compute global mean pixel value and create a mean image filled with this value
	Scalar channel_mean = cv::mean(mean) / 255; // scale from 0-255 to 0-1 to match pixel value range on which model was trained
	mean_ = Mat(input_geometry_, mean.type(), channel_mean);
}

vector<float> Classifier::Predict(const Mat& img)
{
	caffe::Blob<float>* input_layer = net_->input_blobs()[0];
	input_layer->Reshape(1, num_channels_, input_geometry_.height, input_geometry_.width);
	// forward dimension change to all layers
	net_->Reshape();
	
	vector<Mat> input_channels;
	WrapInputLayer(&input_channels);
	
	Preprocess(img, &input_channels);
	
	Vec3b bgrPixel = img.at<Vec3b>(41,41);
	
	net_->Forward();
	
	// copy output layer to a vector
	caffe::Blob<float>* output_layer = net_->output_blobs()[0];
	const float* begin = output_layer->cpu_data();
	const float* end = begin + output_layer->channels();
	
	return vector<float>(begin, end);
}

// wrap input layer of network in separate Mat objects (one per channel).
// this saves one memcpy operation and doesn't rely on cudaMemcpy2D.
// the last preprocessing operation will write the separate channels directly to input layer.
void Classifier:: WrapInputLayer(vector<Mat>* input_channels)
{
	caffe::Blob<float>* input_layer = net_->input_blobs()[0];
	
	int width = input_layer->width();
	int height = input_layer->height();
	float* input_data = input_layer->mutable_cpu_data();
	for (int i = 0; i < input_layer->channels(); ++i)
	{
		Mat channel(height, width, CV_32FC1, input_data);
		input_channels->push_back(channel);
		input_data += width * height;
	}
}

void Classifier::Preprocess(const Mat& img, vector<Mat>* input_channels)
{
	// convert pixel values from unsigned int to float
	Mat img_float;
	img.convertTo(img_float, CV_32FC3);
		
	// rescale from 0-255 to 0-1 to match range with which model was trained
	img_float = img_float / 255;
	
	// subtract image mean (as a single scalar value per color channel)
	subtract(img_float, mean_, img_float);
	
	// this operation will write separate BGR planes directly to input layer of network
	// because it is wrapped by the Mat objects in input_channels
	split(img_float, *input_channels);
	
	CHECK(reinterpret_cast<float*>(input_channels->at(0).data) == net_->input_blobs()[0]->cpu_data()) << "Input channels are not wrapping the input layer of the network.";
}

// initialize Caffe
string caffeDataPrefix = "/home/odroid/ddd/caffe/data/";
string label_file = caffeDataPrefix + "categories.txt";

// cropped still images
string model_file_still = caffeDataPrefix + "ddd_cropped/ddd_cropped_deploy.prototxt";
string trained_file_still = caffeDataPrefix + "ddd_cropped/ddd_cropped_iter_68000.caffemodel";
string mean_file_still = caffeDataPrefix + "ddd_cropped/ddd_cropped_train_lmdb_mean_image.binaryproto";
Classifier classifierStill(model_file_still, trained_file_still, mean_file_still, label_file);

// Track path shapes
string model_file_path = caffeDataPrefix + "ddd_path/ddd_path_4x_107px_4min_deploy.prototxt";
string trained_file_path = caffeDataPrefix + "ddd_path/ddd_path_4x_107px_4min_iter_18000.caffemodel";
string mean_file_path = caffeDataPrefix + "ddd_path/ddd_path_4x_107px_4min_train_lmdb_mean_image.binaryproto";
Classifier classifierPath(model_file_path, trained_file_path, mean_file_path, label_file);

class Track
{
	public:
		char trackName[22]; 			// Track name: t + origin date and time YYYYMMDDHHMMSSdddddd where "d" is microseconds (e.g., t20171112123456000001)
		timeval timeLastUpdated; 		// time track was last updated with a new detection
		int frameLastUpdated {mainFrameNum}; // frame last updated
		bool isRealTrack {false}; 		// flag for whether track has met defined criteria to be considered real (not noise)
		Mat stillImage; 				// most recent still image (pixel values 0-255)
		vector<int> originCoordinates; 	// [y, x] coordinates of Track when it was created (used to declare it a "real" track when it moves far enough from origin)
		float y0pred {0}, x0pred {0}; 	// predicted value of next detection location (like a 1-frame temporary track coast)
		Mat trackPathImage; 			// Track path image (frame_height x frame_width, 0 black value = background, 255 white value = latest track, fading gray trail = history)
		Mat trackPathCrop; 				// cropped area of overall track path image that contains latest track path
		float trackPathScale {4};		// scaling factor from whole frame to track path image
		int yMin, yMax, xMin, xMax, cropTop, cropBottom, cropLeft, cropRight; // coordinates for cropping track path image (in half-scale)
		int trackPathCropSize {107}; 	// pixel size of track path crop square
		float maxDistDetToTrack {globalMaxDistDetToTrack}; // max distance a Detection can match to predicted next Track position (dynamic number)
		float maxDistDetToTrackCoastLimit {50}; 	  // max allowable miss distance due to coasting Track (to keep it from matching far-away noise)
		float ySpeed {0}, xSpeed {0}, totalSpeed {0}; // speed (in pixels/30th-of-a-second) of Track smoothed over last few Frames
		
		// FOR TEST to plot allowable match area from last Frame
		float y0predOld {0}, x0predOld {0}, maxDistDetToTrackOld {globalMaxDistDetToTrack};
		
		// to adjust allowable match distance and to plot latest match distance to a Detection
		float matchDistance {3000};
		
		// FOR TEST each track gets its own color
		int blue {255};
		int green {255};
		int red {255};
		
		// last update (not coast) position (for Track path image to ignore coasting)
		int lastUpdatedY {0}, lastUpdatedX {0};
		
		// last update (not coast) speed (for saved image metadata)
		float lastUpdatedYSpeed {0}, lastUpdatedXSpeed {0}, lastUpdatedTotalSpeed {0};
		
		// recent Track history coordinates
		// [frames ago][y, x]
		// -1 is considered an invalid entry (intentionally)
		vector<vector<int>> trackHistoryCoordinates
		{
			{-1,-1},
			{-1,-1},
			{-1,-1},
			{-1,-1},
			{-1,-1}
		};
		
		// time differences (in units of 1/30ths of seconds) associated with timeHistoryCoordinates
		vector<float> trackHistoryTimeDiffs
		{
			{1},
			{1},
			{1},
			{1},
			{1}
		};
		
		vector<float> trackCatProbabilities {0, 0, 0, 0, 0, 0, 0}; // vector of probabilities of each category from machine learning
		float catProbabilitiesBlendRate {5}; 	// how much new categorization is divided before blending with historical categorization (higher number means slower adoption of new categorization)
		float trackCatProbability {0};			// category number for max probability among machine learning categories
		string trackCatLabel {""}; 				// text label corresponding to max probability among machine learning categories
		
		// FOR TEST
		vector<float> stillCatProbabilities {0, 0, 0, 0, 0, 0, 0}; // vector of probabilities of each category from machine learning
		float stillCatProbability {0};
		string stillCatLabel {""};
		vector<float> pathCatProbabilities {0, 0, 0, 0, 0, 0, 0}; // vector of probabilities of each category from machine learning
		float pathCatProbability {0};
		string pathCatLabel {""};
		
		// FOR TEST to optimize machine learning via a .csv in Excel
		// stores machine learning probabilities for each updated Frame of Track's life
		// A is aircraftfixed, B is bird, D is dronerotor
		string stillA {""}, stillB{""}, stillD{""}, pathA {""}, pathB {""}, pathD {""}, totalA {""}, totalB {""}, totalD {""}, totalO {""};
		
		int coastCounter {0}; 		// how many consecutive Frames the Track has been coasted (is reset each time track gets an update from an actual detection)
		int totalTrackFrames {1}; 	// how many Frames Track has existed
		int totalTrackUpdates {1}; 	// how many times Track has been updated by Detections (throughout its life); used for minimum hits before calling machine learning for track path image
		
		void updateTrack (int, int, Mat *, timeval, float);
		void coastTrack (float);
		void saveBeforeDropTrack ();
		
		Track (int y, int x, Mat * newStillImage, timeval timeCaptured, float frameTimeDiff)
		: stillImage(*newStillImage), timeLastUpdated(timeCaptured), lastUpdatedY(y), lastUpdatedX(x)
		{
			//set track name and time last updated (same as time created)
			int microSeconds = timeLastUpdated.tv_usec;
			char timeYYYYmmddHHMMSS [15];
			strftime(timeYYYYmmddHHMMSS, 15, "%Y%m%d%H%M%S", localtime(&timeLastUpdated.tv_sec));

			// track name (unique each nanosecond)
			sprintf(trackName, "t%s%06d", timeYYYYmmddHHMMSS, microSeconds);
			
			// track history coordinates
			updateTrackCoordinates(y, x, frameTimeDiff);
	
			// initalize track path image to black the size of video frame
			// note: half-scale image to save computing power in machine learning categorization since tracks likely jump at least 2 pixels between frames
			trackPathImage = Mat::zeros(Size(round(frameWidth / trackPathScale), round(frameHeight / trackPathScale)), CV_8UC1);
			
			// set coordinates of track at its origin 
			// (to be able to watch for it leaving its origin, indicating it's not noise)
			originCoordinates.emplace_back(y);
			originCoordinates.emplace_back(x);
			
			// set initial min and max x and y values (in half-scale)
			yMin = round(y / trackPathScale);
			yMax = round(y / trackPathScale);
			xMin = round(x / trackPathScale);
			xMax = round(x / trackPathScale);
			
			// FOR TEST set unique-ish Track color
			switch (trackColor)
			{
				case 1 : blue = 48;  green = 130; red = 255; break; // orange
				case 2 : blue = 0;   green = 127; red = 0;   break; // green
				case 3 : blue = 127; green = 127; red = 0;   break; // teal
				case 4 : blue = 127; green = 0;   red = 127; break; // purple
				case 5 : blue = 0;   green = 0;   red = 127; break; // maroon
				case 6 : blue = 0;   green = 255; red = 0;   break; // lime
				case 7 : blue = 255; green = 127; red = 0;   break; // blue
				case 8 : blue = 255; green = 0;   red = 255; break; // fuchsia
				default: blue = 255; green = 255; red = 255;        // white
			}
			trackColor++; // increment global trackColor
			if (trackColor >= 9) {trackColor = 1;} // set back to 1 if exceeded 8
			
			// FOR TEST
			if (saveProbabilities)
			{
				stillA += to_string(mainFrameNum) + ",SA";
				stillB += to_string(mainFrameNum) + ",SB";
				stillD += to_string(mainFrameNum) + ",SD";
				pathA += to_string(mainFrameNum) + ",PA";
				pathB += to_string(mainFrameNum) + ",PB";
				pathD += to_string(mainFrameNum) + ",PD";
				totalA += to_string(mainFrameNum) + ",TA";
				totalB += to_string(mainFrameNum) + ",TB";
				totalD += to_string(mainFrameNum) + ",TD";
				totalO += to_string(mainFrameNum) + ",TO";
			}
		};
		
	private:
		float weightStill {1}; // multiplier for importance of still image machine learning categorization
		float weightPath {0.5};  // multiplier for importance of path shape machine learning categorization
		
		void updateTrackCoordinates (int, int, float);
		void updateTrackPath ();
		void updateCategorization (Classifier *, Mat, float, string);
		void updateSpeed ();
		void updateMaxDistDetToTrack (float);
};

void Track::updateSpeed() 
{
	// case: Track history has only 1 sets of coordinates to be used for calculating speed
	if (trackHistoryCoordinates[3][0] == -1)
	{
		ySpeed = 0;
		xSpeed = 0;
	}
	else if (trackHistoryCoordinates[2][0] == -1)
	{
		// case: Track history has 2 sets of coordinates to be used for calculating speed
		// (adjusted for variable frame rate via division by timeDiffs)
		ySpeed = (float(trackHistoryCoordinates[4][0]) - 
				  float(trackHistoryCoordinates[3][0])) / 
				  trackHistoryTimeDiffs[4];
		xSpeed = (float(trackHistoryCoordinates[4][1]) - 
				  float(trackHistoryCoordinates[3][1])) / 
				  trackHistoryTimeDiffs[4];
	}
	else //if (trackHistoryCoordinates[1][0] == -1)
	{
		// case: Track history has 3 sets of coordinates to be combined to calculate speed
		// take average of vectors from x0-x2, x0-x1, x1-x2 (x1 cancels out)
		// (adjusted for variable frame rate via division by timeDiffs)
		ySpeed = (float(trackHistoryCoordinates[4][0]) - 
				  float(trackHistoryCoordinates[2][0])) / 
				  (2 * (trackHistoryTimeDiffs[4] + trackHistoryTimeDiffs[3]));
		xSpeed = (float(trackHistoryCoordinates[4][1]) - 
				  float(trackHistoryCoordinates[2][1])) / 
				  (2 * (trackHistoryTimeDiffs[4] + trackHistoryTimeDiffs[3]));
	}
	
	totalSpeed = pow(pow(ySpeed, 2) + pow(xSpeed, 2), 0.5);
}

void Track::updateMaxDistDetToTrack(float frameTimeDiff)
{
	// adjust allowable match distances based on accuracy of Track prediction
	if (coastCounter == 0) 
	{
		// if Track matched a Detection, grow or shrink allowable match distance
		//    based on how far Detection was from predicted Track location
		// to quickly adjust to target acceleration/maneuvers, 
		//    add to or subtract from allowable match distance by 
		//    a multiple of the difference between Detection-to-Track-prediction and 
		//    half of previous allowable miss distance
		// grow allowable miss distance quickly (2x difference), but shrink it slowly (1x difference)
		// this will attempt to ensure next Frame's Detection lies at approximately
		//    half of allowable match distance so that Track gets that update
		// also, lessen growth/shrinkage if Frame rate was < 30 fps
		//    (i.e., frameTimeDiff > 1), as this delay in capturing most recent Frame
		//    explains some of reason for additional inaccuracy in Track prediction
		if (matchDistance > (maxDistDetToTrack / 2)) {maxDistDetToTrack += (2 / frameTimeDiff) * (matchDistance - maxDistDetToTrack / 2);}
		else {maxDistDetToTrack += (1 / frameTimeDiff) * (matchDistance - maxDistDetToTrack / 2);}
	}
	else 
	{
		// increase allowable match distance for coasting Tracks
		// don't grow allowable match distance above 50;
		//    this keeps coasting Track from matching far-away noise
		maxDistDetToTrack = fmin(1.25 * maxDistDetToTrack, maxDistDetToTrackCoastLimit); //totalSpeed * 0.0024 + 0.71;
	}
	
	// keep allowable match distance within limits
	// from observing actual Tracks, most low speed Tracks (~4 pixels/Frame)
	//    are within ~2 pixels of prediction, but some are ~6 pixels off; 
	//    ~15 pixels for 0 pixels/Frame targetsis a comfortable margin above that
	//    to ensure new Detection isn't rejected from a Track and allowed to form 
	//    an erroneous new Track
	// 30 pixels seemed like good lower limit for high speed Tracks (~50 pixels/Frame)
	// above 2 notes result in lower limit equation based on speed
	// upper limit equation yields
	//    50 pixel allowable match distance for high speed targets (~50 pixels/Frame) 
	//    and 20 pixel allowable match distance for 0 speed targets;
	//    this keeps far away noise from matching to Track
	if (maxDistDetToTrack > totalSpeed * 0.6 + 20) {maxDistDetToTrack = totalSpeed * 0.6 + 20;}
	else if (maxDistDetToTrack < (totalSpeed * 0.3 + 15)) {maxDistDetToTrack = totalSpeed * 0.3 + 15;}
	
	// specifically target possible noise
	// if a Track has at least 1 update after creation, speed is small, 
	//    and it hasn't been declared a "real" Track yet,
	//    set small allowable match distance so that it will most likely flag as noise
	//    and cause a rejection area to be created
	if ((totalTrackUpdates > 1) && (totalSpeed <= 1) && !(isRealTrack)) {maxDistDetToTrack = minDistanceFromOrigin - 1;}
}

void Track::updateTrackCoordinates(int y, int x, float frameTimeDiff)
{
	// remove oldest track position coordinates from 0 index
	trackHistoryCoordinates.erase(trackHistoryCoordinates.begin());
	trackHistoryTimeDiffs.erase(trackHistoryTimeDiffs.begin());
	
	// append newest track position coordinates at highest index
	vector<int> newCoordinates {y, x};
	trackHistoryCoordinates.emplace_back(newCoordinates);
	trackHistoryTimeDiffs.emplace_back(frameTimeDiff);
	
	// adjust min and max y and x as required (in half-scale) (for cropping track path image to include latest track path)
	if (round(y / trackPathScale) < yMin) 
	{
		yMin = round(y / trackPathScale);
		
		if ((yMax - yMin) > trackPathCropSize)
		{
			yMax = yMin + trackPathCropSize; // discard old yMax coordinates that are too far from current Track position
		}
	}
	if (round(y / trackPathScale) > yMax) 
	{
		yMax = round(y / trackPathScale);
		
		if ((yMax - yMin) > trackPathCropSize)
		{
			yMin = yMax - trackPathCropSize; // discard old yMin coordinates that are too far from current Track position
		}
	}
	if (round(x / trackPathScale) < xMin) 
	{
		xMin = round(x / trackPathScale);
		
		if ((xMax - xMin) > trackPathCropSize)
		{
			xMax = xMin + trackPathCropSize; // discard old xMax coordinates that are too far from current Track position
		}
	}
	if (round(x / trackPathScale) > xMax) 
	{
		xMax = round(x / trackPathScale);
		
		if ((xMax - xMin) > trackPathCropSize)
		{
			xMin = xMax - trackPathCropSize; // discard old xMin coordinates that are too far from current Track position
		}
	}
}

void Track::updateTrackPath()
{
	// draw line from last coordinates to latest coordinates
	// note: drawn on a half-scale image to save computing power in machine learning categorization since tracks likely jump at least 2 pixels between frames
	// "255" is white line color
	// "1" is pixels of thickness
	// "CV_AA" is an anti-aliased line so that machine learning reacts less to jagged pixels
	// "0" denotes zero decimal digits in point coordinates (they are integers)
	line(trackPathImage, Point(round(lastUpdatedX / trackPathScale), round(lastUpdatedY / trackPathScale)), Point(round(trackHistoryCoordinates[4][1] / trackPathScale), round(trackHistoryCoordinates[4][0] / trackPathScale)), Scalar(255), 1, CV_AA, 0);
	
	// take crop of overall track path image to include latest part of track path (saves machine learning computing power)
	cropTop = yMin + round((yMax - yMin) / 2)  - round(trackPathCropSize / 2);
	if (cropTop < 0) {cropTop = 0;} // don't let crop go off top of track path image
	if (cropTop > (round(frameHeight / trackPathScale) - trackPathCropSize)) {cropTop = round(frameHeight / trackPathScale) - trackPathCropSize;} // don't let crop go off bottom of track path image
	cropBottom = cropTop + trackPathCropSize;

	cropLeft = xMin + round((xMax - xMin) / 2)  - round(trackPathCropSize / 2);
	if (cropLeft < 0) {cropLeft = 0;} // don't let crop go off left side of track path image
	if (cropLeft > (round(frameWidth / trackPathScale) - trackPathCropSize)) {cropLeft = round(frameWidth / trackPathScale) - trackPathCropSize;} // don't let crop go off right side of track path image
	cropRight = cropLeft + trackPathCropSize;
	
	trackPathCrop = trackPathImage(Rect(cropLeft, cropTop, cropRight - cropLeft, cropBottom - cropTop));

	// save last updated coordinates for future Track path image connecting line incase Track beings coasting
	lastUpdatedY = trackHistoryCoordinates[4][0];
	lastUpdatedX = trackHistoryCoordinates[4][1];
}

void Track::updateCategorization(Classifier * classifier, Mat image, float weight, string type)
{
	// get still image categorization probabilities from machine learning
	vector<float> newCatProbabilities = classifier->Predict(image);
	
	// FOR TEST
	if (type == "still")
	{
		// save whole probability vector
		stillCatProbabilities = newCatProbabilities;
		
		// save highest probability and its label
		int maxStillProbabilityIndex = 0;
		stillCatProbability = newCatProbabilities[0];
		for (int i = 1; i < newCatProbabilities.size(); i++)
		{
			if (newCatProbabilities[i] > stillCatProbability)
			{
				maxStillProbabilityIndex = i;
				stillCatProbability = newCatProbabilities[i];
			}
		}
		stillCatLabel = classifier->labels_[maxStillProbabilityIndex];
	}
	else
	{
		// save whole probability vector
		pathCatProbabilities = newCatProbabilities;
		
		// save highest probability and its label
		int maxPathProbabilityIndex = 0;
		pathCatProbability = newCatProbabilities[0];
		for (int i = 1; i < newCatProbabilities.size(); i++)
		{
			if (newCatProbabilities[i] > pathCatProbability)
			{
				maxPathProbabilityIndex = i;
				pathCatProbability = newCatProbabilities[i];
			}
		}
		pathCatLabel = classifier->labels_[maxPathProbabilityIndex];
	}
	
	// only allow path machine learning to be included in total if it meets certain probability tresholds that depend on total Track updates
	// these minimum probabilities were determined experimentally and represent a level above which categorization is almost always correct
	if 
	(
		(type != "path") // allow all still image machine learning to be included in total
		||
		(
			(type == "path")
			&&
			(
				((totalTrackUpdates <= 13) && (pathCatProbability > 0.78))
				||
				((totalTrackUpdates >= 14) && (pathCatProbability > 0.76))
				||
				((totalTrackUpdates >= 15) && (pathCatProbability > 0.66))
				||
				((totalTrackUpdates >= 17) && (pathCatProbability > 0.59))
				||
				((totalTrackUpdates == 20) && (pathCatProbability > 0.55))
				||
				(totalTrackUpdates >= 21)
			)
		)
	)
	{
		float totalCatProbability = 0; // sum of categorization vector probabilities for use in re-scaling it
	
		for (int i = 0; i < trackCatProbabilities.size(); i++)
		{
			// blend new categorization into Track's historical categorization by adding 
			trackCatProbabilities[i] += newCatProbabilities[i] * weight / catProbabilitiesBlendRate;
		
			// update total of vector's probabilities
			totalCatProbability += trackCatProbabilities[i];
		}
	
		for (int i = 0; i < trackCatProbabilities.size(); i++)
		{
			// scale vector elements to add up to 100% total probability
			trackCatProbabilities[i] /= totalCatProbability;
		}
	
		// save highest probability and its label
		int maxProbabilityIndex = 0;
		trackCatProbability = trackCatProbabilities[0];
		for (int i = 1; i < trackCatProbabilities.size(); i++)
		{
			if (trackCatProbabilities[i] > trackCatProbability)
			{
				maxProbabilityIndex = i;
				trackCatProbability = trackCatProbabilities[i];
			}
		}
		trackCatLabel = classifier->labels_[maxProbabilityIndex];
	}
	
	// FOR TEST
	/*cout << "Frame=" << mainFrameNum << " Updates=" << totalTrackUpdates << 
		"\t0=" << newCatProbabilities[0] << 
		"\t1=" << newCatProbabilities[1] << 
		"\t2=" << newCatProbabilities[2] << 
		"\t3=" << newCatProbabilities[3] << 
		"\t4=" << newCatProbabilities[4] << 
		"\t5=" << newCatProbabilities[5] << 
		"\t6=" << newCatProbabilities[6] << endl;*/
}

void Track::updateTrack(int y, int x, Mat * newStillImage, timeval timeCaptured, float frameTimeDiff)
{
	// increment total number of Frames Track has existed
	totalTrackFrames++;
	
	// increment number of times Track has been updated by Detections
	// (used for when to call machine learning for track path image)
	totalTrackUpdates++;
	
	// time last updated
	timeLastUpdated = timeCaptured;
	
	// frame last updated
	frameLastUpdated = mainFrameNum;
	
	// still image
	stillImage = *newStillImage;
	
	// track history coordinates
	updateTrackCoordinates(y, x, frameTimeDiff);
	
	// track path image
	updateTrackPath();
	
	// if track not yet labeled "real", label it real if there have been enough updates and it has moved far enough from its origin coordinates
	if ((!isRealTrack) && (totalTrackUpdates > updatesUntilRealTrack))
	{
		//CALCULATE DISTANCE OF UPDATE FROM TRACK ORIGIN COORDINATES
		if (pow(trackHistoryCoordinates[4][0] - originCoordinates[0], 2) + pow(trackHistoryCoordinates[4][1] - originCoordinates[1], 2) > minDistanceFromOriginSquared)
		{
			isRealTrack = true;
		}
	}
	
	// still image machine learning categorization starts on 2nd Track update
	// (this avoids waisting time on spurious single hits, 
	//   while building quality categorization ASAP)
	if (totalTrackUpdates >= 2) {updateCategorization(&classifierStill, stillImage, weightStill, "still");}
	
	// Track path image machine learning categorization starts on 7th Track update 
	// (determined experimentally as minimum at which path categorization begins to be reliable
	if (totalTrackUpdates >= 7) {updateCategorization(&classifierPath, trackPathCrop, weightPath, "path");}
	
	// FOR TEST
	if ((saveProbabilities) && (totalTrackUpdates >= 4) && (totalTrackUpdates <= 25))
	{
		stillA += "," + to_string(stillCatProbabilities[1] * 100);
		stillB += "," + to_string(stillCatProbabilities[3] * 100);
		stillD += "," + to_string(stillCatProbabilities[6] * 100);
		pathA += "," + to_string(pathCatProbabilities[1] * 100);
		pathB += "," + to_string(pathCatProbabilities[3] * 100);
		pathD += "," + to_string(pathCatProbabilities[6] * 100);
	}
	
	// zeroize coast counter (0 frames are now coasted since this frame is updated)
	coastCounter = 0;
	
	// update Track speed
	updateSpeed();
	lastUpdatedYSpeed = ySpeed;
	lastUpdatedXSpeed = xSpeed;
	lastUpdatedTotalSpeed = totalSpeed;
	
	// decrease allowable match distance (to tighten around track and keep it from straying onto future noise)
	updateMaxDistDetToTrack(frameTimeDiff);
	
	// FOR TEST saving path images
	if (isRealTrack && savePaths)
	{
		string saveImagePath = "/home/odroid/ddd/caffe/data/saved_paths/";
		string catLabel = "dronerotor";
	
		char numberOfFrames[5];
		string numberOfFramesStr;
		sprintf(numberOfFrames, "%04d", totalTrackFrames);
		stringstream ss;
		ss << numberOfFrames;
		ss >> numberOfFramesStr;
		
		char frameNumChar[5];
		string frameNumStr;
		sprintf(frameNumChar, "%04d", mainFrameNum);
		stringstream ssFrameNum;
		ssFrameNum << frameNumChar;
		ssFrameNum >> frameNumStr;
	
		stringstream ss2;
		string trackNameStr;
		ss2 << trackName;
		ss2 >> trackNameStr;
	
		string imagePathAndFileName = saveImagePath + catLabel + "/" + catLabel + "_" + numberOfFramesStr + "_" + frameNumStr + "_" + trackNameStr + ".png";
	
		imwrite(imagePathAndFileName, trackPathCrop);
	}
};

void Track::coastTrack(float frameTimeDiff)
{
	// increment total number of Frames Track has existed
	// (used for when to call machine learning for track path image)
	totalTrackFrames++;
	
	// Track history coordinates (convert predicted position floats to ints)
	updateTrackCoordinates((int)(round(y0pred)), (int)(round(x0pred)), frameTimeDiff);
	
	// increment coast counter
	coastCounter++;
	
	// update Track speed
	updateSpeed();
	
	// increase allowable match distance
	updateMaxDistDetToTrack(frameTimeDiff);
}

void Track::saveBeforeDropTrack()
{
	if (isRealTrack && saveDroppedTrackStills)
	{
		// save Track's last still image
		stringstream ssTrackName;
		string trackNameStr;
		ssTrackName << trackName;
		ssTrackName >> trackNameStr;
		string imagePathAndFileName = "/home/odroid/ddd/caffe/data/metadata_stills/" + trackNameStr + ".png";
		imwrite(imagePathAndFileName, stillImage);
		
		// generate metadata
		Exiv2::ExifData exifData; // create exif metadata container
		exifData["Exif.Image.ImageID"] = trackName;							// ascii-type metadata
		exifData["Exif.Image.TargetPrinter"] = lastUpdatedY; 				// ascii-type metadata
		exifData["Exif.Image.ImageHistory"] = lastUpdatedX; 				// ascii-type metadata
		exifData["Exif.Image.Make"] = to_string(lastUpdatedYSpeed);			// ascii-type metadata
		exifData["Exif.Image.Model"] = to_string(lastUpdatedXSpeed);		// ascii-type metadata
		exifData["Exif.Image.Software"] = to_string(lastUpdatedTotalSpeed);	// ascii-type metadata
		exifData["Exif.Image.ImageDescription"] = trackCatLabel; 			// ascii-type metadata
		exifData["Exif.Image.Artist"] = to_string(trackCatProbability); 	// ascii-type metadata
		exifData["Exif.Image.SpectralSensitivity"] = coastCounter;			// ascii-type metadata
		exifData["Exif.Image.CameraSerialNumber"] = boxNumber;				// ascii-type metadata
		
		int microSeconds = timeLastUpdated.tv_usec;
		char timeYYYYmmddHHMMSS [15];
		char timeLastUpdatedChar [21];
		strftime(timeYYYYmmddHHMMSS, 15, "%Y%m%d%H%M%S", localtime(&timeLastUpdated.tv_sec));
		sprintf(timeLastUpdatedChar, "%s%06d", timeYYYYmmddHHMMSS, microSeconds);
		exifData["Exif.Image.DateTimeOriginal"] = timeLastUpdatedChar;		// ascii-type metadata
		
		// save metadata to saved Track still image
		Exiv2::Image::AutoPtr image = Exiv2::ImageFactory::open(imagePathAndFileName);
		assert(image.get() != 0);
		image->setExifData(exifData);
		image->writeMetadata();
	}
	
	if (saveProbabilities)
	{
		probabilitiesFile.open("/home/odroid/ddd/caffe/data/probabilities.csv", ios_base::out | ios_base::app);
		probabilitiesFile << 
			stillA << "\n" << 
			stillB << "\n" << 
			stillD << "\n" << 
			pathA << "\n" << 
			pathB << "\n" << 
			pathD << "\n" << 
			totalA << "\n" << 
			totalB << "\n" << 
			totalD << "\n" << 
			totalO << "\n";
		probabilitiesFile.close();
	}
}

class Detection
{
	public:
		int y, x;
		Mat stillImage;
		bool isMatched {false};
		
		Detection (int yCenter, int xCenter, Mat * croppedFrame)
		: y(yCenter), x(xCenter), stillImage(*croppedFrame)
		{};
};

// find minimum value in array (used to find next closest match of a Detection and Track in distances array)
tuple<float, int, int> arrayMin(float *arr, int rows, int cols)
{
	int row, col, minRow = 0, minCol = 0;
	float minValue = *arr;
	
	for (row = 0; row < rows; row++)
	{
		for (col = 0; col < cols; col++)
		{
			if (*((arr + row*cols) + col) < minValue)
			{
				minValue = *((arr + row*cols) + col);
				minRow = row;
				minCol = col;
			}
		}
	}
	
	return make_tuple(minValue, minRow, minCol);
}

// method to set a distances array row and column to high values so that the corresponding Detection and Track will not form additional matches
void maxOutRowAndCol(float *arr, int rows, int cols, int rowToMax, int colToMax, int frameHeightPlusWidth)
{
	int row, col;
	
	for (row = 0; row < rows; row++)
	{
		*((arr + row*cols) + colToMax) = frameHeightPlusWidth;
	}
	
	for (col = 0; col < cols; col++)
	{
		*((arr + rowToMax*cols) + col) = frameHeightPlusWidth;
	}
}

// FOR TEST
int noDataCounter {0};
bool noDataCounterTrip {false};

// oCam (code largely copied from withrobot github)
class Camera
{
    int cam_id;
    int cam_width;
    int cam_height;

    VideoCapture captureCamera;

	public:
		// Camera constructor
		// id : camera id (system dependent value)
		// w  : streaming image width
		// h  : streaming image height
		// fps: streaming image rate (frame per second)
		// buffer_size: This class internal buffer size (No camera internal buffer size.)
     
	    Camera(const int id, const int w=1280, const int h=720, const double fps=15)
			: cam_id(id), cam_width(w), cam_height(h)
		{
			captureCamera.open(id);// + CAP_FFMPEG); // note: CAP_FFMPEG is 1900
			if (captureCamera.isOpened()) 
			{
				captureCamera.set(CV_CAP_PROP_FRAME_WIDTH, static_cast<double>(w));
				captureCamera.set(CV_CAP_PROP_FRAME_HEIGHT, static_cast<double>(h));
				captureCamera.set(CV_CAP_PROP_FPS, fps);
				cout << "Cam #" << id << " is opened.\n";
	        }
	        else {cout << "Cam #" << id << " cannot be opened.\n";}
		}

	    ~Camera() {captureCamera.release();}
	    
	    // Get an image
	    bool get_image(UMat& img) 
	    {
	        bool retval = false;
	        
	        captureCamera.read(img);
	        
	        if (!img.empty())
	        {
	        	retval = true;
	        	timeLastCamSuccess = chrono::high_resolution_clock::now();
	        }
	
	        return retval;
	    }
};

class Frame
{
	public:
		atomic<bool> frameReadyForMain {false}; // bool to keep main() from grabbing Frame until thread is done with it
		timeval timeCaptured; // time frame was captured, which will become time Track was last updated
		vector<vector<Point>> conts; // array of coordinates of outline of each Detection
		vector<Vec4i> hierarchy; // throw-away variable for findContours
		
		// OpenCL GPU-aware versions of frame images and motion masks
		UMat frameRawUMat;
		UMat frameBlurUMat;
		UMat maskMOGUMat;
		UMat maskThresholdUMat;
		UMat maskDilateUMat;
		
		// Regular Mat for cropping for machine learning input
		Mat frameRawMat;
};

// vector to hold Frame pointers for each thread
vector<unique_ptr<Frame>> frameList;

class ProgramTimingEntry
{
	public:
		int thread {0}, frame {0}, step {0};
		float time {0};
		
		ProgramTimingEntry (int thread, int frame, int step, float time)
		: thread(thread), frame(frame), step(step), time(time)
		{};
};

// FOR TEST
vector<ProgramTimingEntry> programTiming;
mutex mutexProgramTiming;

// process each frame for main() thread
void processFrame(int threadNum, Camera * ocam, VideoCapture * captureVideoFile)
{
	// create MOG2 subtractor object for this thread so that it doesn't conflict with other threads
	// if using mogLearnRate = -1, then createBackgroundSubtractorMOG2(int history, float varThreshold, bool bShadowDetection)
	//   history setting is superseded by MOG_learn_rate
	//   varThreshold setting was ignored since it didn't seem to affect noise levels given other settings used
	//     sets the squared Mahalanobis distance from each pixel to modeled background to see if it should be considered background
	//     ~15 was high enough to reduce noise but not so high as to reject real targets
	//   shadows are not applicable to these data, so bShadowDetection setting is set to false
	Ptr<BackgroundSubtractorMOG2> MOG2 = createBackgroundSubtractorMOG2();
	
	// first frameNum thread will capture (e.g., frameNum 1 is a throw-away for main() setup, so thread 0 will first capture frameNum 2)
	int nextFrameNumForThisThread = threadNum + 2;
	
	// create capture lock but wait to lock it
	unique_lock<mutex> lockCapture(mutexCapture, defer_lock); 
	
	// FOR TEST timing
	unique_lock<mutex> lockCout(mutexCout, defer_lock);
	unique_lock<mutex> lockProgramTiming(mutexProgramTiming, defer_lock);
	
	vector<float> programTimingEntry {};
	programTimingEntry.reserve(4);
	
	// create mutex lock for rejectionList
	//unique_lock<mutex> lockRejectionList(mutexRejectionList, defer_lock);
	
	// FOR TEST timing
	auto hack = chrono::high_resolution_clock::now();
	
	// FOR TEST timing
	chrono::duration<double> timeWaitForMain, timeBlur, timeWaitForFrameNum;
	chrono::duration<double> timeMOG2, timeWaitForCapture, timeThreshold;
	chrono::duration<double> timeCapture, timeDilate, timeCopyToMat, timeContours;
	chrono::duration<double> currentTime, timeSinceCamSuccess;
	int timeDelay;

	// FOR TEST timing
	float timeWaitForMainSum = 0, timeBlurSum = 0, timeWaitForFrameNumSum = 0;
	float timeMOG2Sum = 0, timeWaitForCaptureSum = 0, timeThresholdSum = 0;
	float timeCaptureSum = 0, timeDilateSum = 0, timeCopyToMatSum = 0, timeContoursSum = 0;
	
	while (1) // loop forever
	{
		// FOR TEST timing
		// hack clocks for timing parts of program
		if (printTimeHacksFlag > 0) {hack = chrono::high_resolution_clock::now();}
		
		// wait until this thread's bool is flipped to false by main() (will already be false when thread is first spawned)
		while (frameList[threadNum]->frameReadyForMain) {usleep(100);};
		
		// FOR TEST timing
		if (printTimeHacksFlag > 0) {timeWaitForMain = chrono::high_resolution_clock::now() - hack; timeWaitForMainSum += timeWaitForMain.count(); hack = chrono::high_resolution_clock::now();}
		if (programTimingFlag && nextFrameNumForThisThread >= programTimingStartFrame && nextFrameNumForThisThread <= programTimingEndFrame)
		{
			currentTime = chrono::high_resolution_clock::now() - globalHack;
			ProgramTimingEntry entry(threadNum, nextFrameNumForThisThread, 1, currentTime.count());
			lockProgramTiming.lock();
			programTiming.emplace_back(entry);
			lockProgramTiming.unlock();
		}
		
		usleep(28000); // sleep thread during inevitable delay for capture so that other threads can use this processing power
		//cout << "Thread " << threadNum << " frame " << nextFrameNumForThisThread << " waiting for proper frame number to come up." << endl;
		while(nextFrameNumForCapture < nextFrameNumForThisThread) {usleep(100);} // wait until it's time to capture correct frame in sequence
		//cout << "Thread " << threadNum << " frame " << nextFrameNumForThisThread << " waiting on capture lock." << endl;
		
		// FOR TEST timing
		if (printTimeHacksFlag > 0) {timeWaitForFrameNum = chrono::high_resolution_clock::now() - hack; timeWaitForFrameNumSum += timeWaitForFrameNum.count(); hack = chrono::high_resolution_clock::now();}
		if (programTimingFlag && nextFrameNumForThisThread >= programTimingStartFrame && nextFrameNumForThisThread <= programTimingEndFrame)
		{
			currentTime = chrono::high_resolution_clock::now() - globalHack;
			ProgramTimingEntry entry(threadNum, nextFrameNumForThisThread, 2, currentTime.count());
			lockProgramTiming.lock();
			programTiming.emplace_back(entry);
			lockProgramTiming.unlock();
		}
		
		timeSinceCamSuccess = chrono::high_resolution_clock::now() - timeLastCamSuccess;
		timeDelay = 33000 - timeSinceCamSuccess.count() * 1000000;
		if (timeDelay > 0) {usleep(timeDelay);}
		
		// lock capture of next frame
		lockCapture.lock();
		
		//cout << "Thread " << threadNum << " frame " << nextFrameNumForThisThread << " proceeding to capture." << endl;
		
		// FOR TEST timing
		if (printTimeHacksFlag > 0) {timeWaitForCapture = chrono::high_resolution_clock::now() - hack; timeWaitForCaptureSum += timeWaitForCapture.count(); hack = chrono::high_resolution_clock::now();}
		if (programTimingFlag && nextFrameNumForThisThread >= programTimingStartFrame && nextFrameNumForThisThread <= programTimingEndFrame)
		{
			currentTime = chrono::high_resolution_clock::now() - globalHack;
			ProgramTimingEntry entry(threadNum, nextFrameNumForThisThread, 3, currentTime.count());
			lockProgramTiming.lock();
			programTiming.emplace_back(entry);
			lockProgramTiming.unlock();
		}
		
		// increment for next thread to see that it's time to proceed to wait for lock on mutexCapture
		nextFrameNumForCapture++;
	
		// copy captured raw frame into Frame object
		if (fromCameraFlag) 
		{
			while(!ocam->get_image(frameList[threadNum]->frameRawUMat)) {usleep(100);}
		}
		else 
		{
			if(!captureVideoFile->read(frameList[threadNum]->frameRawUMat)) 
			{
				cerr << "Unable to read next frame. Might be end of video..." << endl; 
				releaseVideoWriterNow = true; // tell main() to release video writer
				usleep(2000000); // sleep 2 seconds to allow main() to release video writer
				cout << "Exiting now." << endl;
				exit(EXIT_FAILURE);
			}
		}
		
		lockCapture.unlock();
		
		//frameList[threadNum]->frameRawMat.copyTo(frameList[threadNum]->frameRawUMat);
		
		// record approximate time frameRawUMat capture was started to preclude image processing latency from affecting track position accuracy
		gettimeofday(&(frameList[threadNum]->timeCaptured), NULL);
		
		//cout << "Thread " << threadNum << " frame " << nextFrameNumForThisThread << " unlocked capture." << endl;
		
		// FOR TEST timing
		if (printTimeHacksFlag > 0) {timeCapture = chrono::high_resolution_clock::now() - hack; timeCaptureSum += timeCapture.count(); hack = chrono::high_resolution_clock::now();}
		if (programTimingFlag && nextFrameNumForThisThread >= programTimingStartFrame && nextFrameNumForThisThread <= programTimingEndFrame)
		{
			currentTime = chrono::high_resolution_clock::now() - globalHack;
			ProgramTimingEntry entry(threadNum, nextFrameNumForThisThread, 4, currentTime.count());
			lockProgramTiming.lock();
			programTiming.emplace_back(entry);
			lockProgramTiming.unlock();
		}
		
		// copy to Mat for later cropping for machine learning input (Caffe cannot accept UMats)
		frameList[threadNum]->frameRawUMat.copyTo(frameList[threadNum]->frameRawMat);
		// FOR TEST timing
		if (printTimeHacksFlag > 0) {timeCopyToMat = chrono::high_resolution_clock::now() - hack; timeCopyToMatSum += timeCopyToMat.count(); hack = chrono::high_resolution_clock::now();}
		if (programTimingFlag && nextFrameNumForThisThread >= programTimingStartFrame && nextFrameNumForThisThread <= programTimingEndFrame)
		{
			currentTime = chrono::high_resolution_clock::now() - globalHack;
			ProgramTimingEntry entry(threadNum, nextFrameNumForThisThread, 5, currentTime.count());
			lockProgramTiming.lock();
			programTiming.emplace_back(entry);
			lockProgramTiming.unlock();
		}
		
		// blur each frame to reduce chance that noise is detected as movement by mixed Gaussians
		blur(frameList[threadNum]->frameRawUMat, frameList[threadNum]->frameBlurUMat, Size(blurSize, blurSize), Point(-1, -1));
		// FOR TEST timing
		if (printTimeHacksFlag > 0) {timeBlur = chrono::high_resolution_clock::now() - hack; timeBlurSum += timeBlur.count(); hack = chrono::high_resolution_clock::now();}
		if (programTimingFlag && nextFrameNumForThisThread >= programTimingStartFrame && nextFrameNumForThisThread <= programTimingEndFrame)
		{
			currentTime = chrono::high_resolution_clock::now() - globalHack;
			ProgramTimingEntry entry(threadNum, nextFrameNumForThisThread, 6, currentTime.count());
			lockProgramTiming.lock();
			programTiming.emplace_back(entry);
			lockProgramTiming.unlock();
		}
		
		// apply MOG2 mixed Gaussians to distinguish foreground movement from background
		MOG2->apply(frameList[threadNum]->frameBlurUMat, frameList[threadNum]->maskMOGUMat, mogLearnRate);
		// FOR TEST timing
		if (printTimeHacksFlag > 0) {timeMOG2 = chrono::high_resolution_clock::now() - hack; timeMOG2Sum += timeMOG2.count(); hack = chrono::high_resolution_clock::now();}
		if (programTimingFlag && nextFrameNumForThisThread >= programTimingStartFrame && nextFrameNumForThisThread <= programTimingEndFrame)
		{
			currentTime = chrono::high_resolution_clock::now() - globalHack;
			ProgramTimingEntry entry(threadNum, nextFrameNumForThisThread, 7, currentTime.count());
			lockProgramTiming.lock();
			programTiming.emplace_back(entry);
			lockProgramTiming.unlock();
		}
		
		// change all gray to white (0 at the end is threshold binary mode)
		threshold(frameList[threadNum]->maskMOGUMat, frameList[threadNum]->maskThresholdUMat, 50, 255, 0);
		// FOR TEST timing
		if (printTimeHacksFlag > 0) {timeThreshold = chrono::high_resolution_clock::now() - hack; timeThresholdSum += timeThreshold.count(); hack = chrono::high_resolution_clock::now();}
		if (programTimingFlag && nextFrameNumForThisThread >= programTimingStartFrame && nextFrameNumForThisThread <= programTimingEndFrame)
		{
			currentTime = chrono::high_resolution_clock::now() - globalHack;
			ProgramTimingEntry entry(threadNum, nextFrameNumForThisThread, 8, currentTime.count());
			lockProgramTiming.lock();
			programTiming.emplace_back(entry);
			lockProgramTiming.unlock();
		}
		
		// enlarge detections to ensure holes are filled in before target center is determined
		dilate(frameList[threadNum]->maskThresholdUMat, frameList[threadNum]->maskDilateUMat, kernelDilate);
		// FOR TEST timing
		if (printTimeHacksFlag > 0) {timeDilate = chrono::high_resolution_clock::now() - hack; timeDilateSum += timeDilate.count(); hack = chrono::high_resolution_clock::now();}
		if (programTimingFlag && nextFrameNumForThisThread >= programTimingStartFrame && nextFrameNumForThisThread <= programTimingEndFrame)
		{
			currentTime = chrono::high_resolution_clock::now() - globalHack;
			ProgramTimingEntry entry(threadNum, nextFrameNumForThisThread, 9, currentTime.count());
			lockProgramTiming.lock();
			programTiming.emplace_back(entry);
			lockProgramTiming.unlock();
		}
		
		// define outline around each target
		findContours(frameList[threadNum]->maskDilateUMat, frameList[threadNum]->conts, frameList[threadNum]->hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, Point(0, 0)); //INVESTIGATE WHETHER PUTTING THIS INTO AUXILIARY THREADS YIELDS HIGHER FRAME RATE
		// FOR TEST timing
		if (printTimeHacksFlag > 0) {timeContours = chrono::high_resolution_clock::now() - hack; timeContoursSum += timeContours.count(); hack = chrono::high_resolution_clock::now();}
		if (programTimingFlag && nextFrameNumForThisThread >= programTimingStartFrame && nextFrameNumForThisThread <= programTimingEndFrame)
		{
			currentTime = chrono::high_resolution_clock::now() - globalHack;
			ProgramTimingEntry entry(threadNum, nextFrameNumForThisThread, 10, currentTime.count());
			lockProgramTiming.lock();
			programTiming.emplace_back(entry);
			lockProgramTiming.unlock();
		}
		
		// flip bool to let main() know that thread is ready for main() to take its processed Frame
		frameList[threadNum]->frameReadyForMain = true;
		
		// FOR TEST timing
		if ((printDetailedTimeHacksFlag) && (threadNum == 0) && (printTimeHacksFlag > 0) && (nextFrameNumForThisThread >= (printTimeHacksFlag - 1)))
		{
			if ((nextFrameNumForThisThread - numberOfThreads) % printTimeHacksFlag < numberOfThreads)
			{
				lockCout.lock();
			
				cout << endl;
				cout << "Frame number:       " << nextFrameNumForThisThread << endl;
				cout << fixed << setprecision(4);
				cout << "Wait for main():    " << timeWaitForMainSum/0.033333333/printTimeHacksFlag <<
				   "     Blur:      " << timeBlurSum/0.033333333/printTimeHacksFlag << endl;
				cout << "Wait for frame num: " << timeWaitForFrameNumSum/0.03333333/printTimeHacksFlag << 
				   "     MOG2:      " << timeMOG2Sum/0.033333333/printTimeHacksFlag <<  endl;
				cout << "Wait for capture:   " << timeWaitForCaptureSum/0.033333333/printTimeHacksFlag <<
				   "     Threshold: " << timeThresholdSum/0.033333333/printTimeHacksFlag << endl;
				cout << "Capture:            " << timeCaptureSum/0.033333333/printTimeHacksFlag <<
				   "     Dilate:    " << timeDilateSum/0.033333333/printTimeHacksFlag << endl;
				cout << "Copy to Mat:        " << timeCopyToMatSum/0.033333333/printTimeHacksFlag <<
				   "     Contours:  " << timeContoursSum/0.033333333/printTimeHacksFlag << endl;
				
				lockCout.unlock();
				
				timeWaitForMainSum = 0;
				timeBlurSum = 0;
				timeWaitForFrameNumSum = 0;
				timeMOG2Sum = 0;
				timeWaitForCaptureSum = 0;
				timeThresholdSum = 0;
				timeCaptureSum = 0;
				timeDilateSum = 0;
				timeCopyToMatSum = 0;
				timeContoursSum = 0;
			}
		}
		
		// increment which frameNum thread will capture next
		nextFrameNumForThisThread += numberOfThreads;
		
		//currentTime = chrono::high_resolution_clock::now() - globalHack;
		//cout << "Thread " << threadNum << " time " << currentTime.count() << endl;
	}
}

int main(int argc, char ** argv)
{
	// FOR TEST timing
	auto hackFrame = chrono::high_resolution_clock::now();
	auto hack = chrono::high_resolution_clock::now();
	
	namedWindow("frame", WINDOW_AUTOSIZE);
	
	fromCameraFlag =         true;   // true means images from camera; false means images from image file
	bool saveVideoFlag      {false}; // save video
	bool saveStillsFlag     {false}; // save still images of each Frame
	bool drawDiagnosticInfo {false}; // plot extra Detection, Track, and rejection area info
	savePaths =              false;  // save track path crops
	saveProbabilities =      false;  // save .csv of machine learning probabilities for stills and paths
	saveDroppedTrackStills = false;  // save a still image with metadata when each Track is dropped
	int saveStart 		    {0};     // Frame number to start saving video/stills
	int saveEnd   		    {2000};  // Frame number to stop saving video/stills
	
	// create camera capture object
	// Withrobot oCam-5CR-U3 (Ver. 1604) supported resolutions and frame rates
	// USB 3.0 - YUV format
	// 2592 x 1944   @ 3.75 fps, 7.50 fps
	// 1920 x 1080   @ 7.50 fps, 15 fps
	// 1280 x 960    @ 15 fps, 30 fps
	// 1280 x 720    @ 15 fps, 30 fps
	//  640 x 480    @ 30 fps, 60 fps, 90 fps, 120 fps
	//  320 x 240    @ 30 fps, 60 fps, 90 fps, 120 fps
	int cam_number {0};
	int cam_width {1280};
	int cam_height {720};
	double cam_fps {15};
	
	bool showHelp {false};
	bool showList {false};
	string videoJustFileName {""};
	vector<int> frameSkipList; // list of skipped Frames in video files
	if (argc > 1) // arguments passed
	{
		bool invalidCharacter {false};
		
		int fileNumber = 0;
		
		char *flagsString = argv[1];
		
		char *flag = flagsString;
		
		stringstream ssArgv1Str;
		string argv1Str;
		ssArgv1Str << argv[1];
		ssArgv1Str >> argv1Str;
		
		if ((argv1Str.size() == 7) && argv1Str.find_first_of("0123456789") != string::npos) // file number passed
		{
			string argv1StrPrefix = argv1Str.substr(0,4);
			string argv1StrSuffix = argv1Str.substr(4,3);
			
			if ((argv1StrPrefix == "file") && (all_of(argv1StrSuffix.begin(), argv1StrSuffix.end(), ::isdigit)))
			{
				if (argv1Str.substr(4,3) == "000") {showList = true;}
				else {fromCameraFlag = false; fileNumber = stoi(argv1Str.substr(4,3));}
			}
			else
			{
				cout << endl << 
					"Error: Invalid 1st argument." << endl << 
					"If it was character flags, it should not have any digits." << endl << 
					"If it was a file number, it must have format 'file001', where '001' is the file number. 'file000' will show file number legend." << endl;
				showHelp = true;
			}
		}
		else // character flags passed
		{
			while (*flag)
			{
				switch (*flag) //flvsdpc
				{
					case 'v': case 'V': saveVideoFlag = true;          break;
					case 's': case 'S': saveStillsFlag = true;         break;
					case 'd': case 'D': drawDiagnosticInfo = true;     break;
					case 'p': case 'P': savePaths = true;              break;
					case 'c': case 'C': saveProbabilities = true;      break;
					case 'f': case 'F': cam_fps = 30;  			   	   break;
					case 'i': case 'I': saveDroppedTrackStills = true; break;
					default: cout << endl << "Error: Invalid character in 1st argument flags." << endl; showHelp = true;
				}
			
				flag++;
			}
		}
		
		if (argc == 3) // character flags and file number passed
		{
			stringstream ssArgv2Str;
			string argv2Str;
			ssArgv2Str << argv[2];
			ssArgv2Str >> argv2Str;
			
			if (argv2Str.size() == 7)
			{
				string argv2StrPrefix = argv2Str.substr(0,4);
				string argv2StrSuffix = argv2Str.substr(4,3);
			
				if ((argv2StrPrefix == "file") && (all_of(argv2StrSuffix.begin(), argv2StrSuffix.end(), ::isdigit)))
				{
					if (argv2Str.substr(4,3) == "000") {showList = true;}
					else {fromCameraFlag = false; fileNumber = stoi(argv2Str.substr(4,3));}
				}
				else
				{
					cout << endl << 
						"Error: Invalid 2nd argument. File number must have format 'file001', where '001' is the file number. 'file000' will show file number legend." << endl;
					showHelp = true;
				}
			}
			else
			{
				cout << endl << 
					"Error: Invalid 2nd argument. File number must have format 'file001', where '001' is the file number. 'file000' will show file number legend." << endl;
				showHelp = true;
			}
		}
		
		bool invalidSaveStart {false};
		bool invalidSaveEnd {false};
		if (argc == 4) // character flags and frame start & end numbers passed
		{
			stringstream ssSaveStartStr;
			string saveStartStr;
			ssSaveStartStr << argv[2];
			ssSaveStartStr >> saveStartStr;
			
			stringstream ssSaveEndStr;
			string saveEndStr;
			ssSaveEndStr << argv[3];
			ssSaveEndStr >> saveEndStr;
			
			if (all_of(saveStartStr.begin(), saveStartStr.end(), ::isdigit)) {saveStart = atoi(argv[2]);}
			else {invalidSaveStart = true;}
		
			if (all_of(saveEndStr.begin(), saveEndStr.end(), ::isdigit)) {saveEnd = atoi(argv[3]);}
			else {invalidSaveEnd = true;}
			
			if (invalidSaveStart) {cout << endl << "Error: Invalid start frame number in 2nd argument. Must be all digits." << endl << endl; return 0;}
			
			if (invalidSaveEnd) {cout << endl << "Error: Invalid end frame number in 3rd argument. Must be all digits." << endl << endl; return 0;}
		}
		
		if (argc == 5) // character flags, file number, and frame start & end numbers passed
		{
			stringstream ssArgv2Str;
			string argv2Str;
			ssArgv2Str << argv[2];
			ssArgv2Str >> argv2Str;
			
			if (argv2Str.size() == 7)
			{
				string argv2StrPrefix = argv2Str.substr(0,4);
				string argv2StrSuffix = argv2Str.substr(4,3);
			
				if ((argv2StrPrefix == "file") && (all_of(argv2StrSuffix.begin(), argv2StrSuffix.end(), ::isdigit)))
				{
					if (argv2Str.substr(4,3) == "000") {showList = true;}
					else {fromCameraFlag = false; fileNumber = stoi(argv2Str.substr(4,3));}
				}
				else
				{
					cout << endl << 
						"Error: Invalid 2nd argument. File number must have format 'file001', where '001' is the file number. 'file000' will show file number legend." << endl;
					showHelp = true;
				}
			}
			else
			{
				cout << endl << 
					"Error: Invalid 2nd argument. File number must have format 'file001', where '001' is the file number. 'file000' will show file number legend." << endl;
				showHelp = true;
			}
			
			stringstream ssSaveStartStr;
			string saveStartStr;
			ssSaveStartStr << argv[3];
			ssSaveStartStr >> saveStartStr;
			
			stringstream ssSaveEndStr;
			string saveEndStr;
			ssSaveEndStr << argv[4];
			ssSaveEndStr >> saveEndStr;
			
			if (all_of(saveStartStr.begin(), saveStartStr.end(), ::isdigit)) {saveStart = atoi(argv[3]);}
			else {invalidSaveStart = true;}
		
			if (all_of(saveEndStr.begin(), saveEndStr.end(), ::isdigit)) {saveEnd = atoi(argv[4]);}
			else {invalidSaveEnd = true;}
			
			if (invalidSaveStart) {cout << endl << "Error: Invalid start frame number in 3rd argument. Must be all digits." << endl; return 0;}
			
			if (invalidSaveEnd) {cout << endl << "Error: Invalid end frame number in 4th argument. Must be all digits." << endl; return 0;}
		}
		
		if (fileNumber != 0)
		{
			switch (fileNumber)
			{
				// skipped Frame numbers are the Frame immediately after the skip occurs
				case 1:		videoJustFileName = "2017-08-26_14-47-11_0117-0123_bird.mp4"; 
							break;
				case 2:		videoJustFileName = "2017-08-26_14-50-07_0346-0353_bird.mp4"; 
							frameSkipList.emplace_back(188);
							break;
				case 3:		videoJustFileName = "2017-08-26_15-05-07_0213-0219_bird.mp4"; 
							break;
				case 4:		videoJustFileName = "2017-08-26_15-10-07_0307-0309_bird.mp4"; 
							break;
				case 5:		videoJustFileName = "2017-08-26_15-10-07_0445-0447_bird.mp4"; 
							break;
				case 6:		videoJustFileName = "2017-08-26_15-20-07_0430-0432_bird.mp4"; 
							frameSkipList.emplace_back(22);
							break;
				case 7:		videoJustFileName = "2017-08-26_15-25-07_0302-0355_bird.mp4"; 
							frameSkipList.emplace_back(44);
							frameSkipList.emplace_back(76);
							// no Track to assist in finding skipped Frame here
							frameSkipList.emplace_back(150);
							// no Track to assist in finding skipped Frame here
							frameSkipList.emplace_back(219);
							frameSkipList.emplace_back(253);
							frameSkipList.emplace_back(291);
							frameSkipList.emplace_back(324);
							frameSkipList.emplace_back(357);
							frameSkipList.emplace_back(395);
							frameSkipList.emplace_back(427);
							frameSkipList.emplace_back(467);
							frameSkipList.emplace_back(499);
							// no Track to assist in finding skipped Frame here
							frameSkipList.emplace_back(570);
							frameSkipList.emplace_back(603);
							frameSkipList.emplace_back(641);
							frameSkipList.emplace_back(673);
							frameSkipList.emplace_back(713);
							frameSkipList.emplace_back(745);
							frameSkipList.emplace_back(783);
							frameSkipList.emplace_back(817);
							frameSkipList.emplace_back(849);
							frameSkipList.emplace_back(881);
							frameSkipList.emplace_back(921);
							frameSkipList.emplace_back(953);
							frameSkipList.emplace_back(992);
							frameSkipList.emplace_back(1025);
							frameSkipList.emplace_back(1064);
							frameSkipList.emplace_back(1096);
							frameSkipList.emplace_back(1135);
							frameSkipList.emplace_back(1167);
							frameSkipList.emplace_back(1200); 
							// no Track to assist in finding skipped Frame here
							frameSkipList.emplace_back(1275);
							// no Track to assist in finding skipped Frame here
							frameSkipList.emplace_back(1342);
							// no Track to assist in finding skipped Frame here
							frameSkipList.emplace_back(1412);
							frameSkipList.emplace_back(1450);
							break;
				case 8:		videoJustFileName = "2017-08-27_09-10-05_0017-0020_bird.mp4"; 
							break;
				case 9:		videoJustFileName = "2017-08-27_09-10-05_0335-0338_bird.mp4"; 
							break;
				case 10:	videoJustFileName = "2017-08-27_09-20-05_0249-0255_bird.mp4"; 
							break;
				case 11:	videoJustFileName = "2017-08-27_09-55-01_0035-0043_bird.mp4"; 
							break;
				case 12:	videoJustFileName = "2017-08-26_15-15-07_0348-0353_aircraftfixed.mp4"; 
							break;
				case 13:	videoJustFileName = "2017-08-26_15-25-07_0450-0500_aircraftfixed.mp4"; 
							break;
				case 14:	videoJustFileName = "2017-08-27_09-20-05_0030-0048_aircraftfixed.mp4"; 
							break;
				case 15:	videoJustFileName = "2017-08-27_09-20-05_0414-0429_aircraftfixed.mp4"; 
							break;
				case 16:	videoJustFileName = "2017-08-27_09-55-01_0418-0441_aircraftfixed.mp4"; 
							break;
				case 17:	videoJustFileName = "2017-08-27_10-00-01_0131-0146_aircraftfixed.mp4"; 
							break;
				case 18:	videoJustFileName = "2017-08-27_10-20-01_0057-0113_aircraftfixed.mp4"; 
							break;
				case 19:	videoJustFileName = "2017-08-27_10-20-01_0310-0328_aircraftfixed.mp4"; 
							break;
				case 20:	videoJustFileName = "2017-08-27_10-20-01_0451-0500_aircraftfixed.mp4"; 
							break;
				case 21:	videoJustFileName = "2017-08-27_09-10-05_0130-0148_dronerotor.mp4"; 
							break;
				case 22:	videoJustFileName = "2017-08-27_09-09-06_0044-0059_dronerotor.mp4"; 
							break;
				default:	cout << endl << "File number is not in list." << endl; showList = true; int frameSkipArray[] = {};
			}
		}
		
		if (argc > 5)
		{
			cout << endl << "Error: Too many arguments." << endl;
			showHelp = true;
		}
	}
	
	if (showHelp) // arguments help
	{
		cout << endl << 
			"Arguments help:" << endl <<
			endl << 
			"Allowable argument format examples:" << endl << 
			"./tracker" << endl << 
			"./tracker file001" << endl << 
			"./tracker vsdpczi" << endl << 
			"./tracker vsdpci file001" << endl << 
			"./tracker vsdpczi 0 2000" << endl << 
			"./tracker vsdpci file001 0 2000" << endl << 
			endl << 
			"In 'vsdpc', each character is a single flag to program. All character flags are optional and may be in any order." << endl << 
			endl << 
			"'v' means video will be saved as ./data/output_video.avi (significantly reduces frame rate)." << endl << 
			endl << 
			"'s' means still images of each frame will be saved as ./data/saved_stills/ (significantly reduces frame rate)." << endl << 
			endl << 
			"'d' means diagnostic information will be shown on video and still images (if they are saved)." << endl << 
			endl << 
			"'p' means track path images will be saved as ./data/saved_paths/ (white track paths on black backgrounds)." << endl << 
			endl << 
			"'c' means machine learning categorization probabilities will be saved as ./data/probabilities.csv." << endl << 
			endl << 
			"'f' means camera will capture at 30 fps (default is 15 fps). This has no effect on capture from a video file." << endl << 
			endl << 
			"'i' means an image with track info metadata will be saved for each dropped track as ./data/metadata_stills/trackname.png." << endl << 
			endl << 
			"'file001' means to use source video number 001. Without an argument specifying a file, the camera is used." << endl << 
			endl << 
			"'file000' means to list source video file number legend in this help message." << endl << 
			endl << 
			"'0' and '2000' mean that video and still images will be saved from frame numbers 0 through 2000. If unspecified, 0 to 2000 are the default values." << endl << 
			endl << 
			"This help was shown because argument format did not match one of the above examples." << endl << 
			endl;
		return 0;
	}
		
	if (showList) // arguments file name legend
	{
		cout << endl << 
			"File numbers are:" << endl << 
			"file001: 2017-08-26_14-47-11_0117-0123_bird.mp4 (good Sun rejection)" << endl << 
			"file002: 2017-08-26_14-50-07_0346-0353_bird.mp4" << endl << 
			"file003: 2017-08-26_15-05-07_0213-0219_bird.mp4" << endl << 
			"file004: 2017-08-26_15-10-07_0307-0309_bird.mp4 (lost track on HIGHLY maneuvering bird)" << endl << 
			"file005: 2017-08-26_15-10-07_0445-0447_bird.mp4 (Sun noise causes a track)" << endl << 
			"file006: 2017-08-26_15-20-07_0430-0432_bird.mp4" << endl << 
			"file007: 2017-08-26_15-25-07_0302-0355_bird.mp4 (55 birds)" << endl << 
			"file008: 2017-08-27_09-10-05_0017-0020_bird.mp4" << endl << 
			"file009: 2017-08-27_09-10-05_0335-0338_bird.mp4" << endl << 
			"file010: 2017-08-27_09-20-05_0249-0255_bird.mp4" << endl << 
			"file011: 2017-08-27_09-55-01_0035-0043_bird.mp4" << endl << 
			"file012: 2017-08-26_15-15-07_0348-0353_aircraftfixed.mp4" << endl << 
			"file013: 2017-08-26_15-25-07_0450-0500_aircraftfixed.mp4 (high speed bird frames 246-252)" << endl << 
			"file014: 2017-08-27_09-20-05_0030-0048_aircraftfixed.mp4 (lots of cloud movement rejection)" << endl << 
			"file015: 2017-08-27_09-20-05_0414-0429_aircraftfixed.mp4 (lots of cloud movement rejection)" << endl << 
			"file016: 2017-08-27_09-55-01_0418-0441_aircraftfixed.mp4" << endl << 
			"file017: 2017-08-27_10-00-01_0131-0146_aircraftfixed.mp4" << endl << 
			"file018: 2017-08-27_10-20-01_0057-0113_aircraftfixed.mp4 (bird frames 153-156 & 377-396)" << endl << 
			"file019: 2017-08-27_10-20-01_0310-0328_aircraftfixed.mp4 (bird frames 392-397 & 457-461)" << endl << 
			"file020: 2017-08-27_10-20-01_0451-0500_aircraftfixed.mp4" << endl << 
			"file021: 2017-08-27_09-10-05_0130-0148_dronerotor.mp4" << endl << 
			"file022: 2017-08-27_09-09-06_0044-0059_dronerotor.mp4" << endl << 
			endl << 
			"Exiting program. To run program, do not pass 'file000' in argument list." << endl << 
			endl;
			
		return 0;
	}
	
	// create camera capture object
	Camera ocam(cam_number, cam_width, cam_height, cam_fps);
	
	// create video file capture object
	string videoFileName {"/home/odroid/ddd/caffe/data/path_videos/" + videoJustFileName};
	VideoCapture captureVideoFile(videoFileName, CAP_FFMPEG);
	
	//Mat firstFrameMat; // throw-away frame just to get frame height and width
	UMat firstFrameUMat;
	timeval previousFrameTimeCaptured;
	if (fromCameraFlag)
	{
	    //ocam.start();
	    
	    // read 1 frame to calculate frame height and width
		// keep trying to get new image until successful (takes time for separate thread to read first image)
	    while(!ocam.get_image(firstFrameUMat)) {}; 
	    
	    // initialize previous Frame's capture time
	    gettimeofday(&previousFrameTimeCaptured, NULL);
	}
	else
	{
		// read 1 frame to calculate frame height and width
		if(!captureVideoFile.isOpened()) {cerr << "Unable to open video file: " << videoFileName; exit(EXIT_FAILURE);}
		if(!captureVideoFile.read(firstFrameUMat)) {cerr << "Unable to read next frame. Exiting..." << endl; return -1;}
	}
	//firstFrameMat.copyTo(firstFrameUMat);
	
	frameHeight = firstFrameUMat.rows;
	frameWidth = firstFrameUMat.cols;
	
	// used when matching detections and tracks as a distance that will be larger than any desired match distance because it would be out of frame
	int frameHeightPlusWidth = frameHeight + frameWidth;	
	
	// lower MOG2 learn rate if fewer threads are used
	// (since each thread gets its own MOG2, and there'll be less movement between Frames that MOG2 sees)
	if (saveVideoFlag) {mogLearnRate = 0.01;} else {mogLearnRate = 0.035;}
	
	// if operating at 15 fps, increase MOG2 learn rate since there'll be more movement between frames
	if (fromCameraFlag && cam_fps == 15) {mogLearnRate *= 2;}
	
	// FOR TEST initiate video writer
	VideoWriter outputVideo;
	string outputVideoFileName = "/home/odroid/ddd/caffe/data/output_video.avi";
	int fourCC = VideoWriter::fourcc('H','2','6','4');
	double fps = 30;
	if (fromCameraFlag) {fps = cam_fps;}
	//outputVideo.open(outputVideoFileName, fourCC, fps, Size(215, 215), true); // for track path crops
	outputVideo.open(outputVideoFileName, fourCC, fps, Size(frameWidth, frameHeight), true); // for full frame
	if (!outputVideo.isOpened() && saveVideoFlag)
	{
		cout << "Unable to open output video file for writing. Exiting..." << endl;
		return -1;
	}
	
	// FOR TEST
	// set 1 to show image on screen, 0 to hide (will speed up program significantly)
	// set >1 to imshow every X frames
	int imshowFlag {0};
	bool frameNumModulusIsZero {false};
	
	// FOR TEST timing
	chrono::duration<double> timeWholeFrame, timeWaitForThread, timeTimeDiff; 
	chrono::duration<double> timeCenter, timePred, timeDist, timeCoast, timeCreate;
	chrono::duration<double> timeDrop, timeEraseRejection, timeFrameWriting;
	chrono::duration<double> timeWaitKey, timeSaveVideo, timeSaveStills;
	chrono::duration<double> timeFlipReadyFlag, timeClearDets, currentTime;
	
	// FOR TEST timing
	float timeWholeFrameSum = 0, timeWaitForThreadSum = 0, timeTimeDiffSum = 0;
	float timeCenterSum = 0, timePredSum = 0, timeDistSum = 0, timeCoastSum = 0;
	float timeCreateSum = 0, timeDropSum = 0, timeEraseRejectionSum = 0;
	float timeFrameWritingSum = 0, timeWaitKeySum = 0, timeSaveVideoSum = 0;
	float timeSaveStillsSum = 0, timeFlipReadyFlagSum = 0, timeClearDetsSum = 0;
	
	// CONSIDER MAKING MAXDISTDETTOTRACK A DYNAMIC NUMBER BASED ON FPS
	// MAXDISTDETTOTRACK = 600/FPS WORKED WELL
	int maxDetectionSize {100};			// size above which detections are ignored (they are likely background changes like Sun)
	int trackSquareSize {10};			// size of rectangle drawn around each track ("radius", not full side length)
	int coastLimit {15};				// number of frames a track can be coasted until it is dropped
	int edgeRejectionLimit {41};		// how many pixels a Detection center must be from frame edge before it is processed for tracking
	int cropSize {82};					// pixel width & height of cropped still image of each detection
	
	// set kernel size for filling in holes
	// 0 means rectangle (will actually be square)
	kernelDilate = getStructuringElement(0, Size(3, 3), Point(1, 1));
	
	// set size at which detected movement is considered 1-pixel noise (takes place of computing-intensive cv::erode)
	int noiseSize = 1 + 2 * dilation;
	
	// radius for black circle plotted on MOG2 mask for each noise rejection area
	int rejectionRadius {15};
	int rejectionRadiusSquared = pow(rejectionRadius, 2);	// square of above number to save square root calculations later
	
	minDistanceFromOriginSquared = pow(minDistanceFromOrigin, 2);
	
	float distancesMinValue;
	int distancesMinRow, distancesMinCol;
	
	int yCenter, xCenter, cropLeft, cropRight, cropTop, cropBottom;
	Rect boundingRectxywh;
	Mat stillImage;

	vector<shared_ptr<Track>> trackList; // vector of pointers to Tracks
	trackList.reserve(40); // allocate memory for pointers to 40 Tracks so that time isn't later wasted copying vector to enlarge it; 40 was found to keep Frame rate >~15 fps
	
	rejectionList.reserve(100); // allocate memory for 100 Rejection areas so that time isn't later wasted copying vector to enlarge it
	
	int rectTop, rectBottom, rectLeft, rectRight; // for cropping still images of each detection
		
	// number of auxiliary threads to parallel process Frames
	// fewer threads if saving video due to available RAM
	if (saveVideoFlag) {numberOfThreads = 2;} 
	else 
	{
		if (fromCameraFlag) {numberOfThreads = 4;}
		else {numberOfThreads = 7;}
	}
	
	// FOR TEST
	cout << endl;
	cout << "Video source:                    ";
	if (fromCameraFlag) {cout << "from camera (" << cam_fps << " fps)" << endl;} else {cout << "from file " << videoFileName << endl;}
	cout << "Video being saved?           'v' ";
	if (saveVideoFlag) {cout << "yes" << endl;} else {cout << "no" << endl;}
	cout << "Stills being saved?          's' ";
	if (saveStillsFlag) {cout << "yes" << endl;} else {cout << "no" << endl;}
	if (saveVideoFlag || saveStillsFlag) 
	{
		cout << "Diagnostics drawn?           'd' ";
		if (drawDiagnosticInfo) {cout << "yes" << endl;}
		else {cout << "no" << endl;}
	}
	if (saveVideoFlag || saveStillsFlag) {cout << "Frames for video/stills          " << saveStart << " to " << saveEnd << endl;}
	cout << "Paths being saved?           'p' ";
	if (savePaths) {cout << "yes" << endl;} else {cout << "no" << endl;}
	cout << "Probabilities being saved?   'c' ";
	if (saveProbabilities) {cout << "yes" << endl;} else {cout << "no" << endl;}
	cout << "Metadata images being saved? 'i' ";
	if (saveDroppedTrackStills) {cout << "yes" << endl;} else {cout << "no" << endl;}
	cout << "Number of aux threads:           " << numberOfThreads << endl;
	cout << endl;
	
	size_t numberOfThreadsSizeT = (size_t)(numberOfThreads);
	frameList.reserve(numberOfThreadsSizeT);
	
	// spawn threads for parallel image processing pipelines
	vector<thread> threadList;
	threadList.reserve(numberOfThreadsSizeT);
	int threadForMain = 0; // first thread from which main() will take a Frame is thread 0
	for (int i = 0; i < numberOfThreads; i++) // populate vectors of Frames, thread ready for main() bools, and threads
	{
		frameList.emplace_back(make_unique<Frame>());
		threadList.emplace_back(thread(processFrame, i, &ocam, &captureVideoFile));
	}
	
	if (saveProbabilities)
	{
		probabilitiesFile.open("/home/odroid/ddd/caffe/data/probabilities.csv", ios_base::out | ios_base::trunc);
		probabilitiesFile << videoJustFileName << "\n" <<
			",,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25\n";
		probabilitiesFile.close();
	}
	
	// read box number unique to this computer (for outgoing image metadata)
	boxNumber.assign(istreambuf_iterator<char>(boxNumberFile), istreambuf_iterator<char>());
	
	// FOR TEST
	//cout << getBuildInformation() << endl;
	
	// FOR TEST
	programTimingEndFrame = programTimingStartFrame + 4 * numberOfThreads;
	programTiming.reserve(1000);
	
	// FOR TEST timing
	unique_lock<mutex> lockCout(mutexCout, defer_lock);
	unique_lock<mutex> lockProgramTiming(mutexProgramTiming, defer_lock);
	
	// loop for each frame
	while(1)
	{
		// FOR TEST plot program timing
		if (programTimingFlag && (mainFrameNum >= (programTimingEndFrame + numberOfThreads + 1))) 
		{
			// get min and max times
			float minTime {1000000}, maxTime {0};
			for (int i = 0; i < programTiming.size(); i++)
			{
				if (programTiming[i].time < minTime) {minTime = programTiming[i].time;}
				if (programTiming[i].time > maxTime) {maxTime = programTiming[i].time;}
			}
			
			int plotWidth {int((maxTime - minTime) / 0.001 + 1)};
			Mat programTimingPlot {Mat::zeros(Size(plotWidth, 260), CV_8UC3)};
			
			// plot gridlines
			for (int i = 10; i < 260; i += 10) {line(programTimingPlot, Point(0, i), Point(plotWidth - 1, i), Scalar(127, 127, 127), 1, 8, 0);}
			for (int i = 33; i < plotWidth; i += 33) {line(programTimingPlot, Point(i, 0), Point(i, 259), Scalar(127, 127, 127), 1, 8, 0);}
			line(programTimingPlot, Point(0, 100), Point(plotWidth - 1, 100), Scalar(255, 255, 255), 1, 8, 0);
			
			// blue, green, red
			int color[8][3] = { {  0,   0, 255}, // thread 0 red
								{  0, 127, 255}, // thread 1 orange
								{  0, 255, 255}, // thread 2 yellow
								{  0, 200,   0}, // thread 3 green
								{255, 255,   0}, // thread 4 aqua
								{255,   0,   0}, // thread 5 blue
								{255,   0, 255}, // thread 6 fuchsia
								{127,   0, 127}};// thread 7 purple
			
			int prevStepThread[8] = {0,0,0,0,0,0,0,0}; // previous step ending x coordinate for each thread in processFrame()
			int prevStepMain {0}; // previous step ending x coordinate for each thread in main()
			
			int xStart {0}, xEnd {0}, y {0};
			
			// plot thread lines
			for (int i = 0; i < programTiming.size(); i++)
			{
				if (programTiming[i].thread > 7) {continue;} // only plot threads 0 through 7
				
				if (programTiming[i].step <= 10) // step is within processFrame()
				{
					xStart = prevStepThread[programTiming[i].thread];
					xEnd = int((programTiming[i].time - minTime) / 0.001); // round down to nearest increment of 0.001 sec
					prevStepThread[programTiming[i].thread] = xEnd + 1;
				}
				else // step is within main()
				{
					xStart = prevStepMain;
					xEnd = int((programTiming[i].time - minTime) / 0.001); // round down to nearest increment of 0.001 sec
					prevStepMain = xEnd + 1;
				}
				
				if (xStart > xEnd) {xStart = xEnd;}
				
				y = 10 * programTiming[i].step - programTiming[i].thread - 1; // plot thread 0 lowest in each step's horizontal bar
			
				line(programTimingPlot, Point(xStart, y), Point(xEnd, y), Scalar(color[programTiming[i].thread][0], color[programTiming[i].thread][1], color[programTiming[i].thread][2]), 1, 8, 0);
			}
			
			programTimingPlot = programTimingPlot(Rect(330, 0, 33 * 8, 260));
			
			putText(programTimingPlot, "Wait for main()", Point(1, 7), FONT_HERSHEY_SIMPLEX, 0.3, Scalar(255, 255, 255));
			putText(programTimingPlot, "Wait for frame #", Point(1, 17), FONT_HERSHEY_SIMPLEX, 0.3, Scalar(255, 255, 255));
			putText(programTimingPlot, "Wait for capture", Point(1, 27), FONT_HERSHEY_SIMPLEX, 0.3, Scalar(255, 255, 255));
			putText(programTimingPlot, "Capture", Point(1, 37), FONT_HERSHEY_SIMPLEX, 0.3, Scalar(255, 255, 255));
			putText(programTimingPlot, "Copy to Mat", Point(1, 47), FONT_HERSHEY_SIMPLEX, 0.3, Scalar(255, 255, 255));
			putText(programTimingPlot, "Blur", Point(1, 57), FONT_HERSHEY_SIMPLEX, 0.3, Scalar(255, 255, 255));
			putText(programTimingPlot, "MOG2", Point(1, 67), FONT_HERSHEY_SIMPLEX, 0.3, Scalar(255, 255, 255));
			putText(programTimingPlot, "Threshold", Point(1, 77), FONT_HERSHEY_SIMPLEX, 0.3, Scalar(255, 255, 255));
			putText(programTimingPlot, "Dilate", Point(1, 87), FONT_HERSHEY_SIMPLEX, 0.3, Scalar(255, 255, 255));
			putText(programTimingPlot, "Contours", Point(1, 97), FONT_HERSHEY_SIMPLEX, 0.3, Scalar(255, 255, 255));
			putText(programTimingPlot, "Intentional sleep", Point(1, 107), FONT_HERSHEY_SIMPLEX, 0.3, Scalar(255, 255, 255));
			putText(programTimingPlot, "Addl wait for thread", Point(1, 117), FONT_HERSHEY_SIMPLEX, 0.3, Scalar(255, 255, 255));
			putText(programTimingPlot, "Frame time diff", Point(1, 127), FONT_HERSHEY_SIMPLEX, 0.3, Scalar(255, 255, 255));
			putText(programTimingPlot, "Det center", Point(1, 137), FONT_HERSHEY_SIMPLEX, 0.3, Scalar(255, 255, 255));
			putText(programTimingPlot, "Track prediction", Point(1, 147), FONT_HERSHEY_SIMPLEX, 0.3, Scalar(255, 255, 255));
			putText(programTimingPlot, "Det-Track distance", Point(1, 157), FONT_HERSHEY_SIMPLEX, 0.3, Scalar(255, 255, 255));
			putText(programTimingPlot, "Coast tracks", Point(1, 167), FONT_HERSHEY_SIMPLEX, 0.3, Scalar(255, 255, 255));
			putText(programTimingPlot, "Create tracks", Point(1, 177), FONT_HERSHEY_SIMPLEX, 0.3, Scalar(255, 255, 255));
			putText(programTimingPlot, "Drop tracks", Point(1, 187), FONT_HERSHEY_SIMPLEX, 0.3, Scalar(255, 255, 255));
			putText(programTimingPlot, "Erase rejections", Point(1, 197), FONT_HERSHEY_SIMPLEX, 0.3, Scalar(255, 255, 255));
			putText(programTimingPlot, "Frame writing", Point(1, 207), FONT_HERSHEY_SIMPLEX, 0.3, Scalar(255, 255, 255));
			putText(programTimingPlot, "Wait key", Point(1, 217), FONT_HERSHEY_SIMPLEX, 0.3, Scalar(255, 255, 255));
			putText(programTimingPlot, "Save video", Point(1, 227), FONT_HERSHEY_SIMPLEX, 0.3, Scalar(255, 255, 255));
			putText(programTimingPlot, "Save stills", Point(1, 237), FONT_HERSHEY_SIMPLEX, 0.3, Scalar(255, 255, 255));
			putText(programTimingPlot, "Flip thread flag", Point(1, 247), FONT_HERSHEY_SIMPLEX, 0.3, Scalar(255, 255, 255));
			putText(programTimingPlot, "Clear dets", Point(1, 257), FONT_HERSHEY_SIMPLEX, 0.3, Scalar(255, 255, 255));
			
			string programTimingPathAndFileName =  "/home/odroid/ddd/caffe/data/program_timing.png";
	
			imwrite(programTimingPathAndFileName, programTimingPlot);
			
			cout << endl << "PROGRAM ENDING BECAUSE programTimingFlag WAS SET TO TRUE. PROGRAM TIMING PLOT SAVED AS ./data/program_timing.png." << endl << endl;
			
			return 0;
		}
	
		// FOR TEST timing
		if (printTimeHacksFlag > 0) {hackFrame = chrono::high_resolution_clock::now(); hack = chrono::high_resolution_clock::now();}
		
		// FOR TEST
		//cout << "threadForMain: " << threadForMain << ", frames ready for main:";
		//for (int i = 0; i < frameList.size(); i++) {cout << " " << frameList[i]->frameReadyForMain;}
		//cout << endl;
		usleep(15000); // sleep for some of time it will take next thread to be ready (not set to ~33000 because main() sometimes takes awhile to calculate Det-Track distances)
		
		if (programTimingFlag && mainFrameNum >= programTimingStartFrame && mainFrameNum <= programTimingEndFrame)
		{
			currentTime = chrono::high_resolution_clock::now() - globalHack;
			ProgramTimingEntry entry(threadForMain, mainFrameNum, 11, currentTime.count());
			lockProgramTiming.lock();
			programTiming.emplace_back(entry);
			lockProgramTiming.unlock();
		}
		
		// wait until next thread is ready for its Frame to be further processed by main()
		while (!(frameList[threadForMain]->frameReadyForMain)) 
		{
			if ((saveVideoFlag) && (releaseVideoWriterNow)) {outputVideo.release(); cout << "Video writer released. Exiting." << endl; return 0;}
			usleep(100); // give other threads a chance to use this thread's processing power
		}
		// FOR TEST timing
		if (printTimeHacksFlag > 0) {timeWaitForThread = chrono::high_resolution_clock::now() - hack; timeWaitForThreadSum += timeWaitForThread.count(); hack = chrono::high_resolution_clock::now();}
		if (programTimingFlag && mainFrameNum >= programTimingStartFrame && mainFrameNum <= programTimingEndFrame)
		{
			currentTime = chrono::high_resolution_clock::now() - globalHack;
			ProgramTimingEntry entry(threadForMain, mainFrameNum, 12, currentTime.count());
			lockProgramTiming.lock();
			programTiming.emplace_back(entry);
			lockProgramTiming.unlock();
		}
		
		// get time between this Frame and last Frame to adjust Track predictions and allowable match distances
		// units are in 1/30ths of a second such that "1" means Frame captured at 30 fps, or "2" means Frame captured at 15 fps
		float frameTimeDiff {0};
		if (fromCameraFlag) // compare Frames' stored capture times
		{
			// rounds to nearest whole number since frames are actually at either 30 or 15 fps, not in-between numbers
			// NOTE: THIS WON'T AUTOMATICALLY HANDLE OCCASIONAL SKIPPED FRAMES WHEN PROGRAM FPS FALLS BEHIND CAMERA FPS... SKIPPED FRAME MAY CAUSE DROPPED/2ND TRACK
			frameTimeDiff = round(float(((frameList[threadForMain]->timeCaptured).tv_sec * 1000000 + (frameList[threadForMain]->timeCaptured).tv_usec - previousFrameTimeCaptured.tv_sec * 1000000 - previousFrameTimeCaptured.tv_usec)) / 33333);
			previousFrameTimeCaptured = frameList[threadForMain]->timeCaptured;
		}
		else // manually adjust certain Frames from video files known to have skipped a Frame
		{
			frameTimeDiff = 1;
			
			// frameSkipList is set in "argc > 1" line before main() while loop
			if (find(frameSkipList.begin(), frameSkipList.end(), mainFrameNum) != frameSkipList.end()) {frameTimeDiff = 2;}
		}
		// FOR TEST timing
		if (printTimeHacksFlag > 0) {timeTimeDiff = chrono::high_resolution_clock::now() - hack; timeTimeDiffSum += timeTimeDiff.count(); hack = chrono::high_resolution_clock::now();}
		if (programTimingFlag && mainFrameNum >= programTimingStartFrame && mainFrameNum <= programTimingEndFrame)
		{
			currentTime = chrono::high_resolution_clock::now() - globalHack;
			ProgramTimingEntry entry(threadForMain, mainFrameNum, 13, currentTime.count());
			lockProgramTiming.lock();
			programTiming.emplace_back(entry);
			lockProgramTiming.unlock();
		}
		
		vector<shared_ptr<Detection>> detectionList; // vector of pointers to Detections
		detectionList.reserve(100); // allocate memory for pointers to 100 Detections so that time isn't later wasted copying vector to enlarge it
		
		// find center of each Detections's outline
		int contsCount = 0;
		for (auto &cont:(frameList[threadForMain]->conts))
		{
			if (mainFrameNum <= numberOfThreads + 1) {break;} // let MOG2 develop history before processing Detections
			if (contsCount >= 100) {break;} // only process up to 100 Detections

			// calculate rectangle around each Detection outline
			boundingRectxywh = boundingRect(cont);
			
			if (boundingRectxywh.height <= noiseSize) {continue;} // skip small noise
			if (boundingRectxywh.width <= noiseSize) {continue;} // skip small noise
			if (boundingRectxywh.width > maxDetectionSize) {continue;} // skip abnormally large detections (are likely background changes like Sun)
			if (boundingRectxywh.width > maxDetectionSize) {continue;} // skip abnormally large detections (are likely background changes like Sun)
			
			// calculate Detection center coordinates
			yCenter = boundingRectxywh.y + boundingRectxywh.height / 2;
			xCenter = boundingRectxywh.x + boundingRectxywh.width / 2;
			
			// skip Detections within noise rejection areas
			bool skipDetectionFlag = false;
			for (int i = 0; i < rejectionList.size(); i++)
			{
				if ((pow((rejectionList[i][1] - xCenter), 2) + pow((rejectionList[i][0] - yCenter), 2)) < rejectionRadiusSquared)
				{
					skipDetectionFlag = true;
					break;
				}
			}
			if (skipDetectionFlag) {continue;} // skip this Detection because it's within a rejection area
			
			// ignore detection if it is too close to edge of frame
			if (yCenter < edgeRejectionLimit) {continue;}
			if (yCenter > (frameHeight - edgeRejectionLimit)) {continue;}
			if (xCenter < edgeRejectionLimit) {continue;}
			if (xCenter > (frameWidth - edgeRejectionLimit)) {continue;}
			
			// set still image crop boundaries
			cropLeft = xCenter - cropSize / 2;
			cropRight = xCenter + cropSize / 2;
			cropTop = yCenter - cropSize / 2;
			cropBottom = yCenter + cropSize / 2;
			
			// adjust crop limits if they fall outside of frame
			if (cropLeft < 0) 
			{
				cropRight -= cropLeft;
				cropLeft = 0;
			}
			else if (cropRight >= frameWidth) 
			{
				cropLeft -= cropRight - frameWidth + 1; 
				cropRight = frameWidth - 1;
			}
			if (cropTop < 0) 
			{
				cropBottom -= cropTop; 
				cropTop = 0;
			}
			else if (cropBottom >= frameHeight) 
			{
				cropTop -= cropBottom - frameHeight + 1;
				cropBottom = frameHeight - 1;
			}
			
			// crop still image from raw (un-blurred) frame and attach it to detection
			Mat newStillImage = (frameList[threadForMain]->frameRawMat)(Rect(cropLeft, cropTop, cropRight - cropLeft, cropBottom - cropTop));
			
			// create new detection
			detectionList.emplace_back(make_shared<Detection>(yCenter, xCenter, &newStillImage));
			
			contsCount++;
		};
		// FOR TEST timing
		if (printTimeHacksFlag > 0) {timeCenter = chrono::high_resolution_clock::now() - hack; timeCenterSum += timeCenter.count(); hack = chrono::high_resolution_clock::now();}
		if (programTimingFlag && mainFrameNum >= programTimingStartFrame && mainFrameNum <= programTimingEndFrame)
		{
			currentTime = chrono::high_resolution_clock::now() - globalHack;
			ProgramTimingEntry entry(threadForMain, mainFrameNum, 14, currentTime.count());
			lockProgramTiming.lock();
			programTiming.emplace_back(entry);
			lockProgramTiming.unlock();
		}
		
		// calculate predicted position of each Track based on its Track history (to be used to match to new detections)
		// the below equations average vectors from pairs of historical coordiantes to smooth prediction incase a frame introduced a somewhat erroneous Track position
		// x0 is Track's most recent x position; x4 is Track's oldest x position
		for (auto track:trackList)
		{
			//FOR TEST
			int pointsNum;
		
			// case: Track history has only 1 sets of coordinates to be used for new prediction (i.e., predicted position is same as previous position)
			if (track->trackHistoryCoordinates[3][0] == -1)
			{
				track->y0pred = float(track->trackHistoryCoordinates[4][0]);
				track->x0pred = float(track->trackHistoryCoordinates[4][1]);
			}
			else if (track->trackHistoryCoordinates[2][0] == -1)
			{
				// case: Track history has 2 sets of coordinates to be used for predicting new prediction
				// (adjusted for variable frame rate via multiplication/division by timeDiffs)
				track->y0pred =  float(track->trackHistoryCoordinates[4][0]) + 
								(float(track->trackHistoryCoordinates[4][0]) - float(track->trackHistoryCoordinates[3][0])) * frameTimeDiff / track->trackHistoryTimeDiffs[4];
				track->x0pred =  float(track->trackHistoryCoordinates[4][1]) + 
								(float(track->trackHistoryCoordinates[4][1]) - float(track->trackHistoryCoordinates[3][1])) * frameTimeDiff / track->trackHistoryTimeDiffs[4];
			}
			else //if (track->trackHistoryCoordinates[1][0] == -1)
			{
				// case: Track history has 3 sets of coordinates to be combined into new prediction
				// take average of vectors from x0-x2, x0-x1, x1-x2
				// x0pred = (x0 + (x0 - x2) / 2 + x0 + (x0 - x1) + x1 + 2 * (x1 - x2)) / 3
				// y0pred = similar equation
				// (adjusted for variable frame rate via multiplication/division by timeDiffs)
				track->y0pred = (
									2 * float(track->trackHistoryCoordinates[4][0]) + 
								 	   (float(track->trackHistoryCoordinates[4][0]) - float(track->trackHistoryCoordinates[3][0])) * frameTimeDiff / track->trackHistoryTimeDiffs[4] + 
								    	float(track->trackHistoryCoordinates[3][0]) + 
								   	   (float(track->trackHistoryCoordinates[3][0]) - float(track->trackHistoryCoordinates[2][0])) * (frameTimeDiff + track->trackHistoryTimeDiffs[4]) / track->trackHistoryTimeDiffs[3] + 
								  	   (float(track->trackHistoryCoordinates[4][0]) - float(track->trackHistoryCoordinates[2][0])) * frameTimeDiff / (track->trackHistoryTimeDiffs[4] + track->trackHistoryTimeDiffs[3])
								)/3;
				track->x0pred = (
									2 * float(track->trackHistoryCoordinates[4][1]) + 
								 	   (float(track->trackHistoryCoordinates[4][1]) - float(track->trackHistoryCoordinates[3][1])) * frameTimeDiff / track->trackHistoryTimeDiffs[4] + 
								    	float(track->trackHistoryCoordinates[3][1]) + 
								   	   (float(track->trackHistoryCoordinates[3][1]) - float(track->trackHistoryCoordinates[2][1])) * (frameTimeDiff + track->trackHistoryTimeDiffs[4]) / track->trackHistoryTimeDiffs[3] + 
								  	   (float(track->trackHistoryCoordinates[4][1]) - float(track->trackHistoryCoordinates[2][1])) * frameTimeDiff / (track->trackHistoryTimeDiffs[4] + track->trackHistoryTimeDiffs[3])
								)/3;
			}
		}
		// FOR TEST timing
		if (printTimeHacksFlag > 0) {timePred = chrono::high_resolution_clock::now() - hack; timePredSum += timePred.count(); hack = chrono::high_resolution_clock::now();}
		if (programTimingFlag && mainFrameNum >= programTimingStartFrame && mainFrameNum <= programTimingEndFrame)
		{
			currentTime = chrono::high_resolution_clock::now() - globalHack;
			ProgramTimingEntry entry(threadForMain, mainFrameNum, 15, currentTime.count());
			lockProgramTiming.lock();
			programTiming.emplace_back(entry);
			lockProgramTiming.unlock();
		}
		
		// calculate array of distances between each detection and track
		if ((detectionList.size() != 0) && (trackList.size() != 0)) // to avoid error, don't attempt to calculate Detection-Track distances if there are no Tracks or no Detections
		{
			float distances[detectionList.size()][trackList.size()], dist;
			int detectionNum = 0, trackNum = 0;
			for (auto &detection:detectionList)
			{
				for (auto &track:trackList)
				{
					dist = pow(pow(track->x0pred - detection->x, 2) + pow(track->y0pred - detection->y, 2), 0.5);
					
					// adjust max allowable match distance to account for Frame rate slower than 30 fps
					// make adjustments via temporary variable so that Track variables remain as if they were at 30 fps
					float maxDistDetToTrack = track->maxDistDetToTrack;
					if (frameTimeDiff > 1) 
					{
						maxDistDetToTrack += track->totalSpeed * (frameTimeDiff - 1);
						
						// adjust old max allowable match distance that is used solely for plotting diagnostics
						track->maxDistDetToTrackOld = maxDistDetToTrack;
					}
					
					if (dist <= maxDistDetToTrack) {distances[detectionNum][trackNum] = dist;} // record Detection-Track distance if it's within that Track's dynamic distance limit
					else {distances[detectionNum][trackNum] = frameHeightPlusWidth;} // if Detection-Track distance is above Track's dynamic distance limit, set distance to high number to preclude this pair from being declared a match
					
					trackNum++;
				}
				
				trackNum = 0;
				detectionNum++;
			}
			
			for (int i = 0; i < min(detectionList.size(), trackList.size()); i++)
			{
				// calculate next lowest distance between Detection-Track pairs and the corresponding row number (Detection) and column number (Track)
				tie(distancesMinValue, distancesMinRow, distancesMinCol) = arrayMin((float *)distances, detectionList.size(), trackList.size());
				
				// stop trying to match Detections to Tracks if closest pair is further than global max allowable distance
				if (distancesMinValue > globalMaxDistDetToTrack) {break;}
				
				// FOR TEST save match distance to Track
				trackList[distancesMinCol]->matchDistance = distancesMinValue;
				
				// mark Detection as matched so that it isn't used to create a new Track
				detectionList[distancesMinRow]->isMatched = true;
				
				// update matched Track
				trackList[distancesMinCol]->updateTrack(detectionList[distancesMinRow]->y, detectionList[distancesMinRow]->x, &(detectionList[distancesMinRow]->stillImage), (frameList[threadForMain])->timeCaptured, frameTimeDiff);
				
				// if Track has existed for awhile but not been declared "real", create noise Rejection area
				if ((trackList[distancesMinCol]->totalTrackFrames >= framesUntilRejected) && !(trackList[distancesMinCol]->isRealTrack))
				{
					// add Rejection area to list [y, x, frames until Rejection area erased]
					if (rejectionList.size() < 100) // only allow up to 100 rejection areas to keep whole Frame from accidently being rejected and mask plotting from taking too long
					{
						rejectionList.emplace_back(initializer_list<int>{trackList[distancesMinCol]->trackHistoryCoordinates[4][0], trackList[distancesMinCol]->trackHistoryCoordinates[4][1], framesRemaining});
					}
				}
				
				// for last match, set its row and column to a high number so that that Detection and Track won't match anything else
				maxOutRowAndCol((float *)distances, detectionList.size(), trackList.size(), distancesMinRow, distancesMinCol, frameHeightPlusWidth);
			}
		}
		// FOR TEST timing
		if (printTimeHacksFlag > 0) {timeDist = chrono::high_resolution_clock::now() - hack; timeDistSum += timeDist.count(); hack = chrono::high_resolution_clock::now();}
		if (programTimingFlag && mainFrameNum >= programTimingStartFrame && mainFrameNum <= programTimingEndFrame)
		{
			currentTime = chrono::high_resolution_clock::now() - globalHack;
			ProgramTimingEntry entry(threadForMain, mainFrameNum, 16, currentTime.count());
			lockProgramTiming.lock();
			programTiming.emplace_back(entry);
			lockProgramTiming.unlock();
		}
		
		// coast unmatched tracks
		for (auto track:trackList)
		{
			if (track->frameLastUpdated < mainFrameNum)
			{
				track->coastTrack(frameTimeDiff);
			}
		}
		// FOR TEST timing
		if (printTimeHacksFlag > 0) {timeCoast = chrono::high_resolution_clock::now() - hack; timeCoastSum += timeCoast.count(); hack = chrono::high_resolution_clock::now();}
		if (programTimingFlag && mainFrameNum >= programTimingStartFrame && mainFrameNum <= programTimingEndFrame)
		{
			currentTime = chrono::high_resolution_clock::now() - globalHack;
			ProgramTimingEntry entry(threadForMain, mainFrameNum, 17, currentTime.count());
			lockProgramTiming.lock();
			programTiming.emplace_back(entry);
			lockProgramTiming.unlock();
		}
		
		// create new Tracks for unmatched Detections
		for (auto detection:detectionList)
		{
			if (trackList.size() >= 30) {break;} // limited total number of Tracks to guarantee minimal fps
			
			if (!(detection->isMatched))
			{
				trackList.emplace_back(make_shared<Track>(detection->y, detection->x, &(detection->stillImage), (frameList[threadForMain])->timeCaptured, frameTimeDiff));
			}
		}
		// FOR TEST timing
		if (printTimeHacksFlag > 0) {timeCreate = chrono::high_resolution_clock::now() - hack; timeCreateSum += timeCreate.count(); hack = chrono::high_resolution_clock::now();}
		if (programTimingFlag && mainFrameNum >= programTimingStartFrame && mainFrameNum <= programTimingEndFrame)
		{
			currentTime = chrono::high_resolution_clock::now() - globalHack;
			ProgramTimingEntry entry(threadForMain, mainFrameNum, 18, currentTime.count());
			lockProgramTiming.lock();
			programTiming.emplace_back(entry);
			lockProgramTiming.unlock();
		}
		
		// drop Tracks that have coasted beyond the Frame number limit, gotten too close to Frame border, or existed too long without being declared "real"
		for (int i = trackList.size() - 1; i >= 0; i--)
		{
			if 
			(
				(trackList[i]->coastCounter > coastLimit) 
				|| 
				(trackList[i]->trackHistoryCoordinates[4][0] < edgeRejectionLimit)
				||
				(trackList[i]->trackHistoryCoordinates[4][1] < edgeRejectionLimit)
				||
				(trackList[i]->trackHistoryCoordinates[4][0] > (frameHeight - edgeRejectionLimit))
				||
				(trackList[i]->trackHistoryCoordinates[4][1] > (frameWidth - edgeRejectionLimit))
				||
				((trackList[i]->totalTrackFrames >= framesUntilRejected) && (!(trackList[i]->isRealTrack)))
			)
			{
				// save Track info if it was declared a "real" Track
				if (trackList[i]->isRealTrack) 
				{
					// if Track is being dropped due to edge rejection,
					//    reduce the coastCounter because it was erroneously incremented
					//    when new Detection was ignored at edge and Track was coasted
					if (
						(trackList[i]->trackHistoryCoordinates[4][0] < edgeRejectionLimit)
						||
						(trackList[i]->trackHistoryCoordinates[4][1] < edgeRejectionLimit)
						||
						(trackList[i]->trackHistoryCoordinates[4][0] > (frameHeight - edgeRejectionLimit))
						||
						(trackList[i]->trackHistoryCoordinates[4][1] > (frameWidth - edgeRejectionLimit))
					)
					{
						(trackList[i]->coastCounter)--;
					}
					trackList[i]->saveBeforeDropTrack();
				}
				
				// drop Track
				trackList.erase(trackList.begin() + i);
			}
		}
		// FOR TEST timing
		if (printTimeHacksFlag > 0) {timeDrop = chrono::high_resolution_clock::now() - hack; timeDropSum += timeDrop.count(); hack = chrono::high_resolution_clock::now();}
		if (programTimingFlag && mainFrameNum >= programTimingStartFrame && mainFrameNum <= programTimingEndFrame)
		{
			currentTime = chrono::high_resolution_clock::now() - globalHack;
			ProgramTimingEntry entry(threadForMain, mainFrameNum, 19, currentTime.count());
			lockProgramTiming.lock();
			programTiming.emplace_back(entry);
			lockProgramTiming.unlock();
		}
		
		// erase expired rejection areas and decrement remaining frames for unexpired rejection areas
		for (int i = rejectionList.size() - 1; i >= 0; i--)
		{
			if (rejectionList[i][2] <= 0) {rejectionList.erase(rejectionList.begin() + i);}
			else {rejectionList[i][2]--;}
		}
		// FOR TEST timing
		if (printTimeHacksFlag > 0) {timeEraseRejection = chrono::high_resolution_clock::now() - hack; timeEraseRejectionSum += timeEraseRejection.count(); hack = chrono::high_resolution_clock::now();}
		if (programTimingFlag && mainFrameNum >= programTimingStartFrame && mainFrameNum <= programTimingEndFrame)
		{
			currentTime = chrono::high_resolution_clock::now() - globalHack;
			ProgramTimingEntry entry(threadForMain, mainFrameNum, 20, currentTime.count());
			lockProgramTiming.lock();
			programTiming.emplace_back(entry);
			lockProgramTiming.unlock();
		}
		
		// write Track info on frameRaw
		if (imshowFlag > 0)
		{
			if (mainFrameNum % imshowFlag == 0) {frameNumModulusIsZero = true;}
			else {frameNumModulusIsZero = false;}
		}
		
		if (frameNumModulusIsZero || ((saveVideoFlag || saveStillsFlag) && (mainFrameNum >= saveStart) && (mainFrameNum <= saveEnd))) 
		{
			frameNumModulusIsZero = false; // reset frameNumModulusIsZero so that this doesn't trigger on every subsequent frame
			
			// rejection area circles
			if (drawDiagnosticInfo) {for (int i = 0; i < rejectionList.size(); i++) {circle(frameList[threadForMain]->frameRawMat, Point(rejectionList[i][1], rejectionList[i][0]), rejectionRadius, Scalar(0, 0, 255), 2);}}
			
			// write frame number on upper left corner
			putText(frameList[threadForMain]->frameRawMat, to_string(mainFrameNum), Point(1, 44), FONT_HERSHEY_DUPLEX, 1, Scalar(255, 255, 255));
			
			for (auto track:trackList)
			{
				// draw white circle showing allowable match distance from last Frame (thick for updated Track, thin for coasted Track)
				if (drawDiagnosticInfo) 
				{	
					if (track->y0pred != 0 || track->x0pred != 0) // skips first Track Frame where preds are initialized to 0,0
					{
						if (track->coastCounter == 0) {circle(frameList[threadForMain]->frameRawMat, Point(track->x0pred, track->y0pred), track->maxDistDetToTrackOld, Scalar(255, 255, 255), 3);}
						else {circle(frameList[threadForMain]->frameRawMat, Point(track->x0pred, track->y0pred), track->maxDistDetToTrackOld, Scalar(255, 255, 255), 1);}
					}
				}
				
				if (track->trackHistoryCoordinates[4][0] < 0) {continue;}
				if (track->trackHistoryCoordinates[4][0] >= frameHeight) {continue;}
				if (track->trackHistoryCoordinates[4][1] < 0) {continue;}
				if (track->trackHistoryCoordinates[4][1] >= frameWidth) {continue;}
				
				if (track->isRealTrack)
				{
					rectTop = track->trackHistoryCoordinates[4][0] - trackSquareSize;
					rectBottom = track->trackHistoryCoordinates[4][0] + trackSquareSize;
					rectLeft = track->trackHistoryCoordinates[4][1] - trackSquareSize;
					rectRight = track->trackHistoryCoordinates[4][1] + trackSquareSize;
				}
				else
				{
					rectTop = track->trackHistoryCoordinates[4][0] - int(trackSquareSize / 2);
					rectBottom = track->trackHistoryCoordinates[4][0] + int(trackSquareSize / 2);
					rectLeft = track->trackHistoryCoordinates[4][1] - int(trackSquareSize / 2);
					rectRight = track->trackHistoryCoordinates[4][1] + int(trackSquareSize / 2);
				}
				
				// draw square around each "real" Track (thick for currently updated, thin for coasted)							
				if ((track->isRealTrack) || (drawDiagnosticInfo))
				{
					if (track->coastCounter == 0)
					{rectangle(frameList[threadForMain]->frameRawMat, Point(rectLeft, rectTop), Point(rectRight, rectBottom), Scalar(track->blue, track->green, track->red), 3);} // thick square if track is not coasted
					else
					{rectangle(frameList[threadForMain]->frameRawMat, Point(rectLeft, rectTop), Point(rectRight, rectBottom), Scalar(track->blue, track->green, track->red), 1);} // thin square if track is coasted
				}
				
				// place Track category label next to each Track
				if (track->isRealTrack)
				{
					// machine learning categorization
					string trackLabelTitle = "Overall:";
					string trackLabel = track->trackCatLabel + " (" + to_string((int)round(track->trackCatProbability * 100)) + "%)";
					if (drawDiagnosticInfo)
					{
						putText(frameList[threadForMain]->frameRawMat, trackLabelTitle, Point(track->trackHistoryCoordinates[4][1] + trackSquareSize + 3, track->trackHistoryCoordinates[4][0] + 5), FONT_HERSHEY_DUPLEX, 0.5, Scalar(track->blue, track->green, track->red));
						putText(frameList[threadForMain]->frameRawMat, trackLabel, Point(track->trackHistoryCoordinates[4][1] + trackSquareSize + 73, track->trackHistoryCoordinates[4][0] + 5), FONT_HERSHEY_DUPLEX, 0.5, Scalar(track->blue, track->green, track->red));
					}
					else
					{
						putText(frameList[threadForMain]->frameRawMat, trackLabel, Point(track->trackHistoryCoordinates[4][1] + trackSquareSize + 3, track->trackHistoryCoordinates[4][0] + 5), FONT_HERSHEY_DUPLEX, 0.5, Scalar(track->blue, track->green, track->red));
					}
					
					// FOR TEST
					string updateLabelTitle = "Updates:";
					string stillLabelTitle = "Still:";
					string pathLabelTitle = "Path:";
					string updateLabel = to_string(track->totalTrackUpdates);
					string stillLabel = track->stillCatLabel + " (" + to_string((int)round(track->stillCatProbability * 100)) + "%)";
					string pathLabel = track->pathCatLabel + " (" + to_string((int)round(track->pathCatProbability * 100)) + "%)";
					
					// FOR TEST
					if (drawDiagnosticInfo) {putText(frameList[threadForMain]->frameRawMat, updateLabelTitle, Point(track->trackHistoryCoordinates[4][1] + trackSquareSize + 3, track->trackHistoryCoordinates[4][0] - 40), FONT_HERSHEY_DUPLEX, 0.5, Scalar(track->blue, track->green, track->red));}
					if (drawDiagnosticInfo) {putText(frameList[threadForMain]->frameRawMat, stillLabelTitle, Point(track->trackHistoryCoordinates[4][1] + trackSquareSize + 3, track->trackHistoryCoordinates[4][0] + 20), FONT_HERSHEY_DUPLEX, 0.5, Scalar(track->blue, track->green, track->red));}
					if (drawDiagnosticInfo) {putText(frameList[threadForMain]->frameRawMat, pathLabelTitle, Point(track->trackHistoryCoordinates[4][1] + trackSquareSize + 3, track->trackHistoryCoordinates[4][0] + 35), FONT_HERSHEY_DUPLEX, 0.5, Scalar(track->blue, track->green, track->red));}
					if (drawDiagnosticInfo) {putText(frameList[threadForMain]->frameRawMat, updateLabel, Point(track->trackHistoryCoordinates[4][1] + trackSquareSize + 73, track->trackHistoryCoordinates[4][0] - 40), FONT_HERSHEY_DUPLEX, 0.5, Scalar(track->blue, track->green, track->red));}
					if (drawDiagnosticInfo) {putText(frameList[threadForMain]->frameRawMat, stillLabel, Point(track->trackHistoryCoordinates[4][1] + trackSquareSize + 73, track->trackHistoryCoordinates[4][0] + 20), FONT_HERSHEY_DUPLEX, 0.5, Scalar(track->blue, track->green, track->red));}
					if (drawDiagnosticInfo) {putText(frameList[threadForMain]->frameRawMat, pathLabel, Point(track->trackHistoryCoordinates[4][1] + trackSquareSize + 73, track->trackHistoryCoordinates[4][0] + 35), FONT_HERSHEY_DUPLEX, 0.5, Scalar(track->blue, track->green, track->red));}
				}
				
				// place match distance next to each track
				if (drawDiagnosticInfo) 
				{
					string matchDistanceStrTitle = "Match:";
					string matchDistanceStr;
					
					if ((track->matchDistance < 3000) && (track->coastCounter == 0)) {matchDistanceStr += to_string(track->matchDistance);}
					else {matchDistanceStr += "--";}
					
					putText(frameList[threadForMain]->frameRawMat, matchDistanceStrTitle, Point(track->trackHistoryCoordinates[4][1] + trackSquareSize + 3, track->trackHistoryCoordinates[4][0] - 10), FONT_HERSHEY_DUPLEX, 0.5, Scalar(track->blue, track->green, track->red));
					putText(frameList[threadForMain]->frameRawMat, matchDistanceStr + "/" + to_string(int(round(track->maxDistDetToTrackOld))), Point(track->trackHistoryCoordinates[4][1] + trackSquareSize + 73, track->trackHistoryCoordinates[4][0] - 10), FONT_HERSHEY_DUPLEX, 0.5, Scalar(track->blue, track->green, track->red));
					
					track->maxDistDetToTrackOld = track->maxDistDetToTrack;
				}
				
				// place speed next to each track
				if (drawDiagnosticInfo) 
				{
					string speedStrTitle = "Speed:";
					string speedStr = to_string(track->totalSpeed);
					
					putText(frameList[threadForMain]->frameRawMat, speedStrTitle, Point(track->trackHistoryCoordinates[4][1] + trackSquareSize + 3, track->trackHistoryCoordinates[4][0] - 25), FONT_HERSHEY_DUPLEX, 0.5, Scalar(track->blue, track->green, track->red));
					putText(frameList[threadForMain]->frameRawMat, speedStr, Point(track->trackHistoryCoordinates[4][1] + trackSquareSize + 73, track->trackHistoryCoordinates[4][0] - 25), FONT_HERSHEY_DUPLEX, 0.5, Scalar(track->blue, track->green, track->red));
				}
				
				// draw white circle for each Detection
				if (drawDiagnosticInfo) {for (auto detection:detectionList) {circle(frameList[threadForMain]->frameRawMat, Point(detection->x, detection->y), 2, Scalar(255, 255, 255), 1);}}
				
				// plot last 4 historical coordinates
				if (drawDiagnosticInfo)
				{
					for (int i = 0; i <= 4; i++)
					{
						if (track->trackHistoryCoordinates[i][0] != -1)
						{
							(frameList[threadForMain]->frameRawMat).at<Vec3b>(track->trackHistoryCoordinates[i][0], track->trackHistoryCoordinates[i][1]) = Vec3b(track->blue, track->green, track->red);
							//circle(frameList[threadForMain]->frameRawMat, Point(track->trackHistoryCoordinates[i][1], track->trackHistoryCoordinates[i][0]), i, Scalar(track->blue, track->green, track->red), 1);
						}
					}
				}
				
				// draw black pixel for predicted next track location
				if (drawDiagnosticInfo) {if (track->x0pred > 0 && track->x0pred < frameWidth && track->y0pred > 0 && track->y0pred < frameHeight) {(frameList[threadForMain]->frameRawMat).at<Vec3b>(int(round(track->y0pred)), int(round(track->x0pred))) = Vec3b(0, 0, 0);}}
			}
		}
		
		if (imshowFlag > 0) {if (mainFrameNum % imshowFlag == 0) {imshow("frame", frameList[threadForMain]->frameRawMat);}}
		
		// FOR TEST timing
		if (printTimeHacksFlag > 0) {timeFrameWriting = chrono::high_resolution_clock::now() - hack; timeFrameWritingSum += timeFrameWriting.count(); hack = chrono::high_resolution_clock::now();}
		if (programTimingFlag && mainFrameNum >= programTimingStartFrame && mainFrameNum <= programTimingEndFrame)
		{
			currentTime = chrono::high_resolution_clock::now() - globalHack;
			ProgramTimingEntry entry(threadForMain, mainFrameNum, 21, currentTime.count());
			lockProgramTiming.lock();
			programTiming.emplace_back(entry);
			lockProgramTiming.unlock();
		}
		
		if (imshowFlag > 0) {if (mainFrameNum % imshowFlag == 0) {waitKey(1);}}
		// FOR TEST timing
		if (printTimeHacksFlag > 0) {timeWaitKey = chrono::high_resolution_clock::now() - hack; timeWaitKeySum += timeWaitKey.count(); hack = chrono::high_resolution_clock::now();}
		if (programTimingFlag && mainFrameNum >= programTimingStartFrame && mainFrameNum <= programTimingEndFrame)
		{
			currentTime = chrono::high_resolution_clock::now() - globalHack;
			ProgramTimingEntry entry(threadForMain, mainFrameNum, 22, currentTime.count());
			lockProgramTiming.lock();
			programTiming.emplace_back(entry);
			lockProgramTiming.unlock();
		}
		
		if (saveVideoFlag && (mainFrameNum >= saveStart) && (mainFrameNum <= saveEnd))
		{
			// save this frame to video
			
			// FOR TEST output raw frame with mask overlay
			//(frameList[threadForMain]->maskDilateUMat).copyTo(frameList[threadForMain]->maskDilateMat);
			//bitwise_not((frameList[threadForMain])->maskDilateMat, (frameList[threadForMain])->maskDilateMat);
			//Mat combinedFrame;
			//(frameList[threadForMain]->frameRawMat).copyTo(combinedFrame, frameList[threadForMain]->maskDilateMat);
			//cvtColor((frameList[threadForMain])->maskDilateMat, (frameList[threadForMain])->maskDilateMat, CV_GRAY2BGR);
			//outputVideo.write(combinedFrame);
			
			// FOR TEST output mask overlay
			//Mat maskMat;
			//(frameList[threadForMain]->maskMOGUMat).copyTo(maskMat);
			//cvtColor(maskMat, maskMat, CV_GRAY2BGR);
			//outputVideo.write(maskMat);
			
			// FOR TEST output raw frame with overlaid track info
			outputVideo.write(frameList[threadForMain]->frameRawMat);
			
			// FOR TEST output track path crops
			//Mat trackPathCropBGR = Mat::zeros(Size(round(frameWidth / 2), round(frameHeight / 2)), CV_8UC3);
			//if (trackList.size() > 0) {cvtColor(trackList[0]->trackPathCrop, trackPathCropBGR, CV_GRAY2BGR);}
			//outputVideo.write(trackPathCropBGR);
		}
		// FOR TEST timing
		if (printTimeHacksFlag > 0) {timeSaveVideo = chrono::high_resolution_clock::now() - hack; timeSaveVideoSum += timeSaveVideo.count(); hack = chrono::high_resolution_clock::now();}
		if (programTimingFlag && mainFrameNum >= programTimingStartFrame && mainFrameNum <= programTimingEndFrame)
		{
			currentTime = chrono::high_resolution_clock::now() - globalHack;
			ProgramTimingEntry entry(threadForMain, mainFrameNum, 23, currentTime.count());
			lockProgramTiming.lock();
			programTiming.emplace_back(entry);
			lockProgramTiming.unlock();
		}
		
		// FOR TEST save still images of each rawFrame
		if (saveStillsFlag && (mainFrameNum >= saveStart) && (mainFrameNum <= saveEnd))
		{
			string saveStillsPath = "/home/odroid/ddd/caffe/data/saved_stills/";
	
			char frameNumForStills[5];
			string frameNumForStillsStr;
			sprintf(frameNumForStills, "%04d", mainFrameNum);
	
			stringstream ssStills;
			ssStills << frameNumForStills;
			ssStills >> frameNumForStillsStr;
	
			string stillsPathAndFileName = saveStillsPath + "still_" + frameNumForStillsStr + ".png";
	
			imwrite(stillsPathAndFileName, frameList[threadForMain]->frameRawMat);
		}
		// FOR TEST timing
		if (printTimeHacksFlag > 0) {timeSaveStills = chrono::high_resolution_clock::now() - hack; timeSaveStillsSum += timeSaveStills.count(); hack = chrono::high_resolution_clock::now();}
		if (programTimingFlag && mainFrameNum >= programTimingStartFrame && mainFrameNum <= programTimingEndFrame)
		{
			currentTime = chrono::high_resolution_clock::now() - globalHack;
			ProgramTimingEntry entry(threadForMain, mainFrameNum, 24, currentTime.count());
			lockProgramTiming.lock();
			programTiming.emplace_back(entry);
			lockProgramTiming.unlock();
		}
		
		// FOR TEST
		if ((saveVideoFlag) && ((mainFrameNum == saveEnd) || (releaseVideoWriterNow))) {outputVideo.release(); cout << "Video writer released." << endl;}
		
		// flip thread bool to tell it to proceed with next Frame
		frameList[threadForMain]->frameReadyForMain = false;
		// FOR TEST timing
		if (printTimeHacksFlag > 0) {timeFlipReadyFlag = chrono::high_resolution_clock::now() - hack; timeFlipReadyFlagSum += timeFlipReadyFlag.count(); hack = chrono::high_resolution_clock::now();}
		if (programTimingFlag && mainFrameNum >= programTimingStartFrame && mainFrameNum <= programTimingEndFrame)
		{
			currentTime = chrono::high_resolution_clock::now() - globalHack;
			ProgramTimingEntry entry(threadForMain, mainFrameNum, 25, currentTime.count());
			lockProgramTiming.lock();
			programTiming.emplace_back(entry);
			lockProgramTiming.unlock();
		}
		
		// destroy Detection objects incase they aren't destroyed when detectionList pointer vector is re-declared
		// FOR TEST... look at deleting this line if it doesn't have an effect
		detectionList.clear();
		// FOR TEST timing
		if (printTimeHacksFlag > 0) {timeClearDets = chrono::high_resolution_clock::now() - hack; timeClearDetsSum += timeClearDets.count(); hack = chrono::high_resolution_clock::now();}
		if (programTimingFlag && mainFrameNum >= programTimingStartFrame && mainFrameNum <= programTimingEndFrame)
		{
			currentTime = chrono::high_resolution_clock::now() - globalHack;
			ProgramTimingEntry entry(threadForMain, mainFrameNum, 26, currentTime.count());
			lockProgramTiming.lock();
			programTiming.emplace_back(entry);
			lockProgramTiming.unlock();
		}
		
		// FOR TEST timing
		if (printTimeHacksFlag > 0) {timeWholeFrame = chrono::high_resolution_clock::now() - hackFrame; timeWholeFrameSum += timeWholeFrame.count(); }
		if (printTimeHacksFlag > 0) 
		{
			if (mainFrameNum % printTimeHacksFlag == 0) 
			{
				if (mainFrameNum > printTimeHacksFlag)
				{
					int totalRealTracks = 0;
					for (auto track:trackList) {if (track->isRealTrack) {totalRealTracks++;}}
					
					lockCout.lock();
					if (printDetailedTimeHacksFlag) {cout << endl;}
					
					cout << fixed << setprecision(4);
					cout << "Frame number: " << mainFrameNum << ", Track files: " << trackList.size() << ", Real tracks: " << totalRealTracks << ", Rejection areas: " << rejectionList.size() << ", fps: " << float(printTimeHacksFlag)/timeWholeFrameSum << endl;
					
					if (printDetailedTimeHacksFlag)
					{
						cout << "Wait for thread:    " << timeWaitForThreadSum/0.033333333/printTimeHacksFlag <<
						   "     Erase rej: " << timeEraseRejectionSum/0.033333333/printTimeHacksFlag << endl;
						cout << "Calc time diff:     " << timeTimeDiffSum/0.03333333/printTimeHacksFlag << 
						   "     Frame CV:  " << timeFrameWritingSum/0.033333333/printTimeHacksFlag <<  endl;
						cout << "Center:             " << timeCenterSum/0.033333333/printTimeHacksFlag <<
						   "     WaitKey:   " << timeWaitKeySum/0.033333333/printTimeHacksFlag << endl;
						cout << "Predict tracks:     " << timePredSum/0.033333333/printTimeHacksFlag <<
						   "     Video:     " << timeSaveVideoSum/0.033333333/printTimeHacksFlag << endl;
						cout << "Calc distances:     " << timeDistSum/0.033333333/printTimeHacksFlag <<
						   "     Stills:    " << timeSaveStillsSum/0.033333333/printTimeHacksFlag << endl;
						cout << "Coast tracks:       " << timeCoastSum/0.033333333/printTimeHacksFlag <<
						   "     Flip flag: " << timeFlipReadyFlagSum/0.033333333/printTimeHacksFlag << endl;
						cout << "Create tracks:      " << timeCreateSum/0.033333333/printTimeHacksFlag <<
						   "     Clear det: " << timeClearDetsSum/0.033333333/printTimeHacksFlag << endl;
						cout << "Drop tracks:        " << timeDropSum/0.033333333/printTimeHacksFlag << endl;
					}
					
					lockCout.unlock();
				}
				
				timeWaitForThreadSum = 0;
				timeTimeDiffSum = 0;
				timeCenterSum = 0;
				timePredSum = 0;
				timeDistSum = 0;
				timeCoastSum = 0;
				timeCreateSum = 0;
				timeDropSum = 0;
				timeEraseRejectionSum = 0;
				timeFrameWritingSum = 0;
				timeWaitKeySum = 0;
				timeSaveVideoSum = 0;
				timeSaveStillsSum = 0;
				timeFlipReadyFlagSum = 0;
				timeClearDetsSum = 0;
				timeWholeFrameSum = 0;
			}
		}
		
		mainFrameNum++; // increment main()'s frameNum		
		threadForMain++; // increment to next thread
		if (threadForMain == numberOfThreads) {threadForMain = 0;} // if at end of threadList, go back to start of threadList
	}
	
	outputVideo.release(); // finalize saved video
	cvDestroyAllWindows();
	return 0;
}