/*
 * The Example of the Withrobot oCam-1MGN-U API using with OpenCV. (Linux only.)
 *
 * This example shows how to get image from the oCam-1MGN using the Withrobot camera API.
 * And also shows how to control the oCam-1MGN using the Withrobot camera API.
 *
 * This example program usage.:
 * 	- Press the key 'q' for exit this program.
 *  - Press the key ']' for increase the exposure.
 *  - Press the key '[' for decrease the exposure.
 *  - Press the key '=' for increase the brightness.
 *  - Press the key '-' for decrease the brightness.
 */

#include <stdio.h>
#include <errno.h>
#include <sys/stat.h>
#include <dirent.h>
//#include <boost/filesystem.hpp>

#include "opencv2/opencv.hpp"
#include "withrobot_camera.hpp"	/* withrobot camera API */
#include "ConvertColor.h"
#include "format_converter.hpp"

#include "Track.hpp"


using namespace cv;
using namespace std;
//using namespace boost::filesystem;


bool compareContourAreas ( std::vector<cv::Point> contour1, std::vector<cv::Point> contour2 ) {
    double i = fabs( contourArea(cv::Mat(contour1)) );
    double j = fabs( contourArea(cv::Mat(contour2)) );
    return ( i < j );
}

bool intersects(Rect r1, Rect r2) {
    Rect intersection = r1 & r2;
    return (intersection.area() > 0);
}

Rect ensureSize(int maxHeight, int maxWidth, Rect r) {
    int startx = min(max(0, r.x), maxWidth);
    int starty = min(max(0, r.y), maxHeight);
    int newWidth = r.width;
    if ((startx + r.width) > maxWidth) {
        newWidth = maxWidth - startx;
    }
    int newHeight = r.height;
    if ((starty + r.height) > maxHeight) {
        newHeight = maxHeight - starty;
    }
    return Rect(startx, starty, newWidth, newHeight);
}

string getCurrentTime() {
    time_t rawtime;
    struct tm * timeinfo;
    struct timeval tval;
    char buffer[80];
    time(&rawtime);
    timeinfo = localtime(&rawtime);
    gettimeofday(&tval, NULL);

    long int ms = ((long int)tval.tv_usec)/1000;
    strftime(buffer, 80, "%d-%m-%Y_%H-%M-%S", timeinfo);
    stringstream ss;
    ss << string(buffer) << "." << ms;
    return ss.str();
}

vector<Rect> getLargeAreas(vector<vector<Point> > contours, int height, int width) {
    vector<Rect> contourRects;
	for (int i = 0; i < (int)contours.size(); i++) {
		double a = contourArea(contours[i], false);
		if (a > 20 && a < ((height * width) / 4)) {
			Rect r = boundingRect(contours[i]);
            double sizeincrease = r.area() / 200;
            r -= Point(sizeincrease, sizeincrease);
            r += Size(sizeincrease * 2, sizeincrease * 2);
			contourRects.push_back(r);
		}
	}
	return contourRects;
}

// assumes contourRects is not empty
vector<Rect> combineIntersectedAreas(vector<Rect> contourRects) {
	vector<Rect> combinedRects;
	combinedRects.push_back(contourRects[0]);
	bool anyIntersected = true;
	// keep doing this until no intersections are added
	while(anyIntersected)
	{
		anyIntersected = false;
		for (int i = 1; i < (int)contourRects.size(); i++)
		{
			Rect r2 = contourRects[i];
			bool intersected = false;
			for (int j = 0; j < (int)combinedRects.size(); j++)
			{
				Rect r1 = combinedRects[j];
				if (intersects(r1, r2))
				{
					combinedRects[j] = r1 | r2;
					intersected = true;
					anyIntersected = true;
				}
			}
			if (!intersected)
			{
				combinedRects.push_back(r2);
			}
		}

		if (anyIntersected && combinedRects.size()  > 0)
		{
			contourRects.clear();
			contourRects.insert(contourRects.end(), combinedRects.begin(), combinedRects.end());
			combinedRects.clear();
			combinedRects.push_back(contourRects[0]);
		}
	}
	
	return combinedRects;
}

double convertTime(struct timeval time)
{
    return ((double)time.tv_sec) + (((double)time.tv_usec)/(1000*1000));
}

bool is_file(const char* path)
{
    struct stat buf;
    stat(path, &buf);
    return S_ISREG(buf.st_mode);
}

bool is_dir(const char* path)
{
    struct stat buf;
    stat(path, &buf);
    return S_ISDIR(buf.st_mode);
}

bool has_suffix(const std::string &str, const std::string &suffix) {
    return str.size() >= suffix.size() &&
           str.compare(str.size() - suffix.size(), suffix.size(), suffix) == 0;
}

bool write_images(string file, string outdir)
{
    if (!has_suffix(file, ".mp4"))
    {
        cout << "file is not a video file: " << file << "\n";
        return false;
    }

    VideoCapture stream1(file);

    if (!stream1.isOpened())   //check if video device has been initialised
    {
        cout << "cannot open video file" << "\n";
        return false;
    }

    int width = stream1.get(3);
    int height = stream1.get(4);

    cout << "height: " << height << ", width: " << width << ", file " << file << "\n";

    /*unsigned char* frame_buffer;
    unsigned char* rgb_buffer;
    Withrobot::camera_format format;
    std::vector<Withrobot::usb_device_info> dev_list;*/

    bool showVideo = false;
    bool writeFiles = true;

    /*int height = 720;
    int width = 1280;
    int fps = 30;

    format.width = width;
    format.height = height;
    format.pixformat = Withrobot::fourcc_to_pixformat('Y','U','Y','V');
    format.rate_numerator = 1;
    format.rate_denominator = fps;


	const char* devPath = "/dev/video0";

    Withrobot::Camera camera(devPath, &format);

    camera.set_format(width, height, Withrobot::fourcc_to_pixformat('Y','U','Y','V'), 1, fps);


    // get current camera format (image size and frame rate)
    Withrobot::camera_format camFormat;
    camera.get_current_format(camFormat);

    // Print infomations
    std::string camName = camera.get_dev_name();
    std::string camSerialNumber = camera.get_serial_number();

    printf("dev: %s, serial number: %s\n", camName.c_str(), camSerialNumber.c_str());
    printf("----------------- Current format informations -----------------\n");
    camFormat.print();
    printf("---------------------------------------------------------------\n");

    // Start streaming
    if (!camera.start()) {
    	perror("Failed to start.");
    	exit(0);
    }

    // Initialize OpenCV
    std::string windowName = camName + " " + camSerialNumber;
    cv::Mat srcImg(camFormat.height, camFormat.width, CV_8UC2);*/
    std::string windowName = "Object Detect";
    if (showVideo) {
        cv::namedWindow(windowName.c_str(), CV_WINDOW_NORMAL);//CV_WINDOW_KEEPRATIO|CV_WINDOW_AUTOSIZE);
    }

    /*frame_buffer = new unsigned char[format.height*format.width*2];
    rgb_buffer = new unsigned char[format.height*format.width*3];

    //GuvcviewFormatConverter* format_converter;
    //format_converter = new GuvcviewFormatConverter(format.width, format.height);

//		cv::Mat srcImg2(camFormat.height, camFormat.width, CV_8UC3, rgb_buffer);
	cv::Mat srcImg2(camFormat.height, camFormat.width, CV_8UC3);*/
    cv::Mat movieFrame;
    //cv::Mat flipImg;
    cv::Mat gray, gray1, res1, frameDelta, thresh;

    stream1.read(movieFrame);

    cv::Mat acc = Mat::zeros(movieFrame.size(), CV_32FC3);

    vector<int> compression_params;
    compression_params.push_back(CV_IMWRITE_PNG_COMPRESSION);
    compression_params.push_back(0);

    /*
     * Main loop
     */
    bool quit = false;
    while (!quit) {
        /* Copy a single frame(image) from camera. This is a blocking function. */
        /*bool grabbedFrame = stream1.read(movieFrame);
    	if (!grabbedFrame) {
    	    printf("error number: %d\n", errno);
    	    perror("Cannot get image from camera");
    	    continue;
    	}*/

        stream1 >> movieFrame;
        if (movieFrame.empty()) {
            printf("error number: %d\n", errno);
            perror("Cannot get image from camera");
            break;
        }

        cvtColor(movieFrame, gray, CV_BGR2GRAY);
        accumulateWeighted(movieFrame, acc, 0.2);
        convertScaleAbs(acc, res1);
        cvtColor(res1, gray1, CV_BGR2GRAY);
        absdiff(gray1, gray, frameDelta);

        threshold(frameDelta, frameDelta, 40, 255, CV_THRESH_BINARY);
        dilate(frameDelta, frameDelta, Mat(), Point(-1, -1), 2);
        vector<vector<Point> > contours;
        findContours(frameDelta, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);

        if (!contours.empty())
        {
            vector<Rect> contourRects = getLargeAreas(contours, height, width);
            if (!contourRects.empty()) {
                vector<Rect> combinedRects = combineIntersectedAreas(contourRects);

                for (int i = 0; i < (int)combinedRects.size(); i++)
                {
                    if (showVideo)
                    {
                        rectangle(movieFrame, combinedRects[i], Scalar(0, 0, 255), 2);
                    }
                    if (writeFiles)
                    {
                        Rect adjRect = ensureSize(height, width, combinedRects[i]);
                        Mat cropped =  movieFrame(adjRect);
                        if (!cropped.empty())
                        {
                            double vidtime = stream1.get(CV_CAP_PROP_POS_MSEC);
                            if (vidtime > 1000) { // accumulate needs at least a second to get started
                                stringstream ss;
                                int idx = file.rfind("/");
                                if (idx > 0)
                                    idx += 1;
                                string name = file.substr(idx);
                                string basename = name.substr(0, name.length() - 4);
                                ss << outdir << "/" << basename << "-" << vidtime << "-" << i << ".png";
                                string filename = ss.str();
                                cout << filename << "\n";
                                imwrite(filename, cropped, compression_params);
                            }
                        }
                    }
                }

            }
        }


        /* Show image */
        if (showVideo) {
            cv::imshow(windowName.c_str(), movieFrame);

            char key = cv::waitKey(10);

            /* Keyboard options */
            switch (key) {
                /* When press the 'q' key then quit. */
                case 'q':
                    quit = true;
                    break;

                default:
                    break;
            }
        }
    }

    cv::destroyAllWindows();

    //camera.stop();

    printf("Done.\n");

    return true;
}


/*
 *	Main
 */
int main (int argc, char* argv[])
{
    if (argc < 3) {
        cout << "Must specify an input file or directory and output directory";
    }
    std::string file = argv[1];
    std::string outdir = argv[2];

    if (is_dir(file.c_str())) {
//        path p(file);
//        for (auto i = directory_iterator(p); i != directory_iterator(); i++) {
//            if (!is_directory(i->path()))
//            {
//                cout << i->path().string() << endl;
//            }
//            else
//                continue;
//        }
        if (auto dir = opendir(file.c_str())) {
            while (auto f = readdir(dir)) {
                if (!f->d_name || f->d_name[0] == '.')
                    continue;

                stringstream s;
                s << file << "/" << f->d_name;
                //cout << s.str() << "\n";
                write_images(s.str(), outdir);
            }
            closedir(dir);
        }
    } else if (is_file(file.c_str())){
        write_images(file, outdir);
    }
}

