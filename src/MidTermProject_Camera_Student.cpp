/* INCLUDES FOR THIS PROJECT */
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <limits>
#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>

#include "dataStructures.h"
#include "matching2D.hpp"
#include "ring_buffer.hpp"

using namespace std;

/* MAIN PROGRAM */
int main(int argc, const char *argv[])
{
    /* INIT VARIABLES AND DATA STRUCTURES */
    // data location
    string dataPath = "../";

    // camera
    string imgBasePath = "/home/david/onlineLearning/feature-tracking-2D/images/";
    string imgPrefix = "KITTI/2011_09_26/image_00/data/000000"; // left camera, color
    string imgFileType = ".png";
    int imgStartIndex = 0; // first file index to load (assumes Lidar and camera names have identical naming convention)
    int imgEndIndex = 9;   // last file index to load
    int imgFillWidth = 4;  // no. of digits which make up the file index (e.g. img-0001.png)

    // misc
    int dataBufferSize = 3;       // no. of images which are held in memory (ring buffer) at the same time
    vector<DataFrame> dataBuffer; // list of data frames which are held in memory at the same time
    bool bVis = false;            // visualize results

    /* MAIN LOOP OVER ALL IMAGES */

    //.. add start: MP.7, MP.8, and MP.9
    vector<string> detector_type_names = {"SHITOMASI", "HARRIS", "FAST", "BRISK", "ORB", "AKAZE", "SIFT"};
    vector<string> descriptor_type_names = {"BRISK", "BRIEF", "ORB", "FREAK", "AKAZE", "SIFT"};

    ofstream detector_file;
    detector_file.open ("../MP_7_Counts_Keypoints.csv");

    ofstream det_des_matches;
    det_des_matches.open ("../MP_8_Counts_Matched_Keypoints.csv");

    ofstream det_des_time;
    det_des_time.open ("../MP_9_Log_Time.csv");

    for(auto detector_type_name:detector_type_names) // start loop detector_types
    {
        bool write_detector = false;

        for(auto descriptor_type_name:descriptor_type_names) // start loop descriptor_types
        {
            if(detector_type_name.compare("AKAZE")!=0 && descriptor_type_name.compare("AKAZE")==0)
                continue;

            dataBuffer.clear();

            cout << "===================================================================" << endl;
            cout << "Detector Type: " << detector_type_name << "   Descriptor Type: " << descriptor_type_name << endl;
            cout << "===================================================================" << endl;

            //.. add start: MP.7 Performance Evaluation 1
            // Write to detector keypoints number file
            if(!write_detector)
            {
                detector_file << detector_type_name;
            }
            //.. add end: MP.7 Performance Evaluation 1

            //.. add start: MP.8 Performance Evaluation 2
            det_des_matches << detector_type_name << "_" << descriptor_type_name;
            //.. add end: MP.8 Performance Evaluation 2

            // MP.9 Performance Evaluation 3
            det_des_time << detector_type_name << "_" << descriptor_type_name;
            // MP.9 Performance Evaluation 3
            // End of MP.7, MP.8, and MP.9
            circular_buffer<DataFrame> dataBufferRing(dataBufferSize);

            for (size_t imgIndex = 0; imgIndex <= imgEndIndex - imgStartIndex; imgIndex++)
            {
                /* LOAD IMAGE INTO BUFFER */
                double t = (double)cv::getTickCount();
                // assemble filenames for current index
                ostringstream imgNumber;

                imgNumber << setfill('0') << setw(imgFillWidth) << imgStartIndex+imgIndex;

                string imgFullFilename = imgBasePath + imgPrefix + imgNumber.str()+imgFileType;

                cout << "imgFullFilename " << imgFullFilename << endl;

                // load image from file and convert to grayscale
                cv::Mat img, imgGray;
                img = cv::imread(imgFullFilename);
                cv::cvtColor(img, imgGray, cv::COLOR_BGR2GRAY);

                //// STUDENT ASSIGNMENT
                //// TASK MP.1 -> replace the following code with ring buffer of size dataBufferSize

                // push image into data frame buffer
                DataFrame frame;
                frame.cameraImg = imgGray;

                if (  dataBufferRing.size()+1 > dataBufferSize)
                {
                    dataBufferRing.erase(dataBuffer.begin());
                    cout << "REPLACE IMAGE IN BUFFER done" << endl;
                }

                dataBufferRing.put(frame);
                cout << "ring size " <<  dataBufferRing.size() << endl;
                //dataBuffer.push_back(frame);
                // replacing with ring buffer

                //// EOF STUDENT ASSIGNMENT
                cout << "#1 : LOAD IMAGE INTO BUFFER done" << endl;

                /* DETECT IMAGE KEYPOINTS */

                // extract 2D keypoints from current image
                vector<cv::KeyPoint> keypoints; // create empty feature list for        current image
                string detectorType = "SHITOMASI";

                //// STUDENT ASSIGNMENT
                //// TASK MP.2 -> add the following keypoint detectors in file      matching2D.cpp and enable string-based selection based on detectorType
                //// -> HARRIS, FAST, BRISK, ORB, AKAZE, SIFT
                cout << "ring size 2 " <<  dataBufferRing.size() << endl;
                if (detectorType.compare("SHITOMASI") == 0)
                {
                    detKeypointsShiTomasi(keypoints, imgGray, true);
                }
                else if (detectorType.compare("HARRIS") == 0)
                {
                    detKeypointsHarris(keypoints, imgGray, false);
                }

                else if (detectorType.compare("AKAZE") == 0 ||
                         detectorType.compare("BRISK") == 0 ||
                         detectorType.compare("FAST")  == 0 ||
                         detectorType.compare("ORB")   == 0 ||
                         detectorType.compare("SIFT")  == 0)
                {
                    detKeypointsModern(keypoints, imgGray, detectorType, false);
                }
                else
                {
                    throw invalid_argument(detectorType + " not a valid detector. Please use AKAZE, BRISK, FAST, SHITOMASI, HARRIS, ORB or SIFT.");
                }
                //// EOF STUDENT ASSIGNMENT

                //// STUDENT ASSIGNMENT
                //// TASK MP.3 -> only keep keypoints on the preceding vehicle
                vector<cv::KeyPoint>::iterator keypoint;
                vector<cv::KeyPoint> kp_roi;
                cout << "ring size 3 " <<  dataBufferRing.size() << endl;
                // only keep keypoints on the preceding vehicle
                bool bFocusOnVehicle = true;
                cv::Rect vehicleRect(535, 180, 180, 150);
                if (bFocusOnVehicle)
                {
                    // ...add start: MP.3 Keypoint Removal
                     for(keypoint = keypoints.begin(); keypoint != keypoints.end(); ++keypoint)
                    {
                        if (vehicleRect.contains(keypoint->pt))
                        {
                            cv::KeyPoint newKeyPoint;
                            newKeyPoint.pt = cv::Point2f(keypoint->pt);
                            newKeyPoint.size = 1;
                            kp_roi.push_back(newKeyPoint);
                        }
                    }
                    keypoints =  kp_roi;
                    cout << "IN ROI n= " << keypoints.size()<< " keypoints"<<endl;
                    // ...add end: MP.3 Keypoint Removal
                }
                //// EOF STUDENT ASSIGNMENT
                cout << "ring size 4 " <<  dataBufferRing.size() << endl;
                if(!write_detector)
                {
                    detector_file  << ", " << keypoints.size();
                }
                // optional : limit number of keypoints (helpful for debugging and      learning)
                cout << "#after roi" << endl;
                bool bLimitKpts = false;
                if (bLimitKpts)
                {
                    int maxKeypoints = 50;

                    if (detectorType.compare("SHITOMASI") == 0)
                    { // there is no response info, so keep the first 50 as they are        sorted in descending quality order
                        keypoints.erase(keypoints.begin() + maxKeypoints, keypoints.end     ());
                    }
                    cv::KeyPointsFilter::retainBest(keypoints, maxKeypoints);
                    cout << " NOTE: Keypoints have been limited!" << endl;
                }
                 cout << "#after keypoints" << endl;
                // push keypoints and descriptor for current frame to end of data buffer
                cout << "# size " << dataBuffer.size() << endl;
                (dataBufferRing.end() - 1)->keypoints = keypoints;

                cout << "#2 : DETECT KEYPOINTS done" << endl;

                /* EXTRACT KEYPOINT DESCRIPTORS */

                //// STUDENT ASSIGNMENT
                //// TASK MP.4 -> add the following descriptors in file     matching2D.cpp  and enable string-based selection based on  descriptorType
                //// -> BRIEF, ORB, FREAK, AKAZE, SIFT

                cv::Mat descriptors;

                // BRIEF, ORB, FREAK, AKAZE, SIFT
                string descriptorType = descriptor_type_name;

                descKeypoints((dataBuffer.end() - 1)->keypoints, (dataBuffer.end() - 1)->cameraImg, descriptors, descriptorType);
                //// EOF STUDENT ASSIGNMENT
                // push descriptors for current frame to end of data buffer
                (dataBuffer.end() - 1)->descriptors = descriptors;
                cout << "#3 : EXTRACT DESCRIPTORS done" << endl;
                // wait until at least two images    have   been processed
                if (dataBufferRing.size() > 1)
                {
                    /* MATCH KEYPOINT DESCRIPTORS */
                    vector<cv::DMatch> matches;
                    string matcherType = "MAT_BF";        // MAT_BF, MAT_FLANN
                    //string descriptorType = "DES_BINARY"; // DES_BINARY, DES_HOG

                    //.. add start: MP.4 Keypoint Descriptors
                    string descriptorType;
                    if (descriptorType.compare("SIFT") == 0)
                    {
                        descriptorType == "DES_HOG";
                    }
                    else
                    {
                        descriptorType == "DES_BINARY";
                    }
                    //.. add end: MP.4 Keypoint Descriptors

                    //.. modified start: MP.6 Descriptor Distance Ratio
                    string selectorType = "SEL_KNN";       // SEL_NN, SEL_KNN
                    //.. modified end: MP.6 Descriptor Distance Ratio

                    matchDescriptors((dataBuffer.end() - 2)->keypoints, (dataBuffer.end() - 1)->keypoints,
                    (dataBuffer.end() - 2)->descriptors, (dataBuffer.end() - 1)->descriptors, matches, descriptorType, matcherType, selectorType);

                    cout << "matches " << matches.size() << endl;
                    //// EOF STUDENT ASSIGNMENT

                    // store matches in current data frame
                    (dataBuffer.end() - 1)->kptMatches = matches;

                    //.. add start: MP.8 Performance Evaluation 2
                    det_des_matches << ", " << matches.size();
                    //.. add end: MP.8 Performance Evaluation 2

                    //.. add start: MP.9 Performance Evaluation 3
                    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
                    det_des_time << ", " << 1000*t;
                    //.. add end: MP.9 Performance Evaluation 3

                    cout << "#4 : MATCH KEYPOINT DESCRIPTORS done" << endl;

                    // visualize matches between current and previous image
                    bVis = true;
                    if (bVis)
                    {
                        cv::Mat matchImg = ((dataBuffer.end() - 1)->cameraImg).clone();
                        cv::drawMatches((dataBuffer.end() - 2)->cameraImg, (dataBuffer.end() - 2)->keypoints,
                                        (dataBuffer.end() - 1)->cameraImg, (dataBuffer.end() - 1)->keypoints,
                                        matches, matchImg,
                                        cv::Scalar::all(-1), cv::Scalar::all(-1),
                                        vector<char>(), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

                        string windowName = "Matching keypoints between two camera      images";
                        cv::namedWindow(windowName, 7);
                        cv::imshow(windowName, matchImg);
                        cout << "Press key to continue to next image" << endl;
                        cv::waitKey(0); // wait for key to be pressed
                    }
                    bVis = false;
                }
            }
        }
    }
    return 0;
}
