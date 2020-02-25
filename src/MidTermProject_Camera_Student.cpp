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

using namespace std;

/* MAIN PROGRAM */
int main(int argc, char *argv[])
{
    // data location
    string dataPath = "../";
    string imgBasePath = dataPath + "images/";
    string imgPrefix = "KITTI/2011_09_26/image_00/data/000000"; // left camera, color
    string imgFileType = ".png";
    
    int imgStartIndex = 0; // first file index to load 
    int imgEndIndex = 9;   // last file index to load
    int imgFillWidth = 4;  // no. of digits which make up the file index (e.g. img-0001.png)

    // MP.1 Data Buffer Optimization: Start
    // bufferSize: no. of images which in ring buffer at the same time
    int bufferSize = 3; 
    vector<DataFrame> buffer; // list of data frames which are held in memory at the same time
    bool bVis = false;  // visualize results

    // Create strings to loop over all algorithms
    vector<string> detector_names = {"AKAZE", "BRISK", "FAST", "HARRIS", "FAST", "ORB", "SHITOMASI", "SIFT"};
    vector<string> descriptor_names = {"BRISK", "BRIEF", "ORB", "FREAK", "AKAZE", "SIFT"};

    ofstream detector_file;
    detector_file.open ("../MP7_keypoints_count.csv");

    ofstream det_des_matches;
    det_des_matches.open ("../MP8_matched_points_count.csv");

    ofstream det_des_time;
    det_des_time.open ("../MP9_log_time.csv");

    for(auto detector_name:detector_names) // start loop detector_types
    {
        bool write_detector = false;

        for(auto descriptor_name:descriptor_names) // start loop descriptor_types
        {
            if(detector_name.compare("AKAZE")!=0 && descriptor_name.compare("AKAZE")==0)
                continue;

            if(detector_name.compare("AKAZE")==0 && descriptor_name.compare("AKAZE")==0)
                continue;

            buffer.clear();
 
            cout << "Detector: " << detector_name << "   Descriptor: " << descriptor_name << endl;
            // MP.7 Performance wvaluation: Start
            // Write to detector keypoints number file
            if(!write_detector)
            {
                detector_file << detector_name;
            }
            // MP.7 Performance Evaluation: End

            det_des_matches << detector_name << "_" << descriptor_name;
            det_des_time << detector_name << "_" << descriptor_name;
            
            for (size_t imgIndex = 0; imgIndex <= imgEndIndex - imgStartIndex; imgIndex++)
            {
                //: MP.9 Performance Evaluation 3
                double t = (double)cv::getTickCount();
                // MP.9 Performance Evaluation 3

                /* LOAD IMAGE INTO BUFFER */

                // filenames for current index
                ostringstream imgNumber;
                imgNumber << setfill('0') << setw(imgFillWidth) << imgStartIndex + imgIndex;
                string imgFullFilename = imgBasePath + imgPrefix + imgNumber.str() + imgFileType;

                // load image from file and convert to grayscale
                cv::Mat img, imgGray;
                img = cv::imread(imgFullFilename);
                cv::cvtColor(img, imgGray, cv::COLOR_BGR2GRAY);

                //// STUDENT ASSIGNMENT
                //// TASK MP.1 -> replace the following code with ring buffer of size bufferSize

                // push image into data frame buffer
                DataFrame frame;
                frame.cameraImg = imgGray;

                // : MP.1 Data Buffer Optimization
                if (  buffer.size()+1 > bufferSize)
                {
                    buffer.erase(buffer.begin());
                    cout << "REPLACE IMAGE IN BUFFER done" << endl;
                }
                // : MP.1 Data Buffer Optimization

                buffer.push_back(frame);

                //// EOF STUDENT ASSIGNMENT
                cout << "#1 : LOAD IMAGE INTO BUFFER done" << endl;

                /* DETECT IMAGE KEYPOINTS */

                // extract 2D keypoints from current image
                vector<cv::KeyPoint> keypoints; // create empty feature list for current image

                // ...modified start: MP.7, MP.8, MP.9
                 //"SHITOMASI", "HARRIS", "FAST", "BRISK", "ORB", "AKAZE", "SIFT"
                string detectorType = detector_name;
                // ...modified end: MP.7, MP.8, MP.9

                //// STUDENT ASSIGNMENT
                //// TASK MP.2 -> add the following keypoint detectors in file matching2D.cpp and enable string-based selection based on detectorType
                //// -> HARRIS, FAST, BRISK, ORB, AKAZE, SIFT

                if (detectorType.compare("SHITOMASI") == 0)
                {
                    detKeypointsShiTomasi(keypoints, imgGray, false);
                }
                // : MP.2 Keypoint Detection
                // detectorType = HARRIS
                else if (detectorType.compare("HARRIS") == 0)
                {
                    detKeypointsHarris(keypoints, imgGray, false);
                }
                // Modern detector types, detectorType = FAST, BRISK, ORB, AKAZE, SIFT
                else if (detectorType.compare("FAST")  == 0 ||
                        detectorType.compare("BRISK") == 0 ||
                        detectorType.compare("ORB")   == 0 ||
                        detectorType.compare("AKAZE") == 0 ||
                        detectorType.compare("SIFT")  == 0)
                {
                    detKeypointsModern(keypoints, imgGray, detectorType, false);
                }
                else
                {
                    throw invalid_argument(detectorType + "Not a valid detectorType. Try SHITOMASI, HARRIS, FAST, BRISK, ORB, AKAZE or  SIFT.");
                }
                // MP.2 Keypoint Detection: End

                //// EOF STUDENT ASSIGNMENT

                //// STUDENT ASSIGNMENT
                //// TASK MP.3 -> only keep keypoints on the preceding vehicle

                // only keep keypoints on the preceding vehicle
                bool bFocusOnVehicle = true;
                cv::Rect vehicleRect(535, 180, 180, 150);

                // MP.3 Keypoint Removal: Start
                vector<cv::KeyPoint>::iterator keypoint;
                vector<cv::KeyPoint> keypoints_roi;
                // MP.3 Keypoint Removal: End

                if (bFocusOnVehicle)
                {
                    // : MP.3 Keypoint Removal
                    for(keypoint = keypoints.begin(); keypoint != keypoints.end(); ++keypoint)
                    {
                        if (vehicleRect.contains(keypoint->pt))
                        {
                            cv::KeyPoint newKeyPoint;
                            newKeyPoint.pt = cv::Point2f(keypoint->pt);
                            newKeyPoint.size = 1;
                            keypoints_roi.push_back(newKeyPoint);
                        }
                    }

                    keypoints =  keypoints_roi;
                    cout << "IN ROI n= " << keypoints.size()<<" keypoints"<<endl;
                    // MP.3 Keypoint Removal
                }

                //: MP.7 Performance Evaluation 1
                if(!write_detector)
                {
                    detector_file  << ", " << keypoints.size();
                }
                // MP.7 Performance Evaluation 1

                //// EOF STUDENT ASSIGNMENT

                // optional : limit number of keypoints (helpful for debugging and learning)
                bool bLimitKpts = false;
                if (bLimitKpts)
                {
                    int maxKeypoints = 50;

                    if (detectorType.compare("SHITOMASI") == 0)
                    { // there is no response info, so keep the first 50 as they are sorted in descending quality order
                        keypoints.erase(keypoints.begin() + maxKeypoints, keypoints.end());
                    }
                    cv::KeyPointsFilter::retainBest(keypoints, maxKeypoints);
                    cout << " NOTE: Keypoints have been limited!" << endl;
                }

                // push keypoints and descriptor for current frame to end of data buffer
                (buffer.end() - 1)->keypoints = keypoints;
                cout << "#2 : DETECT KEYPOINTS done" << endl;

                /* EXTRACT KEYPOINT DESCRIPTORS */

                //// STUDENT ASSIGNMENT
                //// TASK MP.4 -> add the following descriptors in file matching2D.cpp and enable string-based selection based on descriptorType
                //// -> BRIEF, ORB, FREAK, AKAZE, SIFT

                cv::Mat descriptors;

                // ...modified start: MP.7, MP.8, MP.9
                string descriptorType = descriptor_name; // BRIEF, ORB, FREAK, AKAZE, SIFT
                // ...modified end: MP.7, MP.8, MP.9

                descKeypoints((buffer.end() - 1)->keypoints, (buffer.end() - 1)->cameraImg, descriptors, descriptorType);

                //// EOF STUDENT ASSIGNMENT

                // push descriptors for current frame to end of data buffer
                (buffer.end() - 1)->descriptors = descriptors;

                cout << "#3 : EXTRACT DESCRIPTORS done" << endl;

                if (buffer.size() > 1) // wait until at least two images have been processed
                {
                    /* MATCH KEYPOINT DESCRIPTORS */

                    vector<cv::DMatch> matches;
                    string matcherType = "MAT_BF";        // MAT_BF, MAT_FLANN
                    //string descriptorType = "DES_BINARY"; // DES_BINARY, DES_HOG

                    //: MP.4 Keypoint Descriptors
                    string descriptorType;
                    if (descriptorType.compare("SIFT") == 0)
                    {
                        descriptorType == "DES_HOG";
                    }
                    else
                    {
                        descriptorType == "DES_BINARY";
                    }
                    // MP.4 Keypoint Descriptors

                    //.. modified start: MP.6 Descriptor Distance Ratio
                    string selectorType = "SEL_KNN";       // SEL_NN, SEL_KNN
                    //.. modified end: MP.6 Descriptor Distance Ratio

                    //// STUDENT ASSIGNMENT
                    //// TASK MP.5 -> add FLANN matching in file matching2D.cpp
                    //// TASK MP.6 -> add KNN match selection and perform descriptor distance ratio filtering with t=0.8 in file matching2D.cpp

                    matchDescriptors((buffer.end() - 2)->keypoints, (buffer.end() - 1)->keypoints,
                                    (buffer.end() - 2)->descriptors, (buffer.end() - 1)->descriptors,
                                    matches, descriptorType, matcherType, selectorType);

                    //// EOF STUDENT ASSIGNMENT

                    // store matches in current data frame
                    (buffer.end() - 1)->kptMatches = matches;

                    //: MP.8 Performance Evaluation 2
                    det_des_matches << ", " << matches.size();
                    // MP.8 Performance Evaluation 2

                    //: MP.9 Performance Evaluation 3
                    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
                    det_des_time << ", " << 1000*t;
                    // MP.9 Performance Evaluation 3

                    cout << "#4 : MATCH KEYPOINT DESCRIPTORS done" << endl;

                    // visualize matches between current and previous image
                    bVis = false;
                    if (bVis)
                    {
                        cv::Mat matchImg = ((buffer.end() - 1)->cameraImg).clone();
                        cv::drawMatches((buffer.end() - 2)->cameraImg, (buffer.end() - 2)->keypoints,
                                        (buffer.end() - 1)->cameraImg, (buffer.end() - 1)->keypoints,
                                        matches, matchImg,
                                        cv::Scalar::all(-1), cv::Scalar::all(-1),
                                        vector<char>(), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

                        string windowName = "Matching keypoints between two camera images";
                        cv::namedWindow(windowName, 7);
                        cv::imshow(windowName, matchImg);
                        cout << "Press key to continue to next image" << endl;
                        cv::waitKey(0); // wait for key to be pressed
                    }
                    bVis = false;
                }

            } // eof loop over all images

            //: MP.7, MP.8, and MP.9
            if(!write_detector)
            {
                detector_file << endl;
            }

            write_detector = true;

            det_des_matches << endl;
            det_des_time << endl;
        }// eof loop over descriptor_types
    }// eof loop over detector_types

    detector_file.close();
    det_des_matches.close();
    det_des_time.close();
    // MP.7, MP.8, and MP.9

    return 0;
}
