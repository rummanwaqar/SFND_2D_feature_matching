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

// detectorType = SHITOMASI, HARRIS, FAST, BRISK, ORB, AKAZE, SIFT
// descriptorType = BRISK, BRIEF, ORB, FREAK, AKAZE, SIFT
// matcherType = MAT_BF, MAT_FLANN
// selectorType = SEL_NN, SEL_KNN
void run_process(string detectorType,
                 string descriptorType,
                 string matcherType, 
                 string selectorType,
                 bool visDetector,
                 bool visMatching,
                 string imgBasePath,
                 string imgPrefix,
                 string imgFileType,
                 int imgStartIndex,
                 int imgEndIndex,
                 int imgFillWidth,
                 int dataBufferSize)
{
  vector<DataFrame> dataBuffer;
  dataBuffer.reserve(dataBufferSize);
  
  for (size_t imgIndex = 0; imgIndex <= imgEndIndex - imgStartIndex; imgIndex++)
    {
        /* LOAD IMAGE INTO BUFFER */

        // assemble filenames for current index
        ostringstream imgNumber;
        imgNumber << setfill('0') << setw(imgFillWidth) << imgStartIndex + imgIndex;
        string imgFullFilename = imgBasePath + imgPrefix + imgNumber.str() + imgFileType;

        // load image from file and convert to grayscale
        cv::Mat img, imgGray;
        img = cv::imread(imgFullFilename);
        cv::cvtColor(img, imgGray, cv::COLOR_BGR2GRAY);

        //// STUDENT ASSIGNMENT
        //// TASK MP.1 -> replace the following code with ring buffer of size dataBufferSize

        // push image into data frame buffer
        DataFrame frame;
        frame.cameraImg = imgGray;
        if (dataBuffer.size() == dataBufferSize) 
        {
            dataBuffer.erase(dataBuffer.begin());
        }
        dataBuffer.push_back(frame);


        //// EOF STUDENT ASSIGNMENT
        cout << "#1 : LOAD IMAGE INTO BUFFER done" << endl;

        /* DETECT IMAGE KEYPOINTS */

        // extract 2D keypoints from current image
        vector<cv::KeyPoint> keypoints; // create empty feature list for current image

        //// STUDENT ASSIGNMENT
        //// TASK MP.2 -> add the following keypoint detectors in file matching2D.cpp and enable string-based selection based on detectorType
        //// -> HARRIS, FAST, BRISK, ORB, AKAZE, SIFT

        if (detectorType.compare("SHITOMASI") == 0)
        {
            detKeypointsShiTomasi(keypoints, imgGray, visDetector);
        }
        else if (detectorType.compare("HARRIS") == 0) 
        {
            detKeypointsHarris(keypoints, imgGray, visDetector);
        }
        else
        {
            detKeypointsModern(keypoints, imgGray, detectorType, visDetector);
        }
        //// EOF STUDENT ASSIGNMENT

        //// STUDENT ASSIGNMENT
        //// TASK MP.3 -> only keep keypoints on the preceding vehicle

        // only keep keypoints on the preceding vehicle
        bool bFocusOnVehicle = true;
        cv::Rect vehicleRect(535, 180, 180, 150);
        if (bFocusOnVehicle)
        {
          vector<cv::KeyPoint> filteredKeypoints;
          for(const auto& kp : keypoints)
          {
            if (vehicleRect.contains(kp.pt))
            {
              filteredKeypoints.push_back(kp);
            }
          }
          std::swap(keypoints, filteredKeypoints);
        }
    
    	std::cout << "Task7: " << "n=" << keypoints.size() 
          << " d=" << keypoints[0].size << std::endl;

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
        (dataBuffer.end() - 1)->keypoints = keypoints;
        cout << "#2 : DETECT KEYPOINTS done" << endl;

        /* EXTRACT KEYPOINT DESCRIPTORS */

        //// STUDENT ASSIGNMENT
        //// TASK MP.4 -> add the following descriptors in file matching2D.cpp and enable string-based selection based on descriptorType
        //// -> BRIEF, ORB, FREAK, AKAZE, SIFT

        cv::Mat descriptors;
        descKeypoints((dataBuffer.end() - 1)->keypoints, (dataBuffer.end() - 1)->cameraImg, descriptors, descriptorType);
        //// EOF STUDENT ASSIGNMENT

        // push descriptors for current frame to end of data buffer
        (dataBuffer.end() - 1)->descriptors = descriptors;

        cout << "#3 : EXTRACT DESCRIPTORS done" << endl;

        if (dataBuffer.size() > 1) // wait until at least two images have been processed
        {

            /* MATCH KEYPOINT DESCRIPTORS */

            vector<cv::DMatch> matches;
            string descriptorCategory = "DES_BINARY"; // DES_BINARY, DES_HOG	
            if (descriptorType.compare("SIFT") == 0)
            {
              descriptorCategory = "DES_HOG";
            }

            //// STUDENT ASSIGNMENT
            //// TASK MP.5 -> add FLANN matching in file matching2D.cpp
            //// TASK MP.6 -> add KNN match selection and perform descriptor distance ratio filtering with t=0.8 in file matching2D.cpp

            matchDescriptors((dataBuffer.end() - 2)->keypoints, (dataBuffer.end() - 1)->keypoints,
                             (dataBuffer.end() - 2)->descriptors, (dataBuffer.end() - 1)->descriptors,
                             matches, descriptorCategory, matcherType, selectorType);

            //// EOF STUDENT ASSIGNMENT

            // store matches in current data frame
            (dataBuffer.end() - 1)->kptMatches = matches;
          	std::cout << "Task8: " << "matches=" << matches.size() << std::endl;

            cout << "#4 : MATCH KEYPOINT DESCRIPTORS done" << endl;

            // visualize matches between current and previous image
            if (visMatching)
            {
                cv::Mat matchImg = ((dataBuffer.end() - 1)->cameraImg).clone();
                cv::drawMatches((dataBuffer.end() - 2)->cameraImg, (dataBuffer.end() - 2)->keypoints,
                                (dataBuffer.end() - 1)->cameraImg, (dataBuffer.end() - 1)->keypoints,
                                matches, matchImg,
                                cv::Scalar::all(-1), cv::Scalar::all(-1),
                                vector<char>(), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

                string windowName = "Matching keypoints between two camera images";
                cv::namedWindow(windowName, 7);
                cv::imshow(windowName, matchImg);
                cout << "Press key to continue to next image" << endl;
                cv::waitKey(0); // wait for key to be pressed
            }
        }

    } // eof loop over all images
}

/* MAIN PROGRAM */
int main(int argc, const char *argv[])
{

  /* INIT VARIABLES AND DATA STRUCTURES */

  // data location
  string dataPath = "../";

  // camera
  string imgBasePath = dataPath + "images/";
  string imgPrefix = "KITTI/2011_09_26/image_00/data/000000"; // left camera, color
  string imgFileType = ".png";
  int imgStartIndex = 0; // first file index to load (assumes Lidar and camera names have identical naming convention)
  int imgEndIndex = 9;   // last file index to load
  int imgFillWidth = 4;  // no. of digits which make up the file index (e.g. img-0001.png)

  // misc
  int dataBufferSize = 2;       // no. of images which are held in memory (ring buffer) at the same time
    
  string detectorTypes[] = {"SHITOMASI", "HARRIS", "FAST", "BRISK", "ORB", "AKAZE", "SIFT"};
  string descriptorTypes[] = {"BRISK", "BRIEF", "ORB", "FREAK", "AKAZE", "SIFT"};
  string matcherTypes[] = {"MAT_BF", "MAT_FLANN"};
  string selectorTypes[] = {"SEL_NN", "SEL_KNN"};

// for task 8 and 9 enumerate through all detectors and descriptors
//   for (const auto detectorType : detectorTypes)
//   {
//     for (const auto descriptorType : descriptorTypes)
//     {
//       // only works with AKAZA
//       if (descriptorType.compare("AKAZE") == 0 && detectorType.compare("AKAZE") != 0) continue;
//       // sift and orb are not working together
//       if (descriptorType.compare("SIFT") == 0 && detectorType.compare("ORB") == 0) continue;
//       if (descriptorType.compare("ORB") == 0 && detectorType.compare("SIFT") == 0) continue;
      
//       run_process(detectorType, descriptorType, "MAT_BF", "SEL_KNN", 
//                   false, false, 
//                   dataPath + "images/", imgPrefix, imgFileType, 
//                   imgStartIndex, imgEndIndex, imgFillWidth,
//                   dataBufferSize);
//     }
//   }
  
  run_process("BRISK", "BRISK", "MAT_BF", "SEL_KNN", 
              false, true, 
              dataPath + "images/", imgPrefix, imgFileType, 
              imgStartIndex, imgEndIndex, imgFillWidth,
              dataBufferSize);
  return 0;
}
