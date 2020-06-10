# SFND Write up

## MP.1 Data Buffer Optimization

A vector is used to create a ring buffer. Its size is initialized to buffer size so no 
memory reallocation takes place after initialization.

```
vector<DataFrame> dataBuffer;
dataBuffer.reserve(dataBufferSize);
```

Before adding a new element, we check and remove the oldest element, if the buffer size is
equal to the max size.
```
if (dataBuffer.size() == dataBufferSize)
{
    dataBuffer.erase(dataBuffer.begin());
}
dataBuffer.push_back(frame);
```

## MP.2 Keypoint Detection

All modern detectors (binary based) are implemented in `detKeypointsModern(...)`. Since all these detectors are implementing `cv::FeatureDetector` they are created in the if else block and then the detect call is the same for all. 

```
detector->detect(img, keypoints);
```

Harris detector is implemented in exactly the same way as the lecture quiz.

## MP.3 Keypoint Removal

We iterate over all keypoints and add the ones inside the bounding box to a new vector. Once the process is complete the data in the old vector is swapped with the new filtered vector.

```
vector<cv::KeyPoint> filteredKeypoints;
for (const auto &kp : keypoints)
{
    if (vehicleRect.contains(kp.pt))
    {
        filteredKeypoints.push_back(kp);
    }
}
std::swap(keypoints, filteredKeypoints);
```

## MP.4 Keypoint Descriptors

All descriptors implement the `cv::DescriptorExtractor` base class. They are created in a if else block based on the string value. The compute descriptor call is the same of all of them.
```
extractor->compute(img, keypoints, descriptors);
```

## MP.5 Descriptor Matching
We use the OpenCV FLANN implementation for descriptor matching. This has to be implemented 
differently for HOG based and binary based descriptors. This was investigated in the [this](https://stackoverflow.com/questions/43830849/opencv-use-flann-with-orb-descriptors-to-match-features) stackoverflow question.

```
if (matcherType.compare("MAT_FLANN") == 0)
  {
    if (descriptorType.compare("DES_HOG") == 0)
    {
      matcher = cv::FlannBasedMatcher::create();
    }
    else
    {
      matcher = cv::makePtr<cv::FlannBasedMatcher>(cv::makePtr<cv::flann::LshIndexParams>(12, 20, 2));
    }
  }
```

The KNN matcher is implemented by calling `matcher->knnMatch(...)` function of the matcher object.

## MP.6 Descriptor Distance Ratio

Distance ratio is used to filter the matches by iterating over each match pair from the KNN(2) matcher, and if there is a match and the distance ratio between match 0 and match 1 is less than the threshold (0.8) then the first match is added to the match vector.

```
for (const auto &match : knnMatches)
{
    if (match.size() == 2 && (match[0].distance < descDistRatioThreshold * match[1].distance))
    {
    matches.push_back(match[0]);
    }
}
```

## MP.7,8,9 Performance Evaluation 1/2/3

This is done by logging to cout and using a python script to extact the data. The data is presented in [task7.csv](task7.csv) and [task8_9.csv](task8_9.csv).

The top3 detector/descriptor combinations were evaluated by analysing a combination of processing times (for real-time implementation) and number of matches:

* FAST+ORB
* FAST+BRIEF
* FAST+BRISK

SIFT based detector/descriptors gave high number of matches but were the slowest and are not suitable for this usecase. 