#include <iostream>
#include <fstream>
#include <sstream>

#include <opencv2/tracking.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

const char* keys =
"{help h usage ? | | Usage examples: \n\t\t./object_detection_yolo.out --image=dog.jpg \n\t\t./object_detection_yolo.out --video=run_sm.mp4}"
"{image i        |<none>| input image   }"
"{video v       |<none>| input video   }"
;

using namespace std;
using namespace cv;
using namespace dnn;

// Initialize the parameters
float confThreshold = 0.5; // Confidence threshold
float nmsThreshold = 0.4;  // Non-maximum suppression threshold
int inpWidth = 416;  // Width of network's input image
int inpHeight = 416; // Height of network's input image
vector<string> classes;

// Remove the bounding boxes with low confidence using non-maxima suppression
void postprocess(Mat& frame, const vector<Mat>& out);

// Draw the predicted bounding box
void drawPred(int classId, float conf, int left, int top, int right, int bottom, Mat& frame);

// Get the names of the output layers
vector<String> getOutputsNames(const Net& net);

int main(int argc, char** argv)
//int main() 
{
	CommandLineParser parser(argc, argv, keys);
	parser.about("Use this script to run object detection using YOLO3 in OpenCV.");
	if (parser.has("help"))
	{
		parser.printMessage();
		return 0;
	}
	// Load names of classes
	string classesFile = "coco.names";
	ifstream ifs(classesFile.c_str());
	string line;
	while (getline(ifs, line)) classes.push_back(line);

	// Give the configuration and weight files for the model
	String modelConfiguration = "yolov3.cfg";
	String modelWeights = "yolov3.weights";

	// Load the network
	Net net = readNetFromDarknet(modelConfiguration, modelWeights);
	net.setPreferableBackend(DNN_BACKEND_OPENCV);
	net.setPreferableTarget(DNN_TARGET_CPU);

	Net net2 = readNetFromDarknet(modelConfiguration, modelWeights);
	net2.setPreferableBackend(DNN_BACKEND_OPENCV);
	net2.setPreferableTarget(DNN_TARGET_CPU);


	Net net3 = readNetFromDarknet(modelConfiguration, modelWeights);
	net3.setPreferableBackend(DNN_BACKEND_OPENCV);
	net3.setPreferableTarget(DNN_TARGET_CPU);


	Net net4 = readNetFromDarknet(modelConfiguration, modelWeights);
	net4.setPreferableBackend(DNN_BACKEND_OPENCV);
	net4.setPreferableTarget(DNN_TARGET_CPU);


	// Open a video file or an image file or a camera stream.
	string str, outputFile, outputFile2, outputFile3, outputFile4;
	VideoCapture cap, cap2, cap3, cap4;
	VideoWriter video, video2, video3, video4;
	Mat frame, frame2, frame3, frame4, blob, blob2, blob3, blob4;
	
	
	try {

		outputFile = "yolo_cam1.avi";
		outputFile2 = "yolo_cam2.avi";
		outputFile3 = "yolo_cam3.avi";		
		outputFile4 = "yolo_cam4.avi";
		
		if (parser.has("image"))
		{
			// Open the image file
			str = parser.get<String>("image");
			ifstream ifile(str);
			if (!ifile) throw("error");
			cap.open(str);
			str.replace(str.end() - 4, str.end(), "_yolo_out_cpp.jpg");
			outputFile = str;
		}
		else if (parser.has("video"))
		{
			// Open the video file
			str = parser.get<String>("video");
			ifstream ifile(str);
			if (!ifile) throw("error");
			cap.open(str);
			str.replace(str.end() - 4, str.end(), "_yolo_out_cpp.avi");
			outputFile = str;
		}
		// Open the webcaom
		else {
			//cap.open(parser.get<int>("device"));
			//cap.open(1);
			//cap2.open(2);
			cap.open("http://192.168.0.3:8080/video");
			cap2.open(1);// ("http://192.168.0.11:8080/video");
		}
	}
	catch (...) {
		cout << "Could not open the input image/video stream" << endl;
		return 0;
	}

	// Get the video writer initialized to save the output video
	if (!parser.has("image")) {
		video.open(outputFile, VideoWriter::fourcc('M', 'J', 'P', 'G'), 15, Size(cap.get(CAP_PROP_FRAME_WIDTH), cap.get(CAP_PROP_FRAME_HEIGHT)));
		video2.open(outputFile2, VideoWriter::fourcc('M', 'J', 'P', 'G'), 15, Size(cap.get(CAP_PROP_FRAME_WIDTH), cap.get(CAP_PROP_FRAME_HEIGHT)));
		//video3.open(outputFile3, VideoWriter::fourcc('M', 'J', 'P', 'G'), 15, Size(cap.get(CAP_PROP_FRAME_WIDTH), cap.get(CAP_PROP_FRAME_HEIGHT)));
		//video4.open(outputFile4, VideoWriter::fourcc('M', 'J', 'P', 'G'), 15, Size(cap.get(CAP_PROP_FRAME_WIDTH), cap.get(CAP_PROP_FRAME_HEIGHT)));
	}
	
	//cap.open(0);

	// Create a window
	static const string kWinName = "Deep learning object detection in OpenCV";
	//namedWindow(kWinName, WINDOW_NORMAL);
	namedWindow("video1", WINDOW_NORMAL);
	namedWindow("video2", WINDOW_NORMAL);
	//namedWindow("video3", WINDOW_NORMAL);
	//namedWindow("video4", WINDOW_NORMAL);

	// Process frames.
	//while (waitKey(1) < 0)
	while(1) 
	{
		// get frame from the video
		cap >> frame;
		cap2 >> frame2;
		//cap3 >> frame3;
		//cap4 >> frame4;

		// Stop the program if reached end of video
		//if (frame.empty() || frame2.empty() || frame3.empty() || frame4.empty()) {
		if (frame.empty() || frame2.empty()) {
			cout << "Done processing !!!" << endl;
			cout << "Output file is stored as " << outputFile << endl;
			waitKey(3000);
			break;
		}

		//GaussianBlur(frame, frame, Size(3, 3), 0, 0, BORDER_DEFAULT);

		//cvtColor(frame, frame, CV_RGB2GRAY);

		//Sobel(frame, frame, CV_8U, 1, 0);

		// Create a 4D blob from a frame.
		//blobFromImage(frame, blob, 1 / 255.0, cvSize(inpWidth, inpHeight), Scalar(0, 0, 0), true, false);
		blobFromImage(frame, blob, 1 / 255.0, Size(inpWidth, inpHeight), Scalar(0, 0, 0), true, false);
		blobFromImage(frame2, blob2, 1 / 255.0, Size(inpWidth, inpHeight), Scalar(0, 0, 0), true, false);
		//blobFromImage(frame3, blob3, 1 / 255.0, Size(inpWidth, inpHeight), Scalar(0, 0, 0), true, false);
		//blobFromImage(frame4, blob4, 1 / 255.0, Size(inpWidth, inpHeight), Scalar(0, 0, 0), true, false);


		//Sets the input to the network
		net.setInput(blob);
		net2.setInput(blob2);
		//net3.setInput(blob3);
		//net4.setInput(blob4);

		// Runs the forward pass to get output of the output layers
		vector<Mat> outs, outs2, outs3, outs4;
		net.forward(outs, getOutputsNames(net));
		net2.forward(outs2, getOutputsNames(net2));
		//net3.forward(outs3, getOutputsNames(net3));
		//net4.forward(outs4, getOutputsNames(net4));


		// Remove the bounding boxes with low confidence
		postprocess(frame, outs);
		postprocess(frame2, outs2);
		//postprocess(frame3, outs3);
		//postprocess(frame4, outs4);


		// Put efficiency information. The function getPerfProfile returns the overall time for inference(t) and the timings for each of the layers(in layersTimes)
		vector<double> layersTimes;
		double freq = getTickFrequency() / ( 1000 * 10) ;
		double t = net.getPerfProfile(layersTimes) / freq;
		string label = format("Inference time for a frame : %.2f ms", t);
		putText(frame, label, Point(0, 15), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 255));
		putText(frame2, label, Point(0, 15), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 255));
		//putText(frame3, label, Point(0, 15), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 255));
		//putText(frame4, label, Point(0, 15), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 255));

		// Write the frame with the detection boxes
		Mat detectedFrame, detectedFrame2, detectedFrame3, detectedFrame4;

		frame.convertTo(detectedFrame, CV_8U);
		frame2.convertTo(detectedFrame2, CV_8U);
		//frame3.convertTo(detectedFrame3, CV_8U);
		//frame4.convertTo(detectedFrame4, CV_8U);


		if (parser.has("image")) imwrite(outputFile, detectedFrame);
		else {
			video.write(detectedFrame);
			video2.write(detectedFrame2);
			//video3.write(detectedFrame3);
			//video4.write(detectedFrame4);

		}
		//imshow(kWinName, frame);
		imshow("video1", frame);
		imshow("video2", frame2);
		//imshow("video3", frame3);
		//imshow("video4", frame4);

		if (waitKey(10) == 27)
		{
			cout << "Goodbye" << endl;
			break;
		}

	}

	cap.release();
	cap2.release();
	//cap3.release();
	//cap4.release();

	if (!parser.has("image")) video.release();

	return 0;
}

// Remove the bounding boxes with low confidence using non-maxima suppression
void postprocess(Mat& frame, const vector<Mat>& outs)
{
	vector<int> classIds;
	vector<float> confidences;
	vector<Rect> boxes;

	for (size_t i = 0; i < outs.size(); ++i)
	{
		// Scan through all the bounding boxes output from the network and keep only the
		// ones with high confidence scores. Assign the box's class label as the class
		// with the highest score for the box.
		float* data = (float*)outs[i].data;
		for (int j = 0; j < outs[i].rows; ++j, data += outs[i].cols)
		{
			Mat scores = outs[i].row(j).colRange(5, outs[i].cols);
			Point classIdPoint;
			double confidence;
			// Get the value and location of the maximum score
			minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);
			if (confidence > confThreshold)
			{
				int centerX = (int)(data[0] * frame.cols);
				int centerY = (int)(data[1] * frame.rows);
				int width = (int)(data[2] * frame.cols);
				int height = (int)(data[3] * frame.rows);
				int left = centerX - width / 2;
				int top = centerY - height / 2;

				classIds.push_back(classIdPoint.x);
				confidences.push_back((float)confidence);
				boxes.push_back(Rect(left, top, width, height));
			}
		}
	}

	// Perform non maximum suppression to eliminate redundant overlapping boxes with
	// lower confidences
	vector<int> indices;
	NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, indices);
	for (size_t i = 0; i < indices.size(); ++i)
	{
		int idx = indices[i];
		Rect box = boxes[idx];
		drawPred(classIds[idx], confidences[idx], box.x, box.y,
			box.x + box.width, box.y + box.height, frame);
	}
}

// Draw the predicted bounding box
void drawPred(int classId, float conf, int left, int top, int right, int bottom, Mat& frame)
{
	//cout << classId << endl;
	//cout << left << endl;

	if (classId == 0 && conf > 0.9) {

		//Draw a rectangle displaying the bounding box
		//rectangle(frame, Point(left, top), Point(right, bottom), Scalar(255, 178, 50), 3);
		rectangle(frame, Point(left, top), Point(right, bottom), Scalar(0, 0, 255), 3);

		//Get the label for the class name and its confidence
		//string label = format("%.2f", conf);
		string label = format("%d,%d", left, top);

		if (!classes.empty())
		{
			CV_Assert(classId < (int)classes.size());
			label = classes[classId] + ":" + label;
		}

		//Display the label at the top of the bounding box
		int baseLine;
		//Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
		Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.3, 1, &baseLine);
		top = max(top, labelSize.height);
		//rectangle(frame, Point(left, top - round(1.5*labelSize.height)), Point(left + round(1.5*labelSize.width), top + baseLine), Scalar(255, 255, 255), FILLED);
		rectangle(frame, Point(left, top - round(1.5*labelSize.height)), Point(left + round(1.5*labelSize.width), top + baseLine), Scalar(0, 255, 255), FILLED);
		//putText(frame, label, Point(left, top), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0, 0, 0), 1);
		putText(frame, label, Point(left, top), FONT_HERSHEY_SIMPLEX, 0.3, Scalar(0, 0, 0), 1);

	}
}

// Get the names of the output layers
vector<String> getOutputsNames(const Net& net)
{
	static vector<String> names;
	if (names.empty())
	{
		//Get the indices of the output layers, i.e. the layers with unconnected outputs
		vector<int> outLayers = net.getUnconnectedOutLayers();

		//get the names of all the layers in the network
		vector<String> layersNames = net.getLayerNames();

		// Get the names of the output layers in names
		names.resize(outLayers.size());
		for (size_t i = 0; i < outLayers.size(); ++i)
			names[i] = layersNames[outLayers[i] - 1];
	}
	return names;
}