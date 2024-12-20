#pragma once

#include <string>

#include "bytetrack/STrack.h"

struct ObRect{
	float x;			// topx
	float y;			// lefty
	float width;
	float height;
};

struct Object{
    ObRect rect;
    int label;
    float prob;
};

class BYTETracker
{
public:
	BYTETracker(int frame_rate, int track_buffer, float track_thre, float high_thre, float match_thre);
	~BYTETracker();

	std::vector<STrack> update(const std::vector<Object>& objects);
	std::vector<STrack> update(const std::vector<Object>& objects, unsigned long long ts[100][5], int count);

private:
	std::vector<STrack*> joint_stracks(std::vector<STrack*> &tlista, std::vector<STrack> &tlistb);
	std::vector<STrack> joint_stracks(std::vector<STrack> &tlista, std::vector<STrack> &tlistb);

	std::vector<STrack> sub_stracks(std::vector<STrack> &tlista, std::vector<STrack> &tlistb);
	void remove_duplicate_stracks(std::vector<STrack> &resa, std::vector<STrack> &resb, std::vector<STrack> &stracksa, std::vector<STrack> &stracksb);

	void linear_assignment(std::vector<std::vector<float> > &cost_matrix, int cost_matrix_size, int cost_matrix_size_size, float thresh,
		std::vector<std::vector<int> > &matches, std::vector<int> &unmatched_a, std::vector<int> &unmatched_b);
	std::vector<std::vector<float> > iou_distance(std::vector<STrack*> &atracks, std::vector<STrack> &btracks, int &dist_size, int &dist_size_size);
	std::vector<std::vector<float> > iou_distance(std::vector<STrack> &atracks, std::vector<STrack> &btracks);
	std::vector<std::vector<float> > ious(std::vector<std::vector<float> > &atlbrs, std::vector<std::vector<float> > &btlbrs);

	double lapjv(const std::vector<std::vector<float> > &cost, std::vector<int> &rowsol, std::vector<int> &colsol, 
		bool extend_cost = false, float cost_limit = LONG_MAX, bool return_cost = true);

private:

	float track_thresh;							// Threshold for the score of the track, greater than it will be considered as high scores detections
	float high_thresh;							// Threshold for the new track, detections(unmatched) confidence that greater than it will be considered as a new track
	float match_thresh;							// Threashold for matching, IoU greater than it will be considered as a match
	int frame_id;
	int max_time_lost;

	std::vector<STrack> tracked_stracks;
	std::vector<STrack> lost_stracks;
	std::vector<STrack> removed_stracks;
	byte_kalman::KalmanFilter kalman_filter;
};