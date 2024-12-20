// ------示例代码------
#include <stdio.h>
#include <string>

#include "opencv2/opencv.hpp"
#include "bytetrack/BYTETracker.h"


using namespace std;
using namespace cv;
	

int main(){
	// create object detector
	// ......
	
	// create tracker
	int fps = 30; int track_buffer = 30; 
	float track_thre = 0.4; float high_thre = 0.4; float match_thre = 0.8;
	BYTETracker tracker(fps, track_buffer, track_thre, high_thre, match_thre);

	// start track
	for (int i : tq::trange(total_frames)){
		// read frame
		// ......
		
		// infer with object detector and get bboxes
		// ......
		
		// transform bboxes to objects
		vector<Object> objects;
		int n_bboxes = bboxes.size();
		for (int j = 0; j < n_bboxes; ++j){
			Object object;
			float left = bboxes[j].left;
			float top = bboxes[j].top;
			float right = bboxes[j].right;
			float bottom = bboxes[j].bottom;
			float w = right - left;
			float h = bottom - top;
			ObRect rect = {left, top, w, h};
			object.rect = rect;
			object.prob = bboxes[j].score;
			object.label = bboxes[j].label;
			objects.push_back(object);
		}
		
		// track
		auto tracks = tracker.update(objects);

		// transform back
		for (int i = 0; i < tracks.size(); ++i){
			Box box_one;
			auto &track = tracks[i];
			box_one.left = (int)track.tlbr[0];
			box_one.top = (int)track.tlbr[1];
			box_one.right = (int)track.tlbr[2];
			box_one.bottom = (int)track.tlbr[3];
			box_one.score = track.score;
			box_one.label = track.label;
			box_one.track_id = track.track_id;
			bboxes.push_back(box_one);
		}
		
		// save image
		draw_rectangles(bboxes, frame, save_path);
}




