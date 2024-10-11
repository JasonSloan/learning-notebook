#include "opencv2/opencv.hpp"

#include "bytetrack/BYTETracker.h"
#include <fstream>

using namespace std;

BYTETracker::BYTETracker(int frame_rate, int track_buffer, float track_thre, float high_thre, float match_thre){
	max_time_lost = int(frame_rate / 30.0 * track_buffer);
	track_thresh = track_thre;								// Threshold for the score of the track, greater than it will be considered as high scores detections
	high_thresh = high_thre;								// Threshold for the new track, detections(unmatched) confidence that greater than it will be considered as a new track
	match_thresh = match_thre;								// Threashold for matching, IoU greater than it will be considered as a match
	frame_id = 0;
}

BYTETracker::~BYTETracker(){}

vector<STrack> BYTETracker::update(const vector<Object>& objects){

	// ------------------step1: Get Detections------------------
	this->frame_id++;
	vector<STrack> activated_stracks;				// stores activated tracked tracks(first matched + second matched + new track)
	vector<STrack> refind_stracks;					// tracks that are lost from previous frames but refind in current frame
	vector<STrack> removed_stracks;					// unmatched unconfirmed tracks(after all the associations) after association with remained high scores detections + lost tracks that exceed max_time 
	vector<STrack> lost_stracks;					// unmatched tracks(after all the associations0
	vector<STrack> detections;						// high scores detections
	vector<STrack> detections_low;					// low scores detections

	vector<STrack> detections_cp;					// remained high scores detections after first association
	vector<STrack> tracked_stracks_swap;			// useless
	vector<STrack> resa, resb;
	vector<STrack> output_stracks;					// current frame matched tracks that to be returned out

	vector<STrack*> unconfirmed;					// unconfirmed tracks of tracked tracks
	vector<STrack*> tracked_stracks;				// tracked tracks(including activated and unconfirmed)
	vector<STrack*> strack_pool;					// tracked tracks combined with lost tracks
	vector<STrack*> r_tracked_stracks;				// remained tracked tracks after second association

	if (objects.size() > 0){
		for (int i = 0; i < objects.size(); i++){
			vector<float> tlbr_;
			tlbr_.resize(4);
			tlbr_[0] = objects[i].rect.x;
			tlbr_[1] = objects[i].rect.y;
			tlbr_[2] = objects[i].rect.x + objects[i].rect.width;
			tlbr_[3] = objects[i].rect.y + objects[i].rect.height;

			float score = objects[i].prob;
			int label = objects[i].label;

			STrack strack(STrack::tlbr_to_tlwh(tlbr_), score, label);
			if (score >= track_thresh)
				detections.push_back(strack);
			else
				detections_low.push_back(strack);
		}
	}

	// divide last frame's tracked_stracks into unconfirmed and tracked_stracks
	for (int i = 0; i < this->tracked_stracks.size(); i++){
		if (!this->tracked_stracks[i].is_activated)
			unconfirmed.push_back(&this->tracked_stracks[i]);
		else
			tracked_stracks.push_back(&this->tracked_stracks[i]);
	}

	// ------------------step2: First association, using high scores detections------------------
	strack_pool = joint_stracks(tracked_stracks, this->lost_stracks);
	STrack::multi_predict(strack_pool, this->kalman_filter);

	vector<vector<float>> dists;					// [m, n], m for last frame's tracks, n for current frame's detections
	int dist_size = 0, dist_size_size = 0;
	dists = iou_distance(strack_pool, detections, dist_size, dist_size_size);

	vector<vector<int>> matches;
	vector<int> u_track, u_detection;
	linear_assignment(dists, dist_size, dist_size_size, match_thresh, matches, u_track, u_detection);

	for (int i = 0; i < matches.size(); i++){
		STrack *track = strack_pool[matches[i][0]];
		STrack *det = &detections[matches[i][1]];
		if (track->state == TrackState::Tracked){
			track->update(*det, this->frame_id);
			activated_stracks.push_back(*track);
		}else{
			track->re_activate(*det, this->frame_id, false);
			refind_stracks.push_back(*track);
		}
	}

	// ------------------step3: Second association, using low scores detections------------------
	for (int i = 0; i < u_detection.size(); i++)
		detections_cp.push_back(detections[u_detection[i]]);
	detections.clear();
	detections.assign(detections_low.begin(), detections_low.end());
	
	for (int i = 0; i < u_track.size(); i++)
		if (strack_pool[u_track[i]]->state == TrackState::Tracked)
			r_tracked_stracks.push_back(strack_pool[u_track[i]]);

	dists.clear();
	dists = iou_distance(r_tracked_stracks, detections, dist_size, dist_size_size);

	matches.clear();
	u_track.clear();
	u_detection.clear();
	linear_assignment(dists, dist_size, dist_size_size, 0.5, matches, u_track, u_detection);

	for (int i = 0; i < matches.size(); i++){
		STrack *track = r_tracked_stracks[matches[i][0]];
		STrack *det = &detections[matches[i][1]];
		if (track->state == TrackState::Tracked){
			track->update(*det, this->frame_id);
			activated_stracks.push_back(*track);
		}else{
			track->re_activate(*det, this->frame_id, false);
			refind_stracks.push_back(*track);
		}
	}

	for (int i = 0; i < u_track.size(); i++){
		STrack *track = r_tracked_stracks[u_track[i]];
		if (track->state != TrackState::Lost){
			track->mark_lost();
			lost_stracks.push_back(*track);
		}
	}

	// Deal with unconfirmed tracks, usually tracks with only one beginning frame
	detections.clear();
	detections.assign(detections_cp.begin(), detections_cp.end());

	dists.clear();
	dists = iou_distance(unconfirmed, detections, dist_size, dist_size_size);

	matches.clear();
	vector<int> u_unconfirmed;
	u_detection.clear();
	linear_assignment(dists, dist_size, dist_size_size, 0.7, matches, u_unconfirmed, u_detection);

	for (int i = 0; i < matches.size(); i++){
		unconfirmed[matches[i][0]]->update(detections[matches[i][1]], this->frame_id);
		activated_stracks.push_back(*unconfirmed[matches[i][0]]);
	}

	for (int i = 0; i < u_unconfirmed.size(); i++){
		STrack *track = unconfirmed[u_unconfirmed[i]];
		track->mark_removed();
		removed_stracks.push_back(*track);
	}

	// ------------------step4: Init new track------------------
	for (int i = 0; i < u_detection.size(); i++){
		STrack *track = &detections[u_detection[i]];
		if (track->score < this->high_thresh)
			continue;
		track->activate(this->kalman_filter, this->frame_id);
		activated_stracks.push_back(*track);
	}

	// ------------------step5: update state------------------
	for (int i = 0; i < this->lost_stracks.size(); i++)
		if (this->frame_id - this->lost_stracks[i].end_frame() > this->max_time_lost){
			this->lost_stracks[i].mark_removed();
			removed_stracks.push_back(this->lost_stracks[i]);
		}
	
	for (int i = 0; i < this->tracked_stracks.size(); i++)
		if (this->tracked_stracks[i].state == TrackState::Tracked)
			tracked_stracks_swap.push_back(this->tracked_stracks[i]);

	this->tracked_stracks.clear();
	this->tracked_stracks.assign(tracked_stracks_swap.begin(), tracked_stracks_swap.end());

	this->tracked_stracks = joint_stracks(this->tracked_stracks, activated_stracks);
	this->tracked_stracks = joint_stracks(this->tracked_stracks, refind_stracks);

	this->lost_stracks = sub_stracks(this->lost_stracks, this->tracked_stracks);
	for (int i = 0; i < lost_stracks.size(); i++)
		this->lost_stracks.push_back(lost_stracks[i]);

	this->lost_stracks = sub_stracks(this->lost_stracks, this->removed_stracks);
	this->removed_stracks.clear();
	for (int i = 0; i < removed_stracks.size(); i++)
		this->removed_stracks.push_back(removed_stracks[i]);
	
	remove_duplicate_stracks(resa, resb, this->tracked_stracks, this->lost_stracks);

	this->tracked_stracks.clear();
	this->tracked_stracks.assign(resa.begin(), resa.end());
	this->lost_stracks.clear();
	this->lost_stracks.assign(resb.begin(), resb.end());
	
	for (int i = 0; i < this->tracked_stracks.size(); i++)
		if (this->tracked_stracks[i].is_activated)
			output_stracks.push_back(this->tracked_stracks[i]);
	
	return output_stracks;
}


