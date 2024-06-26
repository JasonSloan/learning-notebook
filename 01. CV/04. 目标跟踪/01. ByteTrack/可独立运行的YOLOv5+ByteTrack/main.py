import cv2
from tqdm import tqdm
from argparse import ArgumentParser

from yolov5.infer_controller import InferenceController
from bytetrack.byte_tracker import BYTETracker


def draw_rectangles(output_results, frame):
    output_results = output_results[output_results[..., 4] > 0.2]
    for result in output_results:
        point1 = (int(result[0]), int(result[1]))
        point2 = (int(result[2]), int(result[3]))
        cv2.rectangle(frame, point1, point2, (255, 0, 0), 2)
        cv2.imwrite("output.jpg", frame)


def track(frame, tgt_cls, detector, tracker):
    im0_size = frame.shape[:2]
    output_results = detector.infer(frame)
    output_results = output_results[output_results[..., 5] == tgt_cls][..., :5]    # only keep tgt_cls (person)
    # draw_rectangles(output_results, frame)
    tracks = tracker.update(output_results, im0_size, im0_size)
    return tracks


def synthesis_video_with_trackids(video_path, frame_ids, tlwhs, trackids, output_video_path, draw_trajectory):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4 format
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    pbar = tqdm(range(int(frame_count)), desc="Synthesis video")
    trajectory_points = {}
    for i in pbar:
        ret, frame = cap.read()
        if not ret:
            break
        if i not in frame_ids:
            continue
        index = frame_ids.index(i)
        itlwhs = tlwhs[index]
        itids = trackids[index]
        for j in range(len(itlwhs)):
            point1 = (int(itlwhs[j][0]), int(itlwhs[j][1]))
            point2 = (int(itlwhs[j][0] + itlwhs[j][2]), int(itlwhs[j][1] + itlwhs[j][3]))
            cv2.rectangle(frame, point1, point2, (0, 0, 255), 2)
            cv2.putText(frame, f'{itids[j]}', (int(itlwhs[j][0]), int(itlwhs[j][1]) - 10), 0, 1, (0, 255, 255), 2)
            if itids[j] not in trajectory_points.keys():
                trajectory_points[itids[j]] = []
            xc, yc = (int(itlwhs[j][0] + itlwhs[j][2] / 2), int(itlwhs[j][1] + itlwhs[j][3] / 2))
            trajectory_points[itids[j]].append([xc, yc])
            if draw_trajectory:
                if len(trajectory_points[itids[j]]) > 1:
                        for i in range(1, len(trajectory_points[itids[j]])):
                            cv2.line(frame, trajectory_points[itids[j]][i-1], trajectory_points[itids[j]][i], (0, 255, 0), 2)
        out.write(frame)
    out.release()
    cap.release()


def main(video_path, cfg, weight, tgt_cls, conf_thre, iou_thre, input_size, track_thresh, 
         match_thresh, track_buffer, min_box_area, output_video_path, draw_trajectory):
    detector = InferenceController(cfg, weight, conf_thre, iou_thre, input_size)
    tracker = BYTETracker(track_thresh, match_thresh, track_buffer, frame_rate=30)

    cap = cv2.VideoCapture(video_path)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    video_bar = tqdm(range(int(frame_count)), desc="Tracking")
    frame_ids, online_tlwhs, online_ids, online_scores = [], [], [], []
    for vi in video_bar:
        ret, frame = cap.read()
        if not ret:
            break
        online_targets = track(frame, tgt_cls, detector, tracker)
        saved = False
        ionline_tlwhs, ionline_ids, ionline_scores = [], [] ,[]
        for t in online_targets:
            tlwh = t.tlwh
            tid = t.track_id
            if tlwh[2] * tlwh[3] > min_box_area:
                saved = True
                ionline_tlwhs.append(tlwh)
                ionline_ids.append(tid)
                ionline_scores.append(t.score)
        if saved:
            frame_ids.append(vi)
            online_tlwhs.append(ionline_tlwhs)
            online_ids.append(ionline_ids)
            online_scores.append(ionline_scores)
        curr_track_counts = len(ionline_scores)
        video_bar.set_description(f"Frame {vi}: {curr_track_counts} tracks")
    cap.release()

    synthesis_video_with_trackids(video_path, frame_ids, online_tlwhs, online_ids, output_video_path, draw_trajectory)
        

def parse_args():
    parser = ArgumentParser(description="Tracking demo. 注意!!!!!!权重文件应该只保存state_dict而不是整个模型, 否则会报错")
    parser.add_argument("--video_path", type=str, default="videos/palace.mp4",help="Path to your video")
    parser.add_argument("--cfg", type=str, default="cfg/yolov5s.yaml", help="Path to the yolov5 detector config file")
    parser.add_argument("--weight", type=str, default="weights/yolov5s.pt", help="Path to the yolov5 detector weight()")
    parser.add_argument("--tgt_cls", type=int, default=0, help="Target class index to track(default 0 means track person only)")
    parser.add_argument("--conf_thre", type=float, default=0.01, help="Confidence threshold of the detector(this value should be extremely small)")
    parser.add_argument("--iou_thre", type=float, default=0.3, help="Detectior nms iou threshold")
    parser.add_argument("--input_size", type=int, nargs='+', default=[384, 640], help="Detector input size")
    parser.add_argument("--track_thresh", type=float, default=0.5, help="Threshold for the score of the track, greater than it will be considered as high scores detections")
    parser.add_argument("--match_thresh", type=float, default=0.9, help="Threshold for the iou between boxes of two frames, " 
                        "if the iou between box0 from T0 and box1 from T1 is greater than it, then we say they are matched")
    parser.add_argument("--track_buffer", type=int, default=30, help="Number of frames that a track is kept without detection,"
                        "lost tracks exceed ? frames will be deleted")
    parser.add_argument("--min_box_area", type=int, default=400, help="The min area of a bounding box to be considered as a target")
    parser.add_argument("--output_video_path", type=str, default="outputs/synthesis_video.mp4", help="Path to the output video")
    parser.add_argument("--draw_trajectory", action="store_true", help="draw trajectory on the synthesis_video or not")
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(**vars(args))
