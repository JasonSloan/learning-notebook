import numpy as np
from collections import deque
import os
import os.path as osp
import copy
import torch
import torch.nn.functional as F

from .kalman_filter import KalmanFilter
from yolox.tracker import matching
from .basetrack import BaseTrack, TrackState

class STrack(BaseTrack):
    shared_kalman = KalmanFilter()
    def __init__(self, tlwh, score):

        # wait activate
        self._tlwh = np.asarray(tlwh, dtype=np.float32)
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.is_activated = False

        self.score = score
        self.tracklet_len = 0

    def predict(self):
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[7] = 0
        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)

    @staticmethod
    def multi_predict(stracks):
        # 卡尔曼滤波预测跟踪框的新位置
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])
            for i, st in enumerate(stracks):
                if st.state != TrackState.Tracked:
                    multi_mean[i][7] = 0
            multi_mean, multi_covariance = STrack.shared_kalman.multi_predict(multi_mean, multi_covariance)
            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                stracks[i].mean = mean
                stracks[i].covariance = cov

    def activate(self, kalman_filter, frame_id):
        """Start a new tracklet"""
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()
        self.mean, self.covariance = self.kalman_filter.initiate(self.tlwh_to_xyah(self._tlwh))     # 生成初始化的mean和covariance

        self.tracklet_len = 0
        self.state = TrackState.Tracked
        if frame_id == 1:   # 第一帧
            self.is_activated = True
        # self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track, frame_id, new_id=False):
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_track.tlwh)
        )
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()
        self.score = new_track.score

    def update(self, new_track, frame_id):
        """
        Update a matched track
        :type new_track: STrack
        :type frame_id: int
        :type update_feature: bool
        :return:
        """
        self.frame_id = frame_id
        self.tracklet_len += 1
        new_tlwh = new_track.tlwh
        # 更新该跟踪框的卡尔曼滤波器的均值和协方差矩阵
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_tlwh))
        self.state = TrackState.Tracked
        self.is_activated = True

        self.score = new_track.score

    @property
    # @jit(nopython=True)
    def tlwh(self):
        """Get current position in bounding box format `(top left x, top left y,
                width, height)`.
        """
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    @property
    # @jit(nopython=True)
    def tlbr(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @staticmethod
    # @jit(nopython=True)
    def tlwh_to_xyah(tlwh):
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    def to_xyah(self):
        return self.tlwh_to_xyah(self.tlwh)

    @staticmethod
    # @jit(nopython=True)
    def tlbr_to_tlwh(tlbr):
        ret = np.asarray(tlbr).copy()
        ret[2:] -= ret[:2]
        return ret

    @staticmethod
    # @jit(nopython=True)
    def tlwh_to_tlbr(tlwh):
        ret = np.asarray(tlwh).copy()
        ret[2:] += ret[:2]
        return ret

    # def __repr__(self):
    #     return 'OT_{}_({}-{})'.format(self.track_id, self.start_frame, self.end_frame)


class BYTETracker(object):
    def __init__(self, args, frame_rate=30):
        self.tracked_stracks = []  # type: list[STrack]
        self.lost_stracks = []  # type: list[STrack]
        self.removed_stracks = []  # type: list[STrack]

        self.frame_id = 0
        self.args = args
        #self.det_thresh = args.track_thresh
        self.det_thresh = args.track_thresh + 0.1
        self.buffer_size = int(frame_rate / 30.0 * args.track_buffer)
        self.max_time_lost = self.buffer_size
        self.kalman_filter = KalmanFilter()

    def update(self, output_results, img_info, img_size):
        self.frame_id += 1
        activated_stracks = []  # 已被匹配上的跟踪框
        refind_stracks = []     # 丢失的跟踪框又被重新匹配的
        lost_stracks = []       # 丢失的跟踪框
        removed_stracks = []    # 丢失的跟踪框中超时被移除的

        if output_results.shape[1] == 5:
            scores = output_results[:, 4]
            bboxes = output_results[:, :4]
        else:
            output_results = output_results.cpu().numpy()
            scores = output_results[:, 4] * output_results[:, 5]
            bboxes = output_results[:, :4]  # x1y1x2y2
        img_h, img_w = img_info[0], img_info[1]
        scale = min(img_size[0] / float(img_h), img_size[1] / float(img_w))
        bboxes /= scale

        remain_inds = scores > self.args.track_thresh
        inds_low = scores > 0.1
        inds_high = scores < self.args.track_thresh

        inds_second = np.logical_and(inds_low, inds_high)   # 0.1 < inds_second < 0.5
        dets_second = bboxes[inds_second]       # 低分框
        dets = bboxes[remain_inds]              # 高分框
        scores_keep = scores[remain_inds]       # 高分框分数
        scores_second = scores[inds_second]     # 低分框分数

        if len(dets) > 0:
            '''Detections'''
            # 第一步先对高分框进行匹配，先初始化跟踪器，传入高分框和高分框分数
            detections = [STrack(STrack.tlbr_to_tlwh(tlbr), s) for
                          (tlbr, s) in zip(dets, scores_keep)]
        else:
            detections = []

        ''' Add newly detected tracklets to tracked_stracks'''
        unconfirmed = []
        tracked_stracks = []  # type: list[STrack]
        for track in self.tracked_stracks:          # 如果不是第一次匹配走这步，这一步的目的是将跟踪框中上次被匹配上的框与丢失的框分开
            if not track.is_activated:  # 如果跟踪框已经丢失，但是还没有超过最大丢失帧数，这时候会走这步
                unconfirmed.append(track)        # 将丢失的跟踪框加入unconfirmed列表
            else:
                tracked_stracks.append(track)   # 将上一次被匹配上的跟踪框加入tracked_stracks列表

        ''' Step 2: First association, with high score detection boxes'''
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)     # 将上一次的跟踪框与丢失框进行合并
        # Predict the current location with KF
        STrack.multi_predict(strack_pool)           # 卡尔曼滤波预测新位置得到预测框（静态方法，类直接调用，与实例无关）
        dists = matching.iou_distance(strack_pool, detections)      # 计算预测框与检测框的1-IOU，得到矩阵[M, N]，M为预测框数，N为检测框数
        if not self.args.mot20:
            dists = matching.fuse_score(dists, detections)         # 将IOU矩阵与检测框分数矩阵进行融合（想乘），得到最终的距离矩阵
        # 线性分配（给定一个N*N的代价矩阵，使分配后的总代价值最小），得到匹配结果
        # matches为匹配结果，u_track为未匹配的预测框，u_detection为未匹配的检测框
        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=self.args.match_thresh)


        for itracked, idet in matches:          # 已匹配的跟踪框和检测框要更新激活状态，卡尔曼滤波的均值和方差
            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                # 更新跟踪框的激活状态，卡尔曼滤波的均值和方差
                track.update(detections[idet], self.frame_id)
                activated_stracks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        ''' Step 3: Second association, with low score detection boxes'''
        # association the untrack to the low score detections
        if len(dets_second) > 0:
            detections_second = [STrack(STrack.tlbr_to_tlwh(tlbr), s) for
                          (tlbr, s) in zip(dets_second, scores_second)]
        else:
            detections_second = []
        # r_tracked_stracks：从高分框匹配剩下的框中选择剩余未匹配的预测框
        r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
        # 剩余未匹配的预测框与低分框计算iou
        dists = matching.iou_distance(r_tracked_stracks, detections_second)
        # 线性分配
        matches, u_track, u_detection_second = matching.linear_assignment(dists, thresh=0.5)
        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections_second[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_stracks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)
        # 低分框匹配完成还剩余的跟踪框就是丢失的跟踪框，将其标记为丢失状态
        for it in u_track:
            track = r_tracked_stracks[it]
            if not track.state == TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)

        '''Deal with unconfirmed tracks, usually tracks with only one beginning frame'''
        # 将高分检测框中仍未匹配的与上一帧中丢失的跟踪框进行匹配
        detections = [detections[i] for i in u_detection]
        dists = matching.iou_distance(unconfirmed, detections)      # unconfirmed是上一次跟踪框中的丢失框
        if not self.args.mot20:
            dists = matching.fuse_score(dists, detections)
        matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=0.7)
        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet], self.frame_id)
            activated_stracks.append(unconfirmed[itracked])
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)

        """ Step 4: Init new stracks"""
        """检测框中高分框、低分框都已经与跟踪框进行了一轮匹配，
        那么检测框中剩下的还有高分框和低分框，剩下的高分框认为直接就是新的跟踪框，
        剩下的低分框因为没有和跟踪框匹配成功，那么这些低分框就是误检，就可以直接不要了"""
        for inew in u_detection:
            track = detections[inew]
            if track.score < self.det_thresh:
                continue
            track.activate(self.kalman_filter, self.frame_id)
            activated_stracks.append(track)
        """ Step 5: Update state"""
        # 丢失的跟踪框中，如果跟踪框的最后一帧与当前帧的差值大于最大丢失帧数，那么就将其标记为删除状态
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        # print('Ramained match {} s'.format(t4-t3))
        #
        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        # 将已有的跟踪框与新的跟踪框进行合并
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_stracks)
        # ？？？refind_stracks是啥
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
        # 将丢失框中又被匹配到（重新发现）的框去掉
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        # 更新丢失框
        self.lost_stracks.extend(lost_stracks)
        # 将丢失框中已经超时的框去掉
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        # 更新移除框
        self.removed_stracks.extend(removed_stracks)
        # ？？
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)
        # get scores of lost tracks
        output_stracks = [track for track in self.tracked_stracks if track.is_activated]

        return output_stracks


def joint_stracks(tlista, tlistb):
    exists = {}
    res = []
    for t in tlista:
        exists[t.track_id] = 1
        res.append(t)
    for t in tlistb:
        tid = t.track_id
        if not exists.get(tid, 0):
            exists[tid] = 1
            res.append(t)
    return res


def sub_stracks(tlista, tlistb):
    stracks = {}
    for t in tlista:
        stracks[t.track_id] = t
    for t in tlistb:
        tid = t.track_id
        if stracks.get(tid, 0):
            del stracks[tid]
    return list(stracks.values())


def remove_duplicate_stracks(stracksa, stracksb):
    pdist = matching.iou_distance(stracksa, stracksb)
    pairs = np.where(pdist < 0.15)
    dupa, dupb = list(), list()
    for p, q in zip(*pairs):
        timep = stracksa[p].frame_id - stracksa[p].start_frame
        timeq = stracksb[q].frame_id - stracksb[q].start_frame
        if timep > timeq:
            dupb.append(q)
        else:
            dupa.append(p)
    resa = [t for i, t in enumerate(stracksa) if not i in dupa]
    resb = [t for i, t in enumerate(stracksb) if not i in dupb]
    return resa, resb
