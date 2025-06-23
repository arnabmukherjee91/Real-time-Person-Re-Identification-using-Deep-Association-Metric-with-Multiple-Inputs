# vim: expandtab:ts=4:sw=4
from __future__ import absolute_import
import numpy as np
from . import kalman_filter
from . import linear_assignment
from . import iou_matching
from .track import Track
from . import mqtt
from .cosinedistance import cosine_distance, search

people = {}


class Tracker:
    """
    This is the multi-target tracker.

    Parameters
    ----------
    metric : nn_matching.NearestNeighborDistanceMetric
        A distance metric for measurement-to-track association.
    max_age : int
        Maximum number of missed misses before a track is deleted.
    n_init : int
        Number of consecutive detections before the track is confirmed. The
        track state is set to `Deleted` if a miss occurs within the first
        `n_init` frames.

    Attributes
    ----------
    metric : nn_matching.NearestNeighborDistanceMetric
        The distance metric used for measurement to track association.
    max_age : int
        Maximum number of missed misses before a track is deleted.
    n_init : int
        Number of frames that a track remains in initialization phase.
    kf : kalman_filter.KalmanFilter
        A Kalman filter to filter target trajectories in image space.
    tracks : List[Track]
        The list of active tracks at the current time step.

    """

    def __init__(self, metric, max_iou_distance=0.7, max_age=500, n_init=3):
        self.metric = metric
        self.max_iou_distance = max_iou_distance
        self.max_age = max_age
        self.n_init = n_init

        self.kf = kalman_filter.KalmanFilter()
        self.tracks = []
        self._next_id = 50

    def predict(self):
        """Propagate track state distributions one time step forward.

        This function should be called once every time step, before `update`.
        """
        for track in self.tracks:
            track.predict(self.kf)

    def update(self, detections):
        """Perform measurement update and track management.

        Parameters
        ----------
        detections : List[deep_sort.detection.Detection]
            A list of detections at the current time step.

        """
        global people

        # Run matching cascade.
        matches, unmatched_tracks, unmatched_detections = \
            self._match(detections)

        # Update track set.
        for track_idx, detection_idx in matches:
            self.tracks[track_idx].update(
                self.kf, detections[detection_idx])
        for track_idx in unmatched_tracks:
            self.tracks[track_idx].mark_missed()
        for detection_idx in unmatched_detections:
            self._initiate_track(detections[detection_idx])
            tmp = self.tracks[-1]

            if (tmp.track_id in people):
                people[tmp.track_id] = tmp.features[:1]
            else:
                # people.setdefault(tmp.mean,tmp.covariance,tmp.track_id,tmp.hits,tmp.age,tmp.time_since_update,tmp.state,tmp.features[:1],tmp._n_init,tmp._max_age)
                # mqtt.deliver(tmp.mean,tmp.covariance,tmp.track_id,tmp.hits,tmp.age,tmp.time_since_update,tmp.state,tmp.features[:1],tmp._n_init,tmp._max_age)
                mqtt.deliver(tmp.mean, tmp.covariance, tmp.track_id, tmp.hits, tmp.age, tmp.time_since_update,
                             tmp.state, tmp.features[:1], tmp._n_init, tmp._max_age)

        self.tracks = [t for t in self.tracks if not t.is_deleted()]

        # receive features
        '''
        temp = mqtt.temp.copy()
        if(len(temp) > 0):
            for k, v,  in temp.items():
                key = int(k)
                value = np.asarray(v)
                print('hello david')
                #print(key)
                #print(value[2])

                if(key in people):
                    people[key] = value
                else:
                    people.setdefault(key, value)
            #mqtt.temp.clear()

        ################################
        for track in self.tracks:
            if(len(track.features) > 0):
                if(track.track_id in people):
                    people[track.track_id] = track.features[:1]
                    mqtt.deliver(track.track_id, track.features[:1])
                else:
                    people.setdefault(track.track_id, track.features[:1])
        ################################
        '''
        # for i, f in people.items():
        #   print(i, f)

        # Update distance metric.
        active_targets = [t.track_id for t in self.tracks if t.is_confirmed()]
        features, targets = [], []
        for track in self.tracks:
            if not track.is_confirmed():
                continue
            features += track.features
            # print("hello welcome to track object")
            # print(track.features)
            targets += [track.track_id for _ in track.features]
            track.features = []
        self.metric.partial_fit(
            np.asarray(features), np.asarray(targets), active_targets)

    def _match(self, detections):

        def gated_metric(tracks, dets, track_indices, detection_indices):
            features = np.array([dets[i].feature for i in detection_indices])
            targets = np.array([tracks[i].track_id for i in track_indices])
            cost_matrix = self.metric.distance(features, targets)
            cost_matrix = linear_assignment.gate_cost_matrix(
                self.kf, cost_matrix, tracks, dets, track_indices,
                detection_indices)

            return cost_matrix

        # Split track set into confirmed and unconfirmed tracks.
        confirmed_tracks = [
            i for i, t in enumerate(self.tracks) if t.is_confirmed()]
        unconfirmed_tracks = [
            i for i, t in enumerate(self.tracks) if not t.is_confirmed()]

        # Associate confirmed tracks using appearance features.
        matches_a, unmatched_tracks_a, unmatched_detections = \
            linear_assignment.matching_cascade(
                gated_metric, self.metric.matching_threshold, self.max_age,
                self.tracks, detections, confirmed_tracks)

        # Associate remaining tracks together with unconfirmed tracks using IOU.
        iou_track_candidates = unconfirmed_tracks + [
            k for k in unmatched_tracks_a if
            self.tracks[k].time_since_update == 1]
        unmatched_tracks_a = [
            k for k in unmatched_tracks_a if
            self.tracks[k].time_since_update != 1]
        matches_b, unmatched_tracks_b, unmatched_detections = \
            linear_assignment.min_cost_matching(
                iou_matching.iou_cost, self.max_iou_distance, self.tracks,
                detections, iou_track_candidates, unmatched_detections)

        matches = matches_a + matches_b
        unmatched_tracks = list(set(unmatched_tracks_a + unmatched_tracks_b))
        return matches, unmatched_tracks, unmatched_detections

    def _initiate_track(self, detection):
        mean, covariance = self.kf.initiate(detection.to_xyah())

        # receive features
        temp = mqtt.temp.copy()

        if (len(temp) > 0):
            for k, v in temp.items():
                key = int(k)
                value_6 = np.asarray(v[6])
                value_0 = np.asarray(v[0])
                value_1 = np.asarray(v[1])

                print("hello")
                print(len(value_6[0]))
                # mean,covariance,self._next_id,self.n_init,self.max_age,detection.feature=np.asarray(value[0]),np.asarray(value[1]),key,value[7],value[8],value[6]
                self._next_id, detection.feature = key, value_6[0]
                self.tracks.append(Track(mean, covariance, self._next_id, self.n_init, self.max_age, detection.feature))
                mqtt.temp.clear()

        '''
        global people
        distances = []
        if(len(people) > 0):
            for k, v in people.items():
                distances.append(cosine_distance([detection.feature], v))
            tmp = search(distances, 0.35)
            if(tmp == None):
                self._next_id = len(people) + 1
            else:
                self._next_id = tmp + 1
            distances = []
            '''
        self.tracks.append(Track(
            mean, covariance, self._next_id, self.n_init, self.max_age,
            detection.feature))
        self._next_id += 1
