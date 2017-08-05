import numpy as np


class track_state(object):
    def __init__(self, bbox_center, interaction_feature, app_feature):
        self._bbox_center = bbox_center

        self._app_feature = app_feature
        self._motion_feature = np.empty([1, 2])
        self._interaction_feature = interaction_feature

        self._track_id = -1
        self._frame_id = -1

    @property
    def bbox_center(self):
        return self._bbox_center

    @bbox_center.setter
    # bbox_center is a tuple type
    def bbox_center(self, val):
        self._bbox_center = val

    @property
    def app_feature(self):
        return self._app_feature

    @app_feature.setter
    def app_feature(self, val):
        self._app_feature = val

    @property
    def motion_feature(self):
        return self._motion_feature

    @motion_feature.setter
    def motion_feature(self, val):
        self._motion_feature = val

    @property
    def interaction_feature(self):
        return self._interaction_feature

    @interaction_feature.setter
    def interaction_feature(self, val):
        self._interaction_feature = val

    @property
    def track_id(self):
        return self._track_id

    @track_id.setter
    def track_id(self, val):
        self._track_id = val

    @property
    def frame_id(self):
        return self._frame_id

    @frame_id.setter
    def frame_id(self, val):
        self._frame_id = val


class track(object):
    def __init__(self, id):
        self._track_id = id
        self._track_state_list = []

    def __len__(self):
        return len(self._track_state_list)

    def __getitem__(self, idx):
        if idx >= len(self._track_state_list):
            raise IndexError

        return self._track_state_list[idx]

    @property
    def id(self):
        return self._track_id

    @id.setter
    def id(self, val):
        self._track_id = val

    def append(self, new_track_state):
        if len(self._track_state_list) == 0:
            new_track_state.motion_feature = np.array([[0.0, 0.0]])
        else:
            pre_bbox_center = np.asarray(self._track_state_list[-1].bbox_center).reshape(1, 2)
            cur_bbox_center = np.asarray(new_track_state.bbox_center).reshape(1, 2)
            new_track_state._motion_feature = cur_bbox_center - pre_bbox_center

        self._track_state_list.append(new_track_state)


class track_set(object):
    def __init__(self):
        self._id_ts_dict = {}

    def __len__(self):
        return len(self._id_ts_dict)

    def __getitem__(self, idx):
        if idx >= len(self._id_ts_dict):
            raise IndexError
        return self._id_ts_dict.items()[idx][1]

    def get_track(self, track_id):
        if track_id not in self._id_ts_dict:
            raise IndexError

        return self._id_ts_dict[track_id]

    def get_all_trackID(self):
        return sorted(self._id_ts_dict.keys())
    
    def get_max_track_ID(self):
        if len(self._id_ts_dict) == 0:
            return 0
        else:
            return max(self.get_all_trackID())

    def add_new_track(self, track):
        if track.track_id is in self.get_all_trackID():
            print("track ID exsit in the track set!!!")
            raise RuntimeError

        self._id_ts_dict[track.track_id] = track
    
    def add_new_track_state(self, track_id, track_state):
        if track_id is in self.get_all_trackID():
            print("track ID exsit in the track set!!!")
            raise RuntimeError
        
        new_track = track(track_id)
        new_track.append(track_state)
        self._id_ts_dict[track_id] = new_track


    def update_track(self, track_id, new_track_state):
        if track_id not in self._id_ts_dict:
            raise IndexError

        self._id_ts_dict[track_id].append(new_track_state)


if __name__ == '__main__':
    t = track(0)
    for i in range(10):
        t.append(track_state((i, i*i*0.1), [], []))

    for item in t[:]:
        print(item.motion_feature)

