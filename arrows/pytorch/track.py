class track_state(object):
    def __init__(self, bbox_center, grid_feature, app_feature):
        self._bbox_center = bbox_center
        self._grid_feature = grid_feature
        self._app_feature = app_feature

    @property
    def bbox_center(self):
        return self._bbox_center

    @bbox_center.setter
    # bbox_center is a tuple type
    def bbox_center(self, val):
        self._bbox_center = val

    @property
    def grid_feature(self):
        return self._grid_feature

    @grid_feature.setter
    def grid_feature(self, val):
        self._grid_feature = val

    @property
    def app_feature(self):
        return self._app_feature

    @app_feature.setter
    def app_feature(self, val):
        self._app_feature = val

    @property
    def track_ID(self):
        return self._track_ID

    @track_ID.setter
    def track_ID(self, val):
        self._track_ID = self._track_ID

    @property
    def frame_num(self):
        return self._frame_num

    @frame_num.setter
    def frame_num(self, val):
        self._frame_num = val


class track(object):
    def __init__(self, id):
        self._track_id = id
        self._track_state_list = []

    def __len__(self):
        return len(self._track_state_list)

    def __getitem__(self, idx):
        return self._track_state_list[idx]

    def append(self, new_track_state):
        self._track_state_list.append(new_track_state)



