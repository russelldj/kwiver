import torch
from torch import nn

import numpy as np

from kwiver.arrows.pytorch.models import TargetLSTM, get_config

TIMESTEP_LEN = 6
g_config = get_config()

class SRNN_matching(object):
    def __init__(self, app_model_path, motion_model_path, interaction_model_path, targetRNN_model_path):
        # load app, motion, interaction LSTM models
        self._targetRNN_model = TargetLSTM(app_model=app_model_path, motion_model=motion_model_path,
                   interaction_model=interaction_model_path).cuda()

        # load target RNN model
        snapshot = torch.load(targetRNN_model_path)
        self._targetRNN_model.load_state_dict(snapshot['state_dict'])
        self._targetRNN_model.train(False)
        self._targetRNN = torch.nn.DataParallel(self._targetRNN).cuda()

    def __call__(self, track_set, track_state_list):
        tracks_num = len(track_set)
        track_states_num = len(track_state_list)

        similarity_mat = np.empty([tracks_num, track_states_num])

        track_idx_list = []

        for t in range(tracks_num):
            cur_track = track_set[t]
            track_idx_list.append(cur_track.track_id)
            for ts in range(track_states_num):
                similarity_mat[t, ts] = self._est_similarity(cur_track, track_state_list[ts])

        return similarity_mat, track_idx_list

    def _est_similarity(self, track, track_state):
        assert(len(track) >= TIMESTEP_LEN)

        in_app_seq, in_app_target, in_motion_seq, in_motion_target, in_interaction_seq, in_interaction_target = \
            self._process_track(track, track_state)

        output = self._targetRNN_model(in_app_seq, in_app_target, in_motion_seq, in_motion_target, in_interaction_seq,
                       in_interaction_target)

        F_softmax = nn.Softmax()
        output = F_softmax(output)
        pred = torch.max(output[:, -1, :], 1)

        pred_lable = pred[1].data.cpu().numpy().squeeze()
        pred_p = pred[0].data.cpu().numpy().squeeze()

        if pred_lable == 0:
            return 1.0
        else:
            return -pred_p

    def _process_track(self, track, track_state):

        for idx, ts in enumerate(track[-TIMESTEP_LEN:]):
            if idx == 0:
                app_f_list = ts.app_feature.reshape(1, g_config.A_F_num)
                motion_f_list = ts.motion_feature.reshape(1, g_config.M_F_num)
                interaction_f_list = ts.interaction_feature.reshape(1, g_config.I_F_num)
            else:
                np.append(app_f_list, ts.app_feature.reshape(1, g_config.A_F_num), axis=0)
                np.append(motion_f_list, ts.motion_feature.reshape(1, g_config.M_F_num), axis=0)
                np.append(interaction_f_list, ts.interaction_feature.reshape(1, g_config.I_F_num), axis=0)

        app_target_f = track_state.app_feature.reshape(1, g_config.A_F_num)
        interaction_target_f = track_state.interaction_feature.reshape(1, g_config.I_F_num)
        motion_target_f = np.asarray(track_state.bbox_center).reshape(1, g_config.M_F_num) - \
                          np.asarray(track[-1].bbox_center).reshape(1, g_config.M_F_num)

        # add batch dim
        app_f_list = np.reshape(app_f_list, (1, TIMESTEP_LEN, -1))
        app_target_f = np.reshape(app_target_f, (1, 1, -1))
        motion_f_list = np.reshape(motion_f_list, (1, TIMESTEP_LEN, -1))
        motion_target_f = np.reshape(motion_target_f, (1, 1, -1))
        interaction_f_list = np.reshape(interaction_f_list, (1, TIMESTEP_LEN, -1))
        interaction_target_f = np.reshape(interaction_target_f, (1, 1, -1))

        return app_f_list, app_target_f, motion_f_list, motion_target_f, interaction_f_list, interaction_target_f

