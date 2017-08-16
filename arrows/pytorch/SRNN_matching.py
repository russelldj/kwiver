import torch
from torch import nn
from torch.autograd import Variable

import numpy as np

from kwiver.arrows.pytorch.models import TargetLSTM, get_config, RnnType

TIMESTEP_LEN = 6
g_config = get_config()

class SRNN_matching(object):
    def __init__(self, targetRNN_full_model_path, targetRNN_AI_model_path):

        # load app, motion, interaction LSTM models
        model_list=(RnnType.Appearance, RnnType.Motion, RnnType.Interaction)
        self._targetRNN_full_model = TargetLSTM(model_list=model_list)
        #self._targetRNN_full_model = torch.nn.DataParallel(self._targetRNN_full_model).cuda()
        self._targetRNN_full_model = self._targetRNN_full_model.cuda()

        # load full target RNN model
        snapshot = torch.load(targetRNN_full_model_path)
        self._targetRNN_full_model.load_state_dict(snapshot['state_dict'])
        self._targetRNN_full_model.train(False)


        # load app, interaction LSTM models
        model_list=(RnnType.Appearance, RnnType.Interaction)
        self._targetRNN_AI_model = TargetLSTM(model_list=model_list).cuda()

        # load full target RNN model
        snapshot = torch.load(targetRNN_AI_model_path)
        self._targetRNN_AI_model.load_state_dict(snapshot['state_dict'])
        self._targetRNN_AI_model.train(False)
        #self._targetRNN_AI_model = torch.nn.DataParallel(self._targetRNN_AI_model).cuda()

    def __call__(self, track_set, track_state_list):
        tracks_num = len(track_set)
        track_states_num = len(track_state_list)

        similarity_mat = np.empty([tracks_num, track_states_num])

        track_idx_list = []

        for t in range(tracks_num):
            cur_track = track_set[t]
            track_idx_list.append(cur_track.id)
            for ts in range(track_states_num):

                if len(cur_track) < TIMESTEP_LEN:
                    # if the track does not have enough track_state, we will duplicate to time-step, but only use app and interaction features
                    temp_track = cur_track.duplicate_track_state(TIMESTEP_LEN)
                    similarity_mat[t, ts] = self._est_similarity(temp_track, track_state_list[ts], RnnType=RnnType.Target_RNN_AI)
                else:
                    # if the track does have enough track states, we use the original targetRNN
                    similarity_mat[t, ts] = self._est_similarity(cur_track, track_state_list[ts], RnnType=RnnType.Target_RNN_FULL)

        return similarity_mat, track_idx_list

    def _est_similarity(self, track, track_state, RnnType):
        assert(len(track) >= TIMESTEP_LEN)

        in_app_seq, in_app_target, in_motion_seq, in_motion_target, in_interaction_seq, in_interaction_target = \
            self._process_track(track, track_state)
        
        if RnnType == RnnType.Target_RNN_FULL:
            output = self._targetRNN_full_model(in_app_seq, in_app_target, in_motion_seq, in_motion_target, in_interaction_seq,
                       in_interaction_target)
        elif RnnType == RnnType.Target_RNN_AI:
            output = self._targetRNN_AI_model(in_app_seq, in_app_target, in_motion_seq, in_motion_target, in_interaction_seq,
                       in_interaction_target)

        F_softmax = nn.Softmax()
        output = F_softmax(output)
        pred = torch.max(output, 1)

        pred_lable = pred[1].data.cpu().numpy().squeeze()
        pred_p = pred[0].data.cpu().numpy().squeeze()

        # return negative value for huganrian algorithm
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
                app_f_list = np.append(app_f_list, ts.app_feature.reshape(1, g_config.A_F_num), axis=0)
                motion_f_list = np.append(motion_f_list, ts.motion_feature.reshape(1, g_config.M_F_num), axis=0)
                interaction_f_list = np.append(interaction_f_list, ts.interaction_feature.reshape(1, g_config.I_F_num), axis=0)

        app_target_f = track_state.app_feature.reshape(1, g_config.A_F_num)
        interaction_target_f = track_state.interaction_feature.reshape(1, g_config.I_F_num)
        motion_target_f = np.asarray(track_state.bbox_center).reshape(1, g_config.M_F_num) - \
                          np.asarray(track[-1].bbox_center).reshape(1, g_config.M_F_num)

        # add batch dim
        # TODO: loader may be simplier this part
        app_f_list = np.reshape(app_f_list, (1, TIMESTEP_LEN, -1))
        app_target_f = np.reshape(app_target_f, (1, -1))
        motion_f_list = np.reshape(motion_f_list, (1, TIMESTEP_LEN, -1))
        motion_target_f = np.reshape(motion_target_f, (1, -1))
        interaction_f_list = np.reshape(interaction_f_list, (1, TIMESTEP_LEN, -1))
        interaction_target_f = np.reshape(interaction_target_f, (1, -1))

        v_app_f_list, v_app_target_f = Variable(torch.from_numpy(app_f_list)).cuda(), Variable(torch.from_numpy(app_target_f)).cuda()
        v_motion_f_list, v_motion_target_f = Variable(torch.from_numpy(motion_f_list)).cuda(), Variable(torch.from_numpy(motion_target_f)).cuda()
        v_interaction_f_list, v_interaction_target_f = Variable(torch.from_numpy(interaction_f_list)).cuda(), Variable(torch.from_numpy(interaction_target_f)).cuda()

        return v_app_f_list, v_app_target_f, v_motion_f_list, v_motion_target_f, v_interaction_f_list, v_interaction_target_f

