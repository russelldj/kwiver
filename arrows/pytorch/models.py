import torch
from torchvision import models
from torch import nn


def get_config():
    class Config():
        # lstm settings
        H = 128
        K = 100
        A_F_num = 500  # appearance CNN output #Dim of feature
        I_F_num = 49
        M_F_num = 2
    return Config()

g_config = get_config()


# Siamese network
# ==================================================================
class Siamese(nn.Module):
    def __init__(self):
        super(Siamese, self).__init__()
        self.resnet = models.resnet50(pretrained=True)
        self.num_fcin = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(self.num_fcin, 500)
        self.pdist = nn.PairwiseDistance(1)

    def forward(self, input1, input2):
        output1 = self.resnet(input1)
        output2 = self.resnet(input2)
        output = self.pdist(output1, output2)

        return output1, output2, output


# Appearance LSTM
# ==================================================================
class AppearanceLSTM(nn.Module):
    def __init__(self):
        super(AppearanceLSTM, self).__init__()

        self.target_fc = nn.Linear(g_config.A_F_num, g_config.H)
        self.lstm = nn.LSTM(
            input_size=g_config.A_F_num,
            hidden_size=g_config.H,
            num_layers=1,
            batch_first=True
        )
        self.fc1 = nn.Linear(g_config.H * 2, g_config.K)
        self.fc2 = nn.Linear(g_config.K, 2)
        self.relu = nn.ReLU()

    # FIXME: we may need to add hidden status from previous
    def forward(self, track_input, target_input):

        target_out = self.target_fc(target_input)
        r_out, (h_t, c_t) = self.lstm(track_input, None)

        outs = []
        relu_outs = []
        for i in range(g_config.timeStep):
            h_t = r_out[:, i, :]
            combined_out = torch.cat((h_t, target_out), 1)
            fc1_output = self.fc1(combined_out)
            relu_output = self.relu(fc1_output)
            relu_outs.append(relu_output)
            outs.append(self.fc2(relu_output))

        return torch.stack(outs, dim=1), torch.stack(relu_outs, dim=1)


# Interaction LSTM
# ==================================================================
class InteractionLSTM(nn.Module):
    def __init__(self):
        super(InteractionLSTM, self).__init__()

        self.target_fc = nn.Linear(g_config.I_F_num, g_config.H)
        self.lstm = nn.LSTM(
            input_size=g_config.I_F_num,
            hidden_size=g_config.H,
            num_layers=1,
            batch_first=True
        )
        self.fc1 = nn.Linear(g_config.H * 2, g_config.K)
        self.fc2 = nn.Linear(g_config.K, 2)
        self.relu = nn.ReLU()

    def forward(self, track_input, target_input):

        target_out = self.target_fc(target_input)
        r_out, (h_t, c_t) = self.lstm(track_input, None)

        outs = []
        relu_outs = []
        for i in range(g_config.timeStep):
            h_t = r_out[:, i, :]
            combined_out = torch.cat((h_t, target_out), 1)
            fc1_output = self.fc1(combined_out)
            relu_output = self.relu(fc1_output)
            relu_outs.append(relu_output)
            outs.append(self.fc2(relu_output))

        return torch.stack(outs, dim=1), torch.stack(relu_outs, dim=1)


# Motion LSTM
# ==================================================================
class MotionLSTM(nn.Module):
    def __init__(self):
        super(MotionLSTM, self).__init__()

        self.target_fc = nn.Linear(g_config.M_F_num, g_config.H)
        self.lstm = nn.LSTM(
            input_size=g_config.M_F_num,
            hidden_size=g_config.H,
            num_layers=1,
            batch_first=True
        )
        self.fc1 = nn.Linear(g_config.H * 2, g_config.K)
        self.fc2 = nn.Linear(g_config.K, 2)
        self.relu = nn.ReLU()

    def forward(self, track_input, target_input):

        target_out = self.target_fc(target_input)
        r_out, (h_t, c_t) = self.lstm(track_input, None)

        outs = []
        relu_outs = []
        for i in range(g_config.timeStep):
            h_t = r_out[:, i, :]
            combined_out = torch.cat((h_t, target_out), 1)
            fc1_output = self.fc1(combined_out)
            relu_output = self.relu(fc1_output)

            relu_outs.append(relu_output)
            outs.append(self.fc2(relu_output))

        return torch.stack(outs, dim=1), torch.stack(relu_outs, dim=1)


# Target LSTM
# ==================================================================
class TargetLSTM(nn.Module):
    def __init__(self, app_model='', motion_model='', interaction_model='', train_flag=False):
        super(TargetLSTM, self).__init__()

        self.appearance = AppearanceLSTM()
        if app_model is not '':
            snapshot = torch.load(app_model)
            self.appearance.load_state_dict(snapshot['state_dict'])
            self.appearance.train(train_flag)

        self.motion = MotionLSTM().cuda()
        if motion_model is not '':
            snapshot = torch.load(motion_model)
            self.motion.load_state_dict(snapshot['state_dict'])
            self.motion.train(train_flag)

        self.interaction = InteractionLSTM().cuda()
        if interaction_model is not '':
            snapshot = torch.load(interaction_model)
            self.interaction.load_state_dict(snapshot['state_dict'])
            self.interaction.train(train_flag)

        self.lstm = nn.LSTM(
            input_size=g_config.K * 3,
            hidden_size=g_config.H,
            num_layers=1,
            batch_first=True
        )

        self.fc1 = nn.Linear(g_config.H, 2)

    def forward(self, appearance_input, appearance_target, motion_input, motion_target, interaction_input,
                interaction_target):
        r"""
        :param appearance_input:    appearance features         (batch, time_step, input_size)
        :param appearance_target:   appearance target feature   (batch, 1, input_size)
        :param motion_input:        motion features             (batch, time_step, input_size)
        :param motion_target:       motion target features      (batch, 1, input_size)
        :param interaction_input:   interaction features        (batch, time_step, input_size)
        :param interaction_target:  interaction target feature  (batch, 1, input_size)
        :return:
        """

        _, app_out = self.appearance(appearance_input, appearance_target)
        _, motion_out = self.motion(motion_input, motion_target)
        _, interaction_out = self.interaction(interaction_input, interaction_target)

        combined_input = torch.cat((app_out, motion_out, interaction_out), 2)
        r_out, (h_t, c_t) = self.lstm(combined_input, None)

        out = self.fc1(r_out[:, -1, :])
        return out

