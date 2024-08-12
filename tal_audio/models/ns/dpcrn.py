import torch
import torch.nn.functional as F

class DprnnBlock(torch.nn.Module):
    def __init__(
        self, rnn_type, bidirectional, num_units, Freq, C, causal=True,
        real_time_mode=False
    ):
        super(DprnnBlock, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_units = num_units
        self.Freq = Freq
        self.C = C
        self.causal = causal
        self.real_time_mode = real_time_mode
        if rnn_type == "LSTM":
            self.intra_rnn = torch.nn.LSTM(
                input_size=C,
                hidden_size=self.num_units // 2
                if bidirectional
                else self.num_units,
                batch_first=True,
                bidirectional=bidirectional,
            )
        elif rnn_type == "GRU":
            self.intra_rnn = torch.nn.GRU(
                input_size=C,
                hidden_size=self.num_units // 2
                if bidirectional
                else self.num_units,
                batch_first=True,
                bidirectional=bidirectional,
            )
        self.intra_fc = torch.nn.Linear(
            in_features=C, out_features=self.num_units
        )
        if self.causal:
            self.intra_ln = torch.nn.LayerNorm([Freq, C])
        if rnn_type == "LSTM":
            self.inter_rnn = torch.nn.LSTM(
                input_size=C, hidden_size=self.num_units, batch_first=True
            )
        elif rnn_type == "GRU":
            self.inter_rnn = torch.nn.GRU(
                input_size=C, hidden_size=self.num_units, batch_first=True
            )
        self.intra_rnn_state = torch.zeros((1, 1, self.num_units))
        self.inter_rnn_state = torch.zeros((1, 1*Freq, self.num_units))    #实时运行时batch size默认为1  
        self.inter_fc = torch.nn.Linear(
            in_features=C, out_features=self.num_units
        )
        if self.causal:
            self.inter_ln = torch.nn.LayerNorm([Freq, C])

    def forward(self, x):
        # Intra-Chunk
        B = x.shape[0]
        T = x.shape[1]
        F = x.shape[2]
        C = x.shape[3]
        intra_lstm_input = x.reshape(B * T, F, C)
        if self.real_time_mode:
            intra_lstm_out = None
            self.intra_rnn_state = torch.zeros((1, 1, self.num_units)).to(self.device)
            for i in range(F):
                """
                流式处理时,B*T=1,因此输入的hidden state可以不按照batch_first=True处理。
                """
                intra_lstm_input_frame = intra_lstm_input[:, i: i + 1, :]
                intra_lstm_out_i, self.intra_rnn_state = self.intra_rnn(intra_lstm_input_frame, self.intra_rnn_state)

                if intra_lstm_out is None:
                    intra_lstm_out = intra_lstm_out_i
                else:
                    intra_lstm_out = torch.cat((intra_lstm_out, intra_lstm_out_i), dim=1)
        else:
            intra_lstm_out, _ = self.intra_rnn(intra_lstm_input)

        intra_dense_out = self.intra_fc(intra_lstm_out)
        if self.causal:
            intra_ln_input = intra_dense_out.reshape(B, T, F, C)
            intra_out = self.intra_ln(intra_ln_input)
        else:
            intra_ln_input = x.reshape(B, T * F * C)
            intra_ln_out = self.intra_ln(intra_ln_input)
            intra_out = intra_ln_out.reshape(B, T, F, C)
        intra_out = x + intra_out
        # Inter-Chunk
        inter_lstm_input = intra_out.permute(0, 2, 1, 3).reshape(B * F, T, C)
        if self.real_time_mode:
            inter_lstm_out = None
            # TODO(YueSu): Modify 64 dynamic
            for i in range(B*F):
                inter_lstm_input_frame = inter_lstm_input[i: i + 1, :, :]
                inter_rnn_state = self.inter_rnn_state[:, i: i + 1, :].to(self.device)
                inter_lstm_out_i, inter_rnn_state_out = self.inter_rnn(inter_lstm_input_frame, inter_rnn_state)
                pre_inter_rnn_state = self.inter_rnn_state[:, :i, :].to(self.device)
                others_inter_rnn_state = self.inter_rnn_state[:, i + 1:, :].to(self.device)
                self.inter_rnn_state = torch.cat([pre_inter_rnn_state, inter_rnn_state_out, others_inter_rnn_state], dim=1)
                
                if inter_lstm_out is None:
                    inter_lstm_out = inter_lstm_out_i
                else:
                    inter_lstm_out = torch.cat((inter_lstm_out, inter_lstm_out_i), dim=0)
        else:
            inter_lstm_out, _ = self.inter_rnn(inter_lstm_input)
        inter_dense_out = self.inter_fc(inter_lstm_out).reshape(B, F, T, C)
        if self.causal:
            inter_ln_input = inter_dense_out.permute(0, 2, 1, 3)
            inter_out = self.inter_ln(inter_ln_input)
        else:
            inter_ln_input = inter_dense_out.reshape(B, F * T * C)
            inter_ln_out = self.inter_ln(inter_ln_input)
            inter_out = inter_ln_out.reshape(B, F, T, C).permute(0, 2, 1, 3)
        inter_out = intra_out + inter_out
        return inter_out
    
class TimeDelay(torch.nn.Module):
    def __init__(self, buffer_shape, dim=1, chunk_size=1):
        super(TimeDelay, self).__init__()
        self.dim = dim
        self.chunk_size = chunk_size
        self.buffer_shape = buffer_shape
        self.buffer = torch.zeros(buffer_shape)

    def forward(self, input):
        if self.dim == 1:
            historical_data = self.buffer[:, self.chunk_size:, ...]
        else:
            historical_data = self.buffer[:, :, self.chunk_size:, ...]
        self.buffer = torch.cat([historical_data, input], dim=self.dim)

        return self.buffer

class causal_encoder(torch.nn.Module):
    def __init__(
        self,
        input_channel,
        channel=None,
        kernel_size=None,
        stride=(1, 2),
        padding=None,
        is_training=None,
        activation_type="PReLU",
        real_time_mode=False,
        buffer_shape=(0,0,0,0),
    ):
        # channel: out_channels
        super(causal_encoder, self).__init__()
        self.channel = channel
        self.kernel_size = kernel_size
        self.stride = stride
        self.is_training = is_training
        self.real_time_mode = real_time_mode
        self.padding = list(padding)
        self.buffer_shape = buffer_shape
        
        if self.real_time_mode:
            # T 方向不补 0
            self.padding[-2] = 0
            
        self.causal_conv = torch.nn.Conv2d(
            in_channels=input_channel,
            out_channels=self.channel,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=0,
        )
        if activation_type == "PReLU":
            self.Prelu = torch.nn.PReLU()
        elif activation_type == "ReLU":
            self.Prelu = torch.nn.ReLU()
        self.batchnorm = torch.nn.BatchNorm2d(num_features=self.channel)
        self.timedelay = TimeDelay(buffer_shape=self.buffer_shape)

    def forward(self, inputs):
        if self.real_time_mode:
            inputs = self.timedelay(inputs)

        inputs = torch.nn.functional.pad(
            inputs, tuple(self.padding), mode="constant"
        )  # 在第三维填充  N,T,F,C
        inputs = inputs.permute(0, 3, 1, 2)  # N,C,T,F
        conv_out = self.causal_conv(inputs)
        conv_out = self.batchnorm(conv_out)
        conv_out = self.Prelu(conv_out).permute(0, 2, 3, 1)  # N,T,F,C
        return conv_out

class decoder_conv_add_skip(torch.nn.Module):
    def __init__(
        self,
        en_channel,
        channel,
        stride=(1, 2),
        kernel_size=None,
        output_padding=(0, 0),
        name=None,
        is_training=None,
        conv_channel=None,
        activation_type="PReLU",
        real_time_mode=False,
        buffer_shape=None,
    ):
        super(decoder_conv_add_skip, self).__init__()
        self.channel = channel
        self.stride = stride
        self.kernel_size = kernel_size
        self.output_padding = output_padding
        self.name = name
        self.is_training = is_training
        self.conv_channel = conv_channel
        self.real_time_mode = real_time_mode
        self.buffer_shape = buffer_shape
        self.conv1 = torch.nn.Conv2d(
            in_channels=en_channel,
            out_channels=conv_channel,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=0,
        )
        conv_trans_padding = (list(self.kernel_size)[0] - 1, 0) if self.real_time_mode else (0, 0)
        self.causal_deconv = torch.nn.ConvTranspose2d(
            in_channels=conv_channel,
            out_channels=channel,
            kernel_size=kernel_size,
            stride=stride,
            output_padding=output_padding,
            padding=conv_trans_padding,
        )
        self.batchnorm = torch.nn.BatchNorm2d(num_features=self.channel)
        if activation_type == "PReLU":
            self.Prelu = torch.nn.PReLU()
        elif activation_type == "ReLU":
            self.Prelu = torch.nn.ReLU()

        if self.real_time_mode:
            self.timedelay = TimeDelay(buffer_shape=self.buffer_shape)
        
    def forward(self, inputs_1, inputs_2):
        # inputs_1:  encoder out;   inputs_2:  decoder out
        # inputs_1 inputs_2   N,T,F,C
        inputs_1 = inputs_1.permute(0, 3, 1, 2)

        inputs_1 = self.conv1(inputs_1).permute(0, 2, 3, 1)
        inputs = inputs_1 + inputs_2

        if self.real_time_mode:
            inputs = self.timedelay(inputs)

        inputs = inputs.permute(0, 3, 1, 2)  # N,T,F,C ->  N,C,T,F
        deconv = self.causal_deconv(inputs)
        
        if self.name == "deconv1":
            if not self.real_time_mode:
                deconv = deconv[:, :, :-1, :]
            deconv = torch.sigmoid(deconv)
        elif self.name in ["deconv5", "deconv4", "deconv3"]:
            if not self.real_time_mode:
                deconv = deconv[:, :, :-1, :]
            deconv = deconv[:, :, :, 1:-1]
            deconv = self.batchnorm(deconv)
            deconv = self.Prelu(deconv)
        else:
            if not self.real_time_mode:
                deconv = deconv[:, :, :-1, :]
            deconv = deconv[:, :, :, :-1]
            deconv = self.batchnorm(deconv)
            deconv = self.Prelu(deconv)

        deconv = deconv.permute(0, 2, 3, 1)  # N,C,T,F->N,T,F,C
        return deconv

class DPCRN_block_snri(torch.nn.Module):
    def __init__(
        self,
        input_channel,
        block_type,
        Freq,
        number_dp,
        rnn_type,
        bidirectional,
        skip_type,
        real_time_mode=False,
    ):
        super(DPCRN_block_snri, self).__init__()
        self.real_time_mode = real_time_mode
        self.layerNorm = torch.nn.LayerNorm(
            normalized_shape=[Freq, input_channel]
        )
        self.conv1 = causal_encoder(
            input_channel,
            channel=8,
            kernel_size=(2, 5),
            stride=(1, 2),
            padding=(0, 0, 0, 2, 1, 0),
            real_time_mode=self.real_time_mode,
            buffer_shape=(1, 2, 257, 1),
        )
        self.conv2 = causal_encoder(
            8,
            channel=16,
            kernel_size=(2, 3),
            stride=(1, 2),
            padding=(0, 0, 0, 1, 1, 0),
            real_time_mode=self.real_time_mode,
            buffer_shape=(1, 2, 128, 8),
        )
        self.conv3 = causal_encoder(
            16,
            channel=32,
            kernel_size=(2, 3),
            stride=(1, 1),
            padding=(0, 0, 1, 1, 1, 0),
            real_time_mode=self.real_time_mode,
            buffer_shape=(1, 2, 64, 16),
        )
        self.conv4 = causal_encoder(
            32,
            channel=32,
            kernel_size=(2, 3),
            stride=(1, 1),
            padding=(0, 0, 1, 1, 1, 0),
            real_time_mode=self.real_time_mode,
            buffer_shape=(1, 2, 64, 32),
        )
        self.conv5 = causal_encoder(
            32,
            channel=96,
            kernel_size=(2, 3),
            stride=(1, 1),
            padding=(0, 0, 1, 1, 1, 0),
            real_time_mode=self.real_time_mode,
            buffer_shape=(1, 2, 64, 32),
        )
        self.num_units = 96
        self.number_dp = number_dp

        if block_type == "DprnnBlock":
            self.bottleneck = DprnnBlock(
                rnn_type=rnn_type,
                bidirectional=bidirectional,
                num_units=self.num_units,
                Freq=64,
                C=96,
                causal=True,
                real_time_mode=self.real_time_mode
            )

        if skip_type == "concatenate":
            pass
        elif skip_type == "conv_add":
            self.deconv5_out = decoder_conv_add_skip(
                en_channel=96,
                channel=32,
                kernel_size=(2, 3),
                stride=(1, 1),
                name="deconv5",
                conv_channel=96,
                real_time_mode=self.real_time_mode,
                buffer_shape=(1, 2, 64, 96),
            )
            self.deconv4_out = decoder_conv_add_skip(
                en_channel=32,
                channel=32,
                kernel_size=(2, 3),
                stride=(1, 1),
                name="deconv4",
                conv_channel=32,
                real_time_mode=self.real_time_mode,
                buffer_shape=(1, 2, 64, 32),
            )
            self.deconv3_out = decoder_conv_add_skip(
                en_channel=32,
                channel=16,
                kernel_size=(2, 3),
                stride=(1, 1),
                name="deconv3",
                conv_channel=32,
                real_time_mode=self.real_time_mode,
                buffer_shape=(1, 2, 64, 32),
            )
            self.deconv2_out = decoder_conv_add_skip(
                en_channel=16,
                channel=8,
                kernel_size=(2, 3),
                stride=(1, 2),
                output_padding=(0, 1),
                name="deconv2",
                conv_channel=16,
                real_time_mode=self.real_time_mode,
                buffer_shape=(1, 2, 64, 16),
            )
            self.deconv1_out = decoder_conv_add_skip(
                en_channel=8,
                channel=1,
                kernel_size=(2, 3),
                stride=(1, 2),
                output_padding=(0, 1),
                name="deconv1",
                conv_channel=8,
                real_time_mode=self.real_time_mode,
                buffer_shape=(1, 2, 128, 8),
            )
        else:
            print("please enter right skip_type!")

    def forward(self, x, export=False, online_frequency=False):
        
        with torch.set_grad_enabled(False):
            x = x.unsqueeze(3)
            
            input = self.layerNorm(x)
            conv1 = self.conv1(input)
            conv2 = self.conv2(conv1)
            conv3 = self.conv3(conv2)
            conv4 = self.conv4(conv3)
            conv5 = self.conv5(conv4)
            for i in range(self.number_dp):
                dp_out = self.bottleneck(conv5)
            deconv_5 = self.deconv5_out(conv5, dp_out)
            deconv_4 = self.deconv4_out(conv4, deconv_5)
            deconv_3 = self.deconv3_out(conv3, deconv_4)
            deconv_2 = self.deconv2_out(conv2, deconv_3)
            deconv_2 = deconv_2[:, :, :-1, :]
            deconv_1 = self.deconv1_out(conv1, deconv_2)
            deconv_1 = deconv_1.permute(0, 3, 1, 2)[:, :, :, :-1]
        return deconv_1
 