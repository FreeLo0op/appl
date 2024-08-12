import os
import argparse
import torch
import torchaudio
from scipy.special import exp1
from models.ns.dpcrn import DPCRN_block_snri
from modules.stft import STFT, ISTFT

eps = 1e-8

class CkptInferenceSnri:
    def __init__(self, args):
        self.args = args
        self.real_time_mode = self.get_real_time_mode()
        print(f"self.real_time_mode of ns is {self.real_time_mode}")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # init model & load weights
        self.post_processing_type = "LSA"
        self.model = self.model_init(self.post_processing_type)

        # paras
        self.sample_rate = 16000 # Hz
        self.win_length = 32 # ms
        self.hop_length = 16 # ms

        # init STFT & ISTFT objects
        self.stft = STFT(
            sample_rate=self.sample_rate,
            win_length=self.win_length,
            hop_length=self.hop_length,
            n_fft=512,
            window_fn=torch.hann_window,
            )
        self.istft = ISTFT(
            sample_rate=self.sample_rate,
            win_length=self.win_length,
            hop_length=self.hop_length,
            window_fn=torch.hann_window,
            )
    
    def model_init(self, post_processing_type):
        #hparams
        input_channel = 1
        block_type = "DprnnBlock"
        Freq = 257
        number_dp = 1
        rnn_type = "GRU"
        bidirectional = False
        skip_type = "conv_add"
        
        if post_processing_type=='LSA':
            ckpt = "./tal_audio/pretrained_models/ns/ns_model_v1.ckpt"
        elif post_processing_type=='NNOmlsa':
            ckpt = "" # Need to be updated
        
        model = DPCRN_block_snri(
            input_channel,
            block_type,
            Freq,
            number_dp,
            rnn_type,
            bidirectional,
            skip_type,
            real_time_mode=self.real_time_mode,
        )
        
        model.load_state_dict(torch.load(ckpt, map_location=torch.device('cpu')))
        model.to(self.device)
        model.eval()
        print("***********: ", model)
        return model

    def get_real_time_mode(self):
        real_time_mode = self.args.realtime
        if real_time_mode == 'true':
            real_time_mode = True
        else:
            real_time_mode = False
        return real_time_mode

    def spectral_magnitude(self, stft, power=1, log=False, eps=1e-14):
        """Returns the magnitude of a complex spectrogram.

        Arguments
        ---------
        stft : torch.Tensor
            A tensor, output from the stft function.
        power : int
            What power to use in computing the magnitude.
            Use power=1 for the power spectrogram.
            Use power=0.5 for the magnitude spectrogram.
        log : bool
            Whether to apply log to the spectral features.

        Example
        -------
        >>> a = torch.Tensor([[3, 4]])
        >>> spectral_magnitude(a, power=0.5)
        tensor([5.])
        """
        spectr = stft.pow(2).sum(-1)

        # Add eps avoids NaN when spectr is zero
        if power < 1:
            spectr = spectr + eps
        spectr = spectr.pow(power)

        if log:
            return torch.log(spectr + eps)
        return spectr

    def compute_amplitude(self, waveforms, lengths=None, amp_type="avg", scale="linear"):
        """Compute amplitude of a batch of waveforms.

        Arguments
        ---------
        waveform : tensor
            The waveforms used for computing amplitude.
            Shape should be `[time]` or `[batch, time]` or
            `[batch, time, channels]`.
        lengths : tensor
            The lengths of the waveforms excluding the padding.
            Shape should be a single dimension, `[batch]`.
        amp_type : str
            Whether to compute "avg" average or "peak" amplitude.
            Choose between ["avg", "peak"].
        scale : str
            Whether to compute amplitude in "dB" or "linear" scale.
            Choose between ["linear", "dB"].

        Returns
        -------
        The average amplitude of the waveforms.

        Example
        -------
        >>> signal = torch.sin(torch.arange(16000.0)).unsqueeze(0)
        >>> compute_amplitude(signal, signal.size(1))
        tensor([[0.6366]])
        """
        if len(waveforms.shape) == 1:
            waveforms = waveforms.unsqueeze(0)

        assert amp_type in ["avg", "peak"]
        assert scale in ["linear", "dB"]

        if amp_type == "avg":
            if lengths is None:
                out = torch.mean(torch.abs(waveforms), dim=1, keepdim=True)
            else:
                wav_sum = torch.sum(input=torch.abs(waveforms), dim=1, keepdim=True)
                out = wav_sum / lengths
        elif amp_type == "peak":
            out = torch.max(torch.abs(waveforms), dim=1, keepdim=True)[0]
        else:
            raise NotImplementedError

        if scale == "linear":
            return out
        elif scale == "dB":
            return torch.clamp(20 * torch.log10(out), min=-80)  # clamp zeros
        else:
            raise NotImplementedError

    def normalize(self, waveforms, lengths=None, amp_type="avg", eps=1e-14):
        """This function normalizes a signal to unitary average or peak amplitude.

        Arguments
        ---------
        waveforms : tensor
            The waveforms to normalize.
            Shape should be `[batch, time]` or `[batch, time, channels]`.
        lengths : tensor
            The lengths of the waveforms excluding the padding.
            Shape should be a single dimension, `[batch]`.
        amp_type : str
            Whether one wants to normalize with respect to "avg" or "peak"
            amplitude. Choose between ["avg", "peak"]. Note: for "avg" clipping
            is not prevented and can occur.
        eps : float
            A small number to add to the denominator to prevent NaN.

        Returns
        -------
        waveforms : tensor
            Normalized level waveform.
        """

        assert amp_type in ["avg", "peak"]

        batch_added = False
        if len(waveforms.shape) == 1:
            batch_added = True
            waveforms = waveforms.unsqueeze(0)

        den = self.compute_amplitude(waveforms, lengths, amp_type) + eps
        if batch_added:
            waveforms = waveforms.squeeze(0)
        return waveforms / den

    def clip_normalize_map_inverse(self, X_bar):
        X_db = X_bar*90 - 50
        X = self.db_inverse(X_db)
        return X

    def instant_priori_SNR(self, clean_spec, noise_spec):
        """Computes the instantaneous a priori SNR.

        Arguments:
        ---------
        clean_spec : torch.Tensor
            clean-speech short-time spectrum.
        noise_spec : torch.Tensor
            noise short-time spectrum.
        
        Returns:
        -------
        snri : torch.Tensor
            instantaneous a priori SNR.
        """
        device = clean_spec.device
        return torch.square(clean_spec)/torch.maximum(torch.square(noise_spec),torch.tensor([1e-12]).to(device=device))

    def speech_presence_pro(self, xi, gamma):
        q = torch.zeros_like(xi).to(device=xi.device)
        q += ((gamma<=1) & (xi<1.67))
        q += ((((gamma>1.0) & (gamma<=3.0)) & (xi<1.67))) * ((torch.tensor([3.0]).to(device=xi.device)-gamma)/torch.tensor([2.0]).to(device=xi.device))
        p = 1 / (1 + (q / (1 - q)) * (1 + xi) * (torch.exp( -gamma * xi / (1 + xi))) )
        p[torch.isnan(p)] = 0  
        return p

    def mmse_lsa_torch(self, xi, gamma):
        """Computes the MMSE-LSA gain function.
        
        Arguments:
        ---------
        xi : torch.Tensor
            a priori SNR
        gamma : torch.Tensor
            a posteriori SNR
        
        Returns:
        -------
        mmse_lsa_gain : torch.Tensor
            MMSE-LSA gain.
        """
        eps = torch.tensor([1e-12]).to(device=xi.device)
        xi = torch.maximum(xi, eps)
        gamma = torch.maximum(gamma, eps)
        coeff = torch.divide(xi, xi + 1)
        v = torch.multiply(coeff, gamma)

        mmse_lsa = torch.multiply(
            coeff, torch.exp(torch.multiply(torch.tensor([0.5]).to(device=xi.device), torch.tensor(exp1(v.detach().cpu().numpy())).to(device=xi.device)))
        )
        return mmse_lsa

    def mmse_omlsa_torch(self, xi, gamma, coef):
        """Computes the MMSE-omLSA gain function.
        
        Arguments
        ---------
        xi : torch.Tensor
            a priori SNR
        gamma : torch.Tensor
            a posteriori SNR
        
        Returns
        -------
        mmse_olsa_gain : torch.Tensor
            MMSE-OLSA gain.
        """
        eps = torch.tensor([1e-12]).to(device=xi.device)
        xi = torch.maximum(xi, eps)
        gamma = torch.maximum(gamma, eps)
        coeff = torch.divide(xi, xi + 1)
        v = torch.multiply(coeff, gamma)
        mmse_lsa = torch.multiply(
            coeff, torch.exp(torch.multiply(torch.tensor([0.5]).to(device=xi.device), torch.tensor(exp1(v.detach().cpu().numpy())).to(device=xi.device)))
        )
        p = self.speech_presence_pro(xi, gamma)
        gain = torch.pow(mmse_lsa,p) * torch.pow(coef,1-p) 
        return gain

    def get_feats(self, audio):
        feats= self.stft(audio)
        feats_mag = self.spectral_magnitude(feats, power=0.5)
        feats_phase = feats/(feats_mag.unsqueeze(dim=3) + eps)
        return feats_mag, feats_phase

    def ckpt_inference_real_time(self, input_frame, model):
        frame_out = model.forward(input_frame)
        priori_SNR = self.clip_normalize_map_inverse(frame_out).squeeze(dim=1)
        return priori_SNR

    def resynthesize(self, feats, audio_phase, priori_SNR, post_processing_type, coef, normalized=False):
        device = torch.device('cpu')
        audio_phase = audio_phase.to(device)
        priori_SNR = priori_SNR.to(device)
        feats = feats.to(device)
        
        predict_wav_list = []
        for phase, priori, feat in zip(audio_phase, priori_SNR, feats):
            priori = priori.unsqueeze(0)
            phase = phase.unsqueeze(0)
            feat = feat.unsqueeze(0)
            if post_processing_type=='LSA':
                mask = self.mmse_lsa_torch(priori, priori+1)
            elif post_processing_type=='NNOmlsa':
                mask = self.mmse_omlsa_torch(priori, priori+1, coef)
            
            assert mask.shape == feat.shape
            predict_spec = torch.mul(mask, feat)

            # Combine with enhanced magnitude
            complex_predictions = torch.mul(
                torch.unsqueeze(predict_spec, -1),
                phase,
            )
            predict_wav = self.istft(complex_predictions)

            # Normalize. Since we're using peak amplitudes, ignore lengths
            if normalized:
                predict_wav = self.normalize(predict_wav, amp_type="peak")
            
            predict_wav_list.append(predict_wav)

        return predict_wav_list

    def db(self, X):
        """converts power value to decibels(dB).

        Arguments
        ---------
        X : torch.Tensor
            power value.
                
        Returns
        -------
            : torch.Tensor
            decibels(dB).
        """
        X = torch.maximum(X, torch.tensor([1e-12], device=X.device))
        return 10.0 * torch.log(X)/torch.log(torch.tensor([10.0], device=X.device))
    
    def db_inverse(self, X_db):
        return torch.exp((X_db/10.0)*torch.log(torch.tensor([10.0], device=X_db.device)))

    def ckpt_main(self, audio, post_processing_type, coef=0.01):
        """Computes the denoised wav.
        
        Arguments
        ---------
        audio : tensor
            noisy wav which you want to denoise.
        post_processing_type: str
            'LSA' is the most commonly used. option:('LSA','NNOmlsa').
        coef: float
            use to adject the noise reduction depth when post_processing_type is 'NNOmlsa'.

        Returns
        -------
        denoised audio and frame-based snr.
        """
        mag, phase = self.get_feats(audio) # [B, T, F]

        if self.real_time_mode:
            frames_out_snr_all = torch.tensor([]).to(self.device)

            for feat in mag:
                feat = feat.unsqueeze(0).to(self.device)
                
                frames_out_snr = torch.tensor([]).to(self.device)
                for i in range(mag.shape[1]):
                    frame_out_snr = self.ckpt_inference_real_time(feat[0:1,i:i+1,:], model=self.model)   #单帧信噪比(非dB)
                    frames_out_snr = torch.cat((frames_out_snr,  frame_out_snr), dim=1)
        
                    if frames_out_snr.shape[1] == feat.shape[1]:
                        break
                frames_out_snr_all = torch.cat((frames_out_snr_all,  frames_out_snr), dim=0)
            
            frames_out_snr = frames_out_snr_all
        else:
            # print(f"mag shape is {mag.shape}")
            frames_out_snr = self.ckpt_inference_real_time(mag, model=self.model)
            # print(f"frames_out_snr shape is {frames_out_snr.shape}")
        
        wav_frame = self.resynthesize(mag, phase, frames_out_snr, post_processing_type, coef)
        wav_frame = torch.cat(wav_frame, dim=0)
        
        return wav_frame, frames_out_snr

    def main(self, batch_audio, batch_len):
        batch_audio = batch_audio.to(self.device)

        audio_frame, snr_frame = self.ckpt_main(
            audio=batch_audio, 
            post_processing_type=self.post_processing_type, 
            coef=0.01)
        
        # Get the mean and convert abs value to dB
        mean_snrs = []
        frame_shift = self.hop_length*self.sample_rate//1000
        for idx, frames_out in enumerate(snr_frame):
            mean_snr = self.db(frames_out[:batch_len[idx]//frame_shift].mean())
            mean_snrs.append(mean_snr)

        return audio_frame, mean_snrs


if __name__ == "__main__":
    pass