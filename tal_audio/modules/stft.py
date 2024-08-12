import torch
from packaging import version


class STFT(torch.nn.Module):
    """computes the Short-Term Fourier Transform (STFT).

    This class computes the Short-Term Fourier Transform of an audio signal.
    It supports multi-channel audio inputs (batch, time, channels).

    Arguments
    ---------
    sample_rate : int
        Sample rate of the input audio signal (e.g 16000).
    win_length : float
        Length (in ms) of the sliding window used to compute the STFT.
    hop_length : float
        Length (in ms) of the hope of the sliding window used to compute
        the STFT.
    n_fft : int
        Number of fft point of the STFT. It defines the frequency resolution
        (n_fft should be <= than win_len).
    window_fn : function
        A function that takes an integer (number of samples) and outputs a
        tensor to be multiplied with each window before fft.
    normalized_stft : bool
        If True, the function returns the  normalized STFT results,
        i.e., multiplied by win_length^-0.5 (default is False).
    center : bool
        If True (default), the input will be padded on both sides so that the
        t-th frame is centered at time t×hop_length. Otherwise, the t-th frame
        begins at time t×hop_length.
    pad_mode : str
        It can be 'constant','reflect','replicate', 'circular', 'reflect'
        (default). 'constant' pads the input tensor boundaries with a
        constant value. 'reflect' pads the input tensor using the reflection
        of the input boundary. 'replicate' pads the input tensor using
        replication of the input boundary. 'circular' pads using  circular
        replication.
    onesided : True
        If True (default) only returns nfft/2 values. Note that the other
        samples are redundant due to the Fourier transform conjugate symmetry.

    Example
    -------
    >>> import torch
    >>> compute_STFT = STFT(
    ...     sample_rate=16000, win_length=25, hop_length=10, n_fft=400
    ... )
    >>> inputs = torch.randn([10, 16000])
    >>> features = compute_STFT(inputs)
    >>> features.shape
    torch.Size([10, 101, 201, 2])
    """

    def __init__(
        self,
        sample_rate,
        win_length=25,
        hop_length=10,
        n_fft=400,
        window_fn=torch.hamming_window,
        normalized_stft=False,
        center=True,
        pad_mode="constant",
        onesided=True,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.win_length = win_length
        self.hop_length = hop_length
        self.n_fft = n_fft
        self.normalized_stft = normalized_stft
        self.center = center
        self.pad_mode = pad_mode
        self.onesided = onesided

        # Convert win_length and hop_length from ms to samples
        self.win_length = int(
            round((self.sample_rate / 1000.0) * self.win_length)
        )
        self.hop_length = int(
            round((self.sample_rate / 1000.0) * self.hop_length)
        )

        self.window = window_fn(self.win_length)

    def forward(self, x):
        """Returns the STFT generated from the input waveforms.

        Arguments
        ---------
        x : tensor
            A batch of audio signals to transform.
        """

        # Managing multi-channel stft
        or_shape = x.shape
        if len(or_shape) == 3:
            x = x.transpose(1, 2)
            x = x.reshape(or_shape[0] * or_shape[2], or_shape[1])

        if version.parse(torch.__version__) <= version.parse("1.6.0"):
            stft = torch.stft(
                x,
                self.n_fft,
                self.hop_length,
                self.win_length,
                self.window.to(x.device),
                self.center,
                self.pad_mode,
                self.normalized_stft,
                self.onesided,
            )
        else:
            stft = torch.stft(
                x,
                self.n_fft,
                self.hop_length,
                self.win_length,
                self.window.to(x.device),
                self.center,
                self.pad_mode,
                self.normalized_stft,
                self.onesided,
                return_complex=False,
            )
        
        # Retrieving the original dimensionality (batch, time, channels)
        if len(or_shape) == 3:
            stft = stft.reshape(
                or_shape[0],
                or_shape[2],
                stft.shape[1],
                stft.shape[2],
                stft.shape[3],
            )
            stft = stft.permute(0, 3, 2, 4, 1)
        else:
            # (batch, time, channels)
            stft = stft.transpose(2, 1)

        return stft

class ISTFT(torch.nn.Module):
    def __init__(
        self,
        sample_rate,
        n_fft=None,
        win_length=25,
        hop_length=10,
        window_fn=torch.hann_window,
        normalized_stft=False,
        center=True,
        onesided=True,
        epsilon=1e-12,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.normalized_stft = normalized_stft
        self.center = center
        self.onesided = onesided
        self.epsilon = epsilon

        # Convert win_length and hop_length from ms to samples
        self.win_length = int(
            round((self.sample_rate / 1000.0) * self.win_length)
        )
        self.hop_length = int(
            round((self.sample_rate / 1000.0) * self.hop_length)
        )

        # Create window using provided function
        self.window = window_fn(self.win_length)

    def forward(self, x, sig_length=None):
    
        or_shape = x.shape

        # Infer n_fft if not provided
        if self.n_fft is None and self.onesided:
            n_fft = (x.shape[2] - 1) * 2
        elif self.n_fft is None and not self.onesided:
            n_fft = x.shape[2]
        else:
            n_fft = self.n_fft

        # Changing the format for (batch, time_step, n_fft, 2, n_channels)
        if len(or_shape) == 5:
            x = x.permute(0, 4, 2, 1, 3)

            # Lumping batch and channel dimension, because torch.istft
            # doesn't support batching.
            x = x.reshape(-1, x.shape[2], x.shape[3], x.shape[4])
        elif len(or_shape) == 4:
            x = x.permute(0, 2, 1, 3)

        # isft ask complex input
        x = torch.complex(x[..., 0], x[..., 1])

        istft = torch.istft(
            input=x,
            n_fft=n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window.to(x.device),
            center=self.center,
            onesided=self.onesided,
            length=sig_length,
        )

        # Convert back to (time, time_step, n_channels)
        if len(or_shape) == 5:
            istft = istft.reshape(or_shape[0], or_shape[4], -1)
            istft = istft.transpose(1, 2)

        return istft