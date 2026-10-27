from noisereduce.spectralgate.base import SpectralGate
import numpy as np
from scipy.signal import fftconvolve, stft, istft
from .utils import _amp_to_db


class SpectralGateStationary(SpectralGate):
    def __init__(
            self,
            y,
            sr,
            y_noise,
            n_std_thresh_stationary,
            chunk_size,
            clip_noise_stationary,
            padding,
            n_fft,
            win_length,
            hop_length,
            time_constant_s,
            freq_mask_smooth_hz,
            time_mask_smooth_ms,
            tmp_folder,
            prop_decrease,
            use_tqdm,
            n_jobs,
            stationary_soft_mask_slope_db=None,
    ):
        super().__init__(
            y=y,
            sr=sr,
            chunk_size=chunk_size,
            padding=padding,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            time_constant_s=time_constant_s,
            freq_mask_smooth_hz=freq_mask_smooth_hz,
            time_mask_smooth_ms=time_mask_smooth_ms,
            tmp_folder=tmp_folder,
            prop_decrease=prop_decrease,
            use_tqdm=use_tqdm,
            n_jobs=n_jobs,
        )

        self.n_std_thresh_stationary = n_std_thresh_stationary
        self.stationary_soft_mask_slope_db = stationary_soft_mask_slope_db

        if y_noise is None:
            self.y_noise = self.y

        else:
            y_noise = np.array(y_noise)
            # reshape data to (#channels, #frames)
            if len(y_noise.shape) == 1:
                self.y_noise = np.expand_dims(y_noise, 0)
            elif len(y.shape) > 2:
                raise ValueError("Waveform must be in shape (# frames, # channels)")
            else:
                self.y_noise = y_noise

        # collapse y_noise to one channel
        self.y_noise = np.mean(self.y_noise, axis=0)

        if clip_noise_stationary:
            self.y_noise = self.y_noise[:chunk_size]

        # calculate statistics over y_noise
        _, _, noise_stft = stft(
            self.y_noise,
            nfft=self._n_fft,
            noverlap=self._win_length - self._hop_length,
            nperseg=self._win_length,
            padded=False
        )

        noise_stft_db = _amp_to_db(noise_stft)
        self.mean_freq_noise = np.mean(noise_stft_db, axis=1)
        self.std_freq_noise = np.std(noise_stft_db, axis=1)

        self.noise_thresh = (
                self.mean_freq_noise + self.std_freq_noise * self.n_std_thresh_stationary
        )

    def spectral_gating_stationary(self, chunk):
        """non-stationary version of spectral gating"""
        denoised_channels = np.zeros(chunk.shape, chunk.dtype)
        for ci, channel in enumerate(chunk):
            _, _, sig_stft = stft(
                channel,
                nfft=self._n_fft,
                noverlap=self._win_length - self._hop_length,
                nperseg=self._win_length,
                padded=False
            )

            # spectrogram of signal in dB
            sig_stft_db = _amp_to_db(sig_stft)

            # calculate the threshold for each frequency/time bin
            db_thresh = np.repeat(
                np.reshape(self.noise_thresh, [1, len(self.mean_freq_noise)]),
                np.shape(sig_stft_db)[1],
                axis=0,
            ).T

            if self.stationary_soft_mask_slope_db is None:
                # mask if the signal is above the threshold (hard binary mask)
                sig_mask = sig_stft_db > db_thresh
            else:
                # soft sigmoid mask: a gradual transition around the threshold.
                # Hard 0/1 masks flip time-frequency bins on/off frame by frame,
                # which produces audible "musical noise"; a soft transition over
                # `stationary_soft_mask_slope_db` dB smooths that out and keeps
                # weak speech components near the threshold.
                sig_mask = 1.0 / (
                        1.0
                        + np.exp(
                                -(sig_stft_db - db_thresh)
                                / self.stationary_soft_mask_slope_db
                        )
                )

            sig_mask = sig_mask * self._prop_decrease + np.ones(np.shape(sig_mask)) * (
                    1.0 - self._prop_decrease
            )

            if self.smooth_mask:
                # convolve the mask with a smoothing filter
                sig_mask = fftconvolve(sig_mask, self._smoothing_filter, mode="same")

            # multiply signal with mask
            sig_stft_denoised = sig_stft * sig_mask

            # invert/recover the signal
            _, denoised_signal = istft(
                sig_stft_denoised,
                nfft=self._n_fft,
                noverlap=self._win_length - self._hop_length,
                nperseg=self._win_length
            )
            denoised_channels[ci, : len(denoised_signal)] = denoised_signal
        return denoised_channels

    def _do_filter(self, chunk):
        """Do the actual filtering"""
        chunk_filtered = self.spectral_gating_stationary(chunk)

        return chunk_filtered
