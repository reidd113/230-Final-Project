import librosa
import tensorflow as tf
import numpy as np
from utils.math_utils import safe_log
from utils.audio_utils import tf_float32, pad

def spectral_centroid(wav, hop_length, sr):
  centroid = librosa.feature.spectral_centroid(y=wav, sr=sr,
                                               hop_length=hop_length)
  return centroid


def tf_stft(audio, win_length, hop_length, n_fft, pad_end=True):
  s = tf.signal.stft(
    signals=audio,
    frame_length=win_length,
    frame_step=hop_length,
    fft_length=n_fft,
    pad_end=pad_end)
  mag = tf.abs(s)
  return tf.cast(mag, tf.float32)


def tf_mel(audio, sample_rate, win_length, hop_length, n_fft, num_mels, fmin=40,
           pad_end=True):
  """Calculate Mel Spectrogram."""
  mag = tf_stft(audio, win_length, hop_length, n_fft, pad_end=pad_end)
  num_spectrogram_bins = int(mag.shape[-1])
  hi_hz = sample_rate // 2
  linear_to_mel_matrix = tf.signal.linear_to_mel_weight_matrix(
    num_mels,
    num_spectrogram_bins,
    sample_rate,
    fmin,
    hi_hz)
  mel = tf.tensordot(mag, linear_to_mel_matrix, 1)
  mel.set_shape(mag.shape[:-1].concatenate(linear_to_mel_matrix.shape[-1:]))
  return mel


def tf_log_mel(audio, sample_rate, win_length, hop_length, n_fft, num_mels,
               fmin=40, pad_end=True):
  mel = tf_mel(audio, sample_rate, win_length, hop_length, n_fft, num_mels,
               fmin=fmin, pad_end=pad_end)
  return safe_log(mel)


def stft(audio, frame_size=2048, overlap=0.75, pad_end=True):
  """Differentiable stft in tensorflow, computed in batch."""
  # Remove channel dim if present.
  audio = tf_float32(audio)
  if len(audio.shape) == 3:
    audio = tf.squeeze(audio, axis=-1)

  s = tf.signal.stft(
      signals=audio,
      frame_length=int(frame_size),
      frame_step=int(frame_size * (1.0 - overlap)),
      fft_length=None,  # Use enclosing power of 2.
      pad_end=pad_end)
  return s


def stft_np(audio, frame_size=2048, overlap=0.75, pad_end=True):
  """Non-differentiable stft using librosa, one example at a time."""
  assert frame_size * overlap % 2.0 == 0.0
  hop_size = int(frame_size * (1.0 - overlap))
  is_2d = (len(audio.shape) == 2)

  if pad_end:
    audio = pad(audio, frame_size, hop_size, 'same', axis=is_2d).numpy()

  def stft_fn(y):
    return librosa.stft(
        y=y, n_fft=int(frame_size), hop_length=hop_size, center=False).T

  s = np.stack([stft_fn(a) for a in audio]) if is_2d else stft_fn(audio)
  return s