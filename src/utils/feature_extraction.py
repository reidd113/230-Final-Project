import librosa
import tensorflow as tf
import numpy as np
import crepe
from utils.audio_utils import pad, predict_voicing
from utils.spectral_utils import stft, stft_np
from utils.math_utils import power_to_db

F0_RANGE = 127.0
DB_RANGE = 80.0

def extract_f0(wav, frame_shift_ms=5, sr=44100, unvoice=True, no_log=False):
  """Extract f0 from audio using CREPE."""
  if sr != 16000:
    raise RuntimeError('CREPE method should use sr=16khz')
  _, frequency, confidence, _ = crepe.predict(
    wav, sr=sr,
    viterbi=True,
    step_size=frame_shift_ms,
    verbose=0 if no_log else 1)
  f0 = frequency
  if unvoice:
    is_voiced = predict_voicing(confidence)
    frequency_unvoiced = frequency * is_voiced
    f0 = frequency_unvoiced

  return f0

def compute_loudness(audio,
                     sample_rate=16000,
                     frame_rate=250,
                     n_fft=512,
                     range_db=DB_RANGE,
                     ref_db=0.0,
                     use_tf=True,
                     padding='center'):
  """Perceptual loudness (weighted power) in dB.

  Function is differentiable if use_tf=True.
  Args:
    audio: Numpy ndarray or tensor. Shape [batch_size, audio_length] or
      [audio_length,].
    sample_rate: Audio sample rate in Hz.
    frame_rate: Rate of loudness frames in Hz.
    n_fft: Fft window size.
    range_db: Sets the dynamic range of loudness in decibles. The minimum
      loudness (per a frequency bin) corresponds to -range_db.
    ref_db: Sets the reference maximum perceptual loudness as given by
      (A_weighting + 10 * log10(abs(stft(audio))**2.0). The old (<v2.0.0)
      default value corresponded to white noise with amplitude=1.0 and
      n_fft=2048. With v2.0.0 it was set to 0.0 to be more consistent with power
      calculations that have a natural scale for 0 dB being amplitude=1.0.
    use_tf: Make function differentiable by using tensorflow.
    padding: 'same', 'valid', or 'center'.

  Returns:
    Loudness in decibels. Shape [batch_size, n_frames] or [n_frames,].
  """
  # Pick tensorflow or numpy.
  lib = tf if use_tf else np
  reduce_mean = tf.reduce_mean if use_tf else np.mean
  stft_fn = stft if use_tf else stft_np

  # Make inputs tensors for tensorflow.
  frame_size = n_fft
  hop_size = sample_rate // frame_rate
  audio = pad(audio, frame_size, hop_size, padding=padding)
  audio = audio if use_tf else np.array(audio)

  # Temporarily a batch dimension for single examples.
  is_1d = (len(audio.shape) == 1)
  audio = audio[lib.newaxis, :] if is_1d else audio

  # Take STFT.
  overlap = 1 - hop_size / frame_size
  s = stft_fn(audio, frame_size=frame_size, overlap=overlap, pad_end=False)

  # Compute power.
  amplitude = lib.abs(s)
  power = amplitude**2

  # Perceptual weighting.
  frequencies = librosa.fft_frequencies(sr=sample_rate, n_fft=n_fft)
  a_weighting = librosa.A_weighting(frequencies)[lib.newaxis, lib.newaxis, :]

  # Perform weighting in linear scale, a_weighting given in decibels.
  weighting = 10**(a_weighting/10)
  power = power * weighting

  # Average over frequencies (weighted power per a bin).
  avg_power = reduce_mean(power, axis=-1)
  #loudness = librosa.power_to_db(avg_power, ref=ref_db, top_db = range_db)
  loudness = power_to_db(avg_power, ref_db=ref_db, range_db=range_db, use_tf=use_tf)

  # Remove temporary batch dimension.
  loudness = loudness[0] if is_1d else loudness

  return loudness

