import numpy as np
import librosa 
from utils.spectral_utils import tf_log_mel
from utils.feature_extraction import extract_f0, compute_loudness
from utils.multiband_decomp import compute_pqmf

def zero_pad_audio(audio, example_size=4, sr=16000):
  """
  Zero-pads the audio so that it is evenly sliced into examples of given length in seconds

  Args:
    audio: the raw audio in samples
    example_size: the length of each example in seconds (default 4)
    sr: sample rate of audio (default 16000)
  Returns:
    padded: the zero-padded audio
  """
  example_samps = example_size * sr
  pad_amount = example_samps - len(audio) % example_samps
  z = np.zeros(pad_amount)
  padded = np.concatenate([audio, z])

  return padded


def preprocess_custom(filename, example_size=4, sr=16000, frame_size=64):
  """
  Takes input audio, extracts features, and slices into examples.

  Args:
    audio: the raw audio in
    example_size: the length of each example in seconds (default 4)
    sr: sample rate of audio (default 16000)
    frame_size: the number of samples per processing frame (default 64)
  """

  # load whole audio file
  y, _ = librosa.load(filename, sr=16000)
  # zero-pad the raw audio
  y_padded = zero_pad_audio(y, example_size, sr)

  # processing parameters
  hop_length = frame_size
  win_length = hop_length * 2

  # extract features
  mel = tf_log_mel(y_padded, sample_rate=sr, win_length=win_length, hop_length=hop_length, n_fft=1024, num_mels=64)
  pqmf = compute_pqmf(y_padded, block_size=frame_size)

  # reshape
  #feature_length = sr * example_size / frame_size

  mel_split = np.reshape(mel, (-1, 1000, 64))
  pqmf_split = np.reshape(pqmf, (-1, 1000, 64))
  audio_split = np.reshape(y_padded, (-1, 1000 * frame_size))

  split_output = {'audio': audio_split.astype(np.float32), 'pqmf': pqmf_split.astype(np.float32), 'mel': mel_split.astype(np.float32)}

  return split_output

def pad_feature_array(feature_array):
  """
  Zero-pads the extracted feature arrays to be size 1000

  Args:
    feature_array: the extracted feature array
  Returns:
    padded: the z
    ero-padded feature array
  """
  if feature_array.ndim == 2:
    1000 - feature_array.shape[0] % 1000
    z = np.zeros((1000 - feature_array.shape[0] % 1000, feature_array.shape[1]))
    padded = np.concatenate([feature_array, z])
  else:
    1000 - len(feature_array) % 1000
    z = np.zeros(1000 - len(feature_array) % 1000)
    padded = np.concatenate([feature_array, z])

  return padded

def pad_audio(y, frame_size, padded_shape):
  """
    Zero-pads the audio based on the frame size

    Args:
      y: the audio as an array
      frame_size: number of frames per window
      padded_shape: the shape of the feature array
    Returns:
      padded: the zero-padded audio
  """
  target_size = padded_shape[0] * frame_size
  pad_amount = target_size - len(y)
  z = np.zeros(pad_amount)
  padded = np.concatenate([y, z])

  return padded


def preprocess_file(filename, frame_size=64, sr=16000):
  """
  Takes input audio, extracts features, and slices into examples.

  Args:
    filename: path to audio file (string)
    frame_size: the number of samples per processing frame (int)
    sr: sample rate of audio file (int)

  Returns:
    split_output: a dict containing f0 (shape=(m, 1000)), loudness (shape=(m, 1000)), and log-mel (shape=(m, 1000, 64))
    output: a dict containing the raw audio, as well as f0, loudness, and log-mel for the full audio length
  """

  # load whole audio file
  y, _ = librosa.load(filename, sr=16000)

  # processing parameters
  hop_length = frame_size
  win_length = hop_length * 2
  frame_shift_ms = 1000 / sr * frame_size
  frame_rate = sr / frame_size
  total_frames = len(y)

  # extract features
  f0_hz = extract_f0(y,frame_shift_ms=frame_shift_ms, sr=sr)
  loudness_db = compute_loudness(y, sample_rate=sr, frame_rate=frame_rate, n_fft=512)
  mel = tf_log_mel(y, sample_rate=sr, win_length=win_length, hop_length=hop_length, n_fft=1024, num_mels=64)

  # slice into examples, reshape

  f0_hz_padded = pad_feature_array(f0_hz)
  loudness_db_padded = pad_feature_array(loudness_db)
  mel_padded = pad_feature_array(mel)
  y_padded = pad_audio(y, frame_size, f0_hz_padded.shape)

  f0_hz_split = np.reshape(f0_hz_padded, (-1, 1000))
  loudness_db_split = np.reshape(loudness_db_padded, (-1, 1000))
  mel_split = np.reshape(mel_padded, (-1, 1000, 64))
  audio_split = np.reshape(y_padded, (-1, 1000 * frame_size))

  split_output = {'audio': audio_split.astype(np.float32), 'f0_hz': f0_hz_split.astype(np.float32), 'loudness_db': loudness_db_split.astype(np.float32), 'mel': mel_split.astype(np.float32)}
  output = {'audio': y, 'f0_hz': f0_hz, 'loudness_db': loudness_db, 'mel': mel}

  return split_output, output

def create_dataset_ddsp(preprocess_output):
  """Creates a TF Dataset for DDSP training.

  Args:
    preprocess_output: A dictionary containing preprocessed audio features.

  Returns:
    A tf.data.Dataset object.
  """
  dataset = tf.data.Dataset.from_tensor_slices((
      {
          'f0_hz': preprocess_output['f0_hz'],
          'loudness_db': preprocess_output['loudness_db'],
          'mel': preprocess_output['mel']
      },
      preprocess_output['audio']  # Target (audio)
  ))
  dataset = dataset.batch(4)
  return dataset

def main():
  print("test")
    # @todo implement from python notebook


if __name__ == '__main__':
  main()