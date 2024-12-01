import numpy as np
import librosa
import tensorflow as tf
from utils.math_utils import tf_float32
from hmmlearn import hmm

def load_audio(file_path, sample_rate, mono=True, window_size=0,
               from_numpy=False):
  """
  Load audio file from disk.
  :param file_path: Audio file path.
  :param sample_rate: Sample rate to load, will resample to the sample_rate
  if file has a different sample rate.
  :param mono: Whether to load a mono audio file.
  :param window_size: If larger than 0, then the waveform will be cut off to
  have a length that is multiple of window_size,
  :param from_numpy: If the file_path provided is pointing to a npy file.
  :return:
  """
  if from_numpy:
    y = np.load(file_path)
  else:
    y, _ = librosa.load(file_path, sample_rate, mono=mono, dtype=np.float64)
  if window_size > 0:
    output_length = len(y) // window_size * window_size
    y = y[:output_length]
  return y

def predict_voicing(confidence):
  # https://github.com/marl/crepe/pull/26
  """
  Find the Viterbi path for voiced versus unvoiced frames.
  Parameters
  ----------
  confidence : np.ndarray [shape=(N,)]
      voicing confidence array, i.e. the confidence in the presence of
      a pitch
  Returns
  -------
  voicing_states : np.ndarray [shape=(N,)]
      HMM predictions for each frames state, 0 if unvoiced, 1 if
      voiced
  """
  # uniform prior on the voicing confidence
  starting = np.array([0.5, 0.5])

  # transition probabilities inducing continuous voicing state
  transition = np.array([[0.99, 0.01], [0.01, 0.99]])

  # mean and variance for unvoiced and voiced states
  means = np.array([[0.0], [1.0]])
  variances = np.array([[0.25], [0.25]])

  # fix the model parameters because we are not optimizing the model
  model = hmm.GaussianHMM(n_components=2)
  model.startprob_, model.covars_, model.transmat_, model.means_, \
  model.n_features = starting, variances, transition, means, 1

  # find the Viterbi path
  voicing_states = model.predict(confidence.reshape(-1, 1), [len(confidence)])

  return np.array(voicing_states)


def get_framed_lengths(input_length, frame_size, hop_size, padding='center'):
  """Give a strided framing, such as tf.signal.frame, gives output lengths.

  Args:
    input_length: Original length along the dimension to be framed.
    frame_size: Size of frames for striding.
    hop_size: Striding, space between frames.
    padding: Type of padding to apply, ['valid', 'same', 'center']. 'valid' is
      a no-op. 'same' applies padding to the end such that
      n_frames = n_t / hop_size. 'center' applies padding to both ends such that
      each frame timestamp is centered and n_frames = n_t / hop_size + 1.

  Returns:
    n_frames: Number of frames left after striding.
    padded_length: Length of the padded signal before striding.
  """
  # Use numpy since this function isn't used dynamically.
  def get_n_frames(length):
    return int(np.floor((length - frame_size) // hop_size)) + 1

  if padding == 'valid':
    padded_length = input_length
    n_frames = get_n_frames(input_length)

  elif padding == 'center':
    padded_length = input_length + frame_size
    n_frames = get_n_frames(padded_length)

  elif padding == 'same':
    n_frames = int(np.ceil(input_length / hop_size))
    padded_length = (n_frames - 1) * hop_size + frame_size

  return n_frames, padded_length

def pad(x, frame_size, hop_size, padding='center',
        axis=1, mode='CONSTANT', constant_values=0):
  """Pad a tensor for strided framing such as tf.signal.frame.

  Args:
    x: Tensor to pad, any shape.
    frame_size: Size of frames for striding.
    hop_size: Striding, space between frames.
    padding: Type of padding to apply, ['valid', 'same', 'center']. 'valid' is
      a no-op. 'same' applies padding to the end such that
      n_frames = n_t / hop_size. 'center' applies padding to both ends such that
      each frame timestamp is centered and n_frames = n_t / hop_size + 1.
    axis: Axis along which to pad `x`.
    mode: Padding mode for tf.pad(). One of "CONSTANT", "REFLECT", or
      "SYMMETRIC" (case-insensitive).
    constant_values: Passthrough kwarg for tf.pad().

  Returns:
    A padded version of `x` along axis. Output sizes can be computed separately
      with strided_lengths.
  """
  x = tf_float32(x)

  if padding == 'valid':
    return x

  if hop_size > frame_size:
    raise ValueError(f'During padding, frame_size ({frame_size})'
                     f' must be greater than hop_size ({hop_size}).')

  if len(x.shape) <= 1:
    axis = 0

  n_t = x.shape[axis]
  _, n_t_padded = get_framed_lengths(n_t, frame_size, hop_size, padding)
  pads = [[0, 0] for _ in range(len(x.shape))]

  if padding == 'same':
    pad_amount = int(n_t_padded - n_t)
    pads[axis] = [0, pad_amount]

  elif padding == 'center':
    pad_amount = int(frame_size // 2)  # Symmetric even padding like librosa.
    pads[axis] = [pad_amount, pad_amount]

  else:
    raise ValueError('`padding` must be one of [\'center\', \'same\''
                     f'\'valid\'], received ({padding}).')

  return tf.pad(x, pads, mode=mode, constant_values=constant_values)

