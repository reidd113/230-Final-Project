import tensorflow as tf
from typing import TypeVar
import numpy as np

def safe_divide(numerator, denominator, eps=1e-7):
  """Avoid dividing by zero by adding a small epsilon."""
  safe_denominator = tf.where(denominator == 0.0, eps, denominator)
  return numerator / safe_denominator

def safe_log(x, eps=1e-5):
  """Avoid taking the log of a non-positive number."""
  safe_x = tf.where(x <= 0.0, eps, x)
  return tf.math.log(safe_x)

def logb(x, base=2.0, eps=1e-5):
  """Logarithm with base as an argument."""
  return safe_divide(safe_log(x, eps), safe_log(base, eps), eps)

def log10(x, eps=1e-5):
  """Logarithm with base 10."""
  return logb(x, base=10, eps=eps)

def power_to_db(power, ref_db=0.0, range_db=DB_RANGE, use_tf=True):
  """Converts power from linear scale to decibels."""
  # Choose library.
  maximum = tf.maximum if use_tf else np.maximum
  log_base10 = log10 if use_tf else np.log10

  # Convert to decibels.
  pmin = 10**-(range_db / 10.0)
  power = maximum(pmin, power)
  db = 10.0 * log_base10(power)

  # Set dynamic range.
  db -= ref_db
  db = maximum(db, -range_db)
  return db

Number = TypeVar('Number', int, float, np.ndarray, tf.Tensor)

def tf_float32(x):
  """Ensure array/tensor is a float32 tf.Tensor."""
  if isinstance(x, tf.Tensor):
    return tf.cast(x, dtype=tf.float32)  # This is a no-op if x is float32.
  else:
    return tf.convert_to_tensor(x, tf.float32)

def hz_to_midi(frequencies: Number) -> Number:
  """TF-compatible hz_to_midi function."""
  frequencies = tf_float32(frequencies)
  notes = 12.0 * (logb(frequencies, 2.0) - logb(440.0, 2.0)) + 69.0
  # Map 0 Hz to MIDI 0 (Replace -inf MIDI with 0.)
  notes = tf.where(tf.less_equal(frequencies, 0.0), 0.0, notes)
  return notes