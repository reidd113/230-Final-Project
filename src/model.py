from ddsp import core, synths, processors
import tensorflow as tf
tfk = tf.keras
tfkl = tfk.layers

class DDSP_Decoder_manual(tfk.Layer):

    def __init__(self, nhid=256, n_frames=1000, frame_size=64, sample_rate=16000, **kwargs):
        super(DDSP_Decoder_manual, self).__init__(**kwargs)
        self.nhid = nhid
        self.n_frames = n_frames
        self.frame_size = frame_size
        self.sample_rate = sample_rate

        # Initialize synths and processors as layers
        self.harmonic_synth = synths.Harmonic(self.n_frames * self.frame_size, self.sample_rate,
                                            use_angular_cumsum=True)
        self.noise_synth = synths.FilteredNoise(self.n_frames * self.frame_size, self.sample_rate)
        self.add = processors.Add(name='add')
        # Initialize dense layers within the __init__ method
        self.harmonic_amp_layer = tfkl.Dense(1, bias_initializer='ones')
        self.harmonic_distribution_layer = tfkl.Dense(100)
        self.noise_mag_layer = tfkl.Dense(65)

    def compute_output_shape(self, input_shape):
        """Computes the output shape of the layer.

        Args:
        input_shape: Shape tuple (tuple of integers) or list of shape tuples (one
            per output tensor of the layer). Shape tuples can include None for free
            dimensions, instead of an integer.

        Returns:
        Shape tuple (tuple of integers) or list of shape tuples (one per output
            tensor of the layer).
        """
        # Assuming the output shape is (batch_size, n_frames * frame_size)
        batch_size = input_shape[0][0]  # Get batch size from the first input (z)
        return (batch_size, self.n_frames * self.frame_size)

    def process_audio(self, z, f0):
        # Call the dense layers to get harmonic_amp, harmonic_distribution, and noise_mag
        harmonic_amp = self.harmonic_amp_layer(z)
        harmonic_distribution = self.harmonic_distribution_layer(z)
        noise_mag = self.noise_mag_layer(z)
        
        # harmonic_signal = self.harmonic_synth(harmonic_controls["amplitudes"],harmonic_controls["harmonic_distribution"],harmonic_controls["f0_hz"])
        harmonic_signal = self.harmonic_synth(harmonic_amp,harmonic_distribution,f0)

        noise_signal = self.noise_synth(noise_mag)

        synth_audio = self.add(noise_signal, harmonic_signal)
        return synth_audio


    def call(self, inputs):
        z, f0 = inputs
        TensorArr = tf.TensorArray(tf.float32, 1, dynamic_size=True, infer_shape=False)
        synth_audio = self.process_audio(z, f0)
        # print(synth_audio)
        return synth_audio

def get_process_group(n_frames, frame_size=64, sample_rate=16000,
                use_angular_cumsum=True):
    harmonic_synth = synths.Harmonic(n_frames * frame_size, sample_rate,
                                            use_angular_cumsum=use_angular_cumsum)
    noise_synth = synths.FilteredNoise(n_frames * frame_size, sample_rate)
    add = processors.Add(name='add')
    # Create ProcessorGroup.
    # Change here: Use string names for modules
    dag = [('harmonic_synth', ['amplitudes', 'harmonic_distribution', 'f0_hz']),
            ('noise_synth', ['noise_magnitudes']),
            ('add', ['noise_synth/signal', 'harmonic_synth/signal'])]

    processor_group = processors.ProcessorGroup(dag=dag, name='processor_group')

    # Add modules as attributes to the processor_group
    processor_group.harmonic_synth = harmonic_synth
    processor_group.noise_synth = noise_synth
    processor_group.add = add

    return processor_group

def ConvBlock(inputs, out_channels, pool_size=(2, 2)):
  x = inputs
  x = tfkl.Conv2D(filters=out_channels,
                  kernel_size=(3, 3), strides=(1, 1),
                  padding='same', use_bias=False,
                  kernel_initializer=
                  tfk.initializers.GlorotUniform())(x)
  x = tfkl.BatchNormalization(beta_initializer='zeros',
                              gamma_initializer='ones')(x)
  x = tfkl.ReLU()(x)
  x = tfkl.Conv2D(filters=out_channels,
                  kernel_size=(3, 3), strides=(1, 1),
                  padding='same', use_bias=False,
                  kernel_initializer=
                  tfk.initializers.GlorotUniform())(x)
  x = tfkl.BatchNormalization(beta_initializer='zeros',
                              gamma_initializer='ones')(x)
  x = tfkl.ReLU()(x)
  x = tfkl.AveragePooling2D(pool_size=pool_size, padding='same')(x)

  return x


def CNN(inputs, pool_size=(1,2), dropout=0.2, nhid=256):
  x = inputs
  x = ConvBlock(x, out_channels=64, pool_size=pool_size)
  x = tfkl.Dropout(rate=dropout)(x)
  x = ConvBlock(x, out_channels=128, pool_size=pool_size)
  x = tfkl.Dropout(rate=dropout)(x)
  x = ConvBlock(x, out_channels=256, pool_size=pool_size)
  x = tfkl.Dropout(rate=dropout)(x)
  x = ConvBlock(x, out_channels=512, pool_size=pool_size)
  x = tfkl.Dropout(rate=dropout)(x)
  x = tfkl.Reshape((1000, -1))(x)
  x = tfkl.Dense(nhid)(x)

  return x

def DDSP_Encoder(inputs, nhid=256):
  mel = inputs['mel']
  z_cnn = CNN(mel, nhid=nhid)

  x = tfkl.Concatenate(axis=-1)([inputs['f0_hz'], inputs['loudness_db']])
  #x = inputs['f0_loudness']
  #x = tf.concat([hz_to_midi(inputs['f0_hz']) / F0_RANGE,
  #                 inputs['loudness_db'] / DB_RANGE], -1)
  x_z = tfkl.Dense(nhid)(x)
  #x_z = tfkl.Reshape((1,-1))(x_z)
  x_z_concat = tfkl.Concatenate(axis=-1)([x_z, z_cnn])
  z_out = tfkl.Bidirectional(tfkl.LSTM(units=nhid, return_sequences=True), name='bilstm')(x_z_concat)

  return z_out

def DDSP_Decoder(inputs, nharmonic=100, nnoise=65):
  z, data = inputs

  harmonic_amp = tfkl.Dense(1, bias_initializer='ones')(z)
  harmonic_distribution = tfkl.Dense(nharmonic)(z)
  noise_mag = tfkl.Dense(nnoise)(z)

  synth_params = {
      'f0_hz': data['f0_hz'],
      'amplitudes': harmonic_amp,
      'harmonic_distribution': harmonic_distribution,
      'noise_magnitudes': noise_mag,
    }

  n_frames = inputs[0].shape[1]
  frame_size = 64 #todo: set this better or get from input

  processing_group = get_process_group(inputs[0].shape[1])
  # print("type(processing_group):", type(processing_group))

  controls = processing_group.get_controls(synth_params)
  # Convert KerasTensors to NumPy arrays before passing to get_controls
  #synth_params_numpy = {k: v.numpy() for k, v in synth_params.items()}

  #control_params = processing_group.get_controls(synth_params, verbose=False)
  synth_audio = processing_group.get_signal(controls)

  return synth_audio

def DDSP_model():
  f0_hz_input = tfk.Input(shape=(1000,1), name="f0_hz")
  loudness_db_input = tfk.Input(shape=(1000,1), name="loudness_db")
  mel_input = tfk.Input(shape=(1000, 64, 1), name="mel")
  inputs = {'f0_hz': f0_hz_input, 'loudness_db': loudness_db_input, 'mel': mel_input}

  z = DDSP_Encoder(inputs)
  synth_audio = DDSP_Decoder_manual()([z, f0_hz_input])
  model = tfk.Model(inputs=[f0_hz_input, loudness_db_input, mel_input], outputs=synth_audio)

  return model
