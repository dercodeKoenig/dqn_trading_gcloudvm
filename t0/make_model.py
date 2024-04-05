
## tx + gru large

import tensorflow as tf

#config
batch_size = 32
gamma = 0.995
learning_rate=0.00005
num_data_generation_threads = 4 #12
batch_generation_threads = 8    #8
#num_data_generation_threads = 1
#batch_generation_threads = 4
memory_size = 50_000
ep_len = 100


num_model_inputs = 3+5+3+1
n_actions = 3
batch_q_size = 512
data_q_maxlen = 128
target_model_sync = 5000

save_freq = 20 # save after x epochs


from tensorflow.keras import layers
from tensorflow import keras

class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super().__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim),]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training=False):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)


class PositionEmbedding(layers.Layer):
    def __init__(self, maxlen, embed_dim):
        super().__init__()
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        maxlen = tf.shape(x)[-2]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        #print(positions.shape, x.shape)
        return x+positions


def make_model():

  tx_units = 32+5
  c_len = 120*4
  chart_m15 = tf.keras.layers.Input(shape = (c_len,tx_units))
  chart_m5 = tf.keras.layers.Input(shape = (c_len,tx_units))
  chart_m1 = tf.keras.layers.Input(shape = (c_len,tx_units))
  

  input_closing_prices = tf.keras.layers.Input(shape=(1,))
  input_current_pos = tf.keras.layers.Input(shape=(1,))
  input_current_day_in = tf.keras.layers.Input(shape=(1,))
  input_current_day = tf.keras.layers.Embedding(7,3)(input_current_day_in)
  input_current_day = tf.keras.layers.Flatten()(input_current_day)
  input_closing_times = tf.keras.layers.Input(shape=(1,))


  input_closing_prices2 = tf.keras.layers.Input(shape=(1,))
    
  _2_chart_m15 = tf.keras.layers.Input(shape = (c_len,tx_units))
  _2_chart_m5 = tf.keras.layers.Input(shape = (c_len,tx_units))
  _2_chart_m1 = tf.keras.layers.Input(shape = (c_len,tx_units))
    
    
  
 


  tx_embed_len = 4
  pos_embed_units = 24
  

  total_units = pos_embed_units+tx_units
    

  def embed_information(input_state):
      input_state = tf.keras.layers.Dense(128, activation = "relu")(input_state)
      input_state = tf.keras.layers.Dense(128, activation = "relu")(input_state)
      input_state_tx = tf.keras.layers.Dense(tx_units*tx_embed_len, activation = "relu")(input_state)
      input_state_tx = tf.keras.layers.Reshape((-1,tx_units))(input_state_tx)
      return input_state_tx
  
  def process_chart(chart, additional_info):
    
    input_state_tx = embed_information(additional_info)
    chart = tf.keras.layers.Concatenate(axis=1)([input_state_tx, chart])
    pos_z = tf.zeros_like(chart)
    #print(pos_z.shape)
    positions = PositionEmbedding(c_len+tx_embed_len, tx_units)(pos_z)
    positions = tf.keras.layers.Dense(pos_embed_units)(positions)
    chart = tf.keras.layers.Concatenate()([positions, chart])
    chart = TransformerBlock(total_units, 8, 256)(chart)
    chart = TransformerBlock(total_units, 8, 256)(chart)
    chart = TransformerBlock(total_units, 8, 256)(chart)
    chart = TransformerBlock(total_units, 8, 256)(chart)
    #chart = TransformerBlock(total_units, 24, 256)(chart)
    #chart = TransformerBlock(total_units, 24, 256)(chart)
    #chart = TransformerBlock(total_units, 24, 256)(chart)
    #chart = TransformerBlock(total_units, 24, 256)(chart)
    
    attention_tokens = chart[::,0:tx_embed_len]
    attention_tokens = tf.keras.layers.Flatten()(attention_tokens)
    attention_tokens = tf.keras.layers.Dense(128)(attention_tokens)
    attention_tokens = tf.keras.layers.LeakyReLU()(attention_tokens)
    attention_tokens = tf.keras.layers.Dense(128)(attention_tokens)
    attention_tokens = tf.keras.layers.LeakyReLU()(attention_tokens)
    
    return attention_tokens

  input_state = tf.keras.layers.Concatenate()([input_current_pos, input_closing_prices, input_closing_times, input_current_day])
  output_m15 = process_chart(chart_m15, input_state)
  input_state = tf.keras.layers.Concatenate()([input_current_pos, input_closing_prices, input_closing_times, input_current_day, output_m15])
  output_m5 = process_chart(chart_m5, input_state)
  input_state = tf.keras.layers.Concatenate()([input_current_pos, input_closing_prices, input_closing_times, input_current_day, output_m15, output_m5])
  output_m1 = process_chart(chart_m1, input_state)

  _2_input_state = tf.keras.layers.Concatenate()([input_closing_prices2, input_closing_times, input_current_day])
  _2_output_m15 = process_chart(_2_chart_m15, _2_input_state)
  _2_input_state = tf.keras.layers.Concatenate()([input_closing_prices2, input_closing_times, input_current_day, _2_output_m15])
  _2_output_m5 = process_chart(_2_chart_m5, _2_input_state)
  _2_input_state = tf.keras.layers.Concatenate()([input_closing_prices2, input_closing_times, input_current_day, _2_output_m15, _2_output_m5])
  _2_output_m1 = process_chart(_2_chart_m1, _2_input_state)
  
  dense_input = tf.keras.layers.Concatenate()([input_current_pos, input_closing_prices, input_closing_times, input_current_day, output_m15, output_m5, output_m1, _2_output_m15, _2_output_m5, _2_output_m1])
  

  x = tf.keras.layers.Dense(1024)(dense_input)
  x = tf.keras.layers.LeakyReLU()(x)
    
  x = tf.keras.layers.Dense(1024)(x)
  x = tf.keras.layers.LeakyReLU()(x)
    
  x = tf.keras.layers.Dense(1024)(x)
  x = tf.keras.layers.LeakyReLU()(x)
    

  x = tf.keras.layers.Dense(3, activation = "linear")(x)

  model = tf.keras.Model(inputs = [input_current_pos, input_closing_prices, input_closing_times, input_current_day_in, chart_m1, chart_m5, chart_m15, input_closing_prices2, _2_chart_m1, _2_chart_m5, _2_chart_m15], outputs=x)
  return model
