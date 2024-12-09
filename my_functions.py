# Collections of utility functions used in music gen project
import sqlite3
import collections
import pandas as pd
import pretty_midi
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
from typing import Optional
import seaborn as sns



# Generate training data from database

def extract_training_data() -> pd.DataFrame :
    try:
        conn = sqlite3.connect("wjazzd.db")
    except Exception as e:
        print(e)

    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    #print(f"Table Name : {cursor.fetchall()}")

    df = pd.read_sql_query('''
                        SELECT melody.melid, 
                           melody.pitch,
                           melody.pitch / 128 as pitch_norm,
                           melody.onset as start,
                           melody.onset + melody.duration as end,
                           melody.duration, 
                           solo_info.instrument,
                           solo_info.key,
                           solo_info.style,
                           solo_info.avgtempo as tempo,
                           solo_info.rhythmfeel as feel,
                           solo_info.title,
                           solo_info.performer
                        FROM melody
                        JOIN solo_info
                        ON melody.melid = solo_info.melid
                        ''', conn)
    conn.close()
    # Calc the gap between start of consecutive notes
    df['step'] = df['start'].shift(-1) - df['start']
    #fix problems at boundaries
    df.fillna({ 'step': df['step'].median()}, inplace=True)

    # Calculate the inverval between successive notes
    df['interval'] = df['pitch'].shift(-1) - df['pitch']
    #fix problems at boundaries
    df.fillna({'interval': 0}, inplace=True)

    # apply a contour function
    df['contour'] = df['interval'].apply(contour)
    df['contour'] = df['contour'].astype('float32')
    # fix out of bound steps.  Negative steps will have the median step size
    median = df['step'].median()
    df.loc[df['step'] < 0.0, 'step'] = median

    #binds interval to range to ensure vocab size, two octaves either direction
    df['interval'] = np.clip(df['interval'], -24.0, +24.0)

    return df

# Trim data for even number of sequences
def trim_data_sequence(
      pitchInst: pd.DataFrame, 
      skip_list: list[int],
      seq_length: int,
      key_order: list[str]
) -> tf.data.Dataset:
    #REMOVE REMAINDERS FROM TRAINING SET HERE
    dfs = dict(tuple(pitchInst.groupby('melid')))

    # Empty array to build up
    train_subset = pd.DataFrame(None, columns=pitchInst.columns)
    # Loop through the solos
    for i, df in dfs.items():
        # skip the first 10 songs so they can be used for test generation
        if (i not in skip_list):
            n = len(dfs[i])%seq_length  # leftovers
            # uncomment this next line to truncate each song to seq_lenght multiple
            dfs[i].drop(df.tail(n).index, inplace = True) # Drop the remnants
            train_subset = pd.concat([train_subset, dfs[i]], ignore_index=True)  # append to the set

    # n_notes will be used later to build batches
    return len(train_subset), np.stack([train_subset[key] for key in key_order], axis=1)

# from tensorFlow MusGen tutorial 
def create_sequences(
    dataset: tf.data.Dataset,
    key_order: list[str],
    seq_length: int,
    vocab_size: int,
) -> tf.data.Dataset:
  """Returns TF Dataset of sequence and label examples."""
  seq_length = seq_length+1

  # Take 1 extra for the labels
  windows = dataset.window(seq_length, shift=1, stride=1,
                              drop_remainder=True)

  # `flat_map` flattens the" dataset of datasets" into a dataset of tensors
  flatten = lambda x: x.batch(seq_length, drop_remainder=True)
  sequences = windows.flat_map(flatten)

  # Split the labels
  def split_labels(sequences):
    inputs = sequences[:-1]
    labels_dense = sequences[-1]
    labels = {key:labels_dense[i] for i,key in enumerate(key_order)}

    return inputs, labels

  return sequences.map(split_labels, num_parallel_calls=tf.data.AUTOTUNE)



# Generate a contour of intervals
def contour(interval: int): 
  if 12 >= interval > 4:
    return 1
  elif interval > 12:
    return 2
  elif -12 <= interval < -4:
    return -1
  elif interval < -12:
    return -2
  else:
    return 0

# Function to take a frame of notes and convert them into a midi file
def notes_to_midi(
  notes: pd.DataFrame,
  out_file: str, 
  instrument_name: str,
  velocity: int = 100,  # note loudness
) -> pretty_midi.PrettyMIDI:

  pm = pretty_midi.PrettyMIDI(initial_tempo=notes['tempo'][0])
  instrument = pretty_midi.Instrument(
      program=pretty_midi.instrument_name_to_program(
          instrument_name))

  start = 0
  for i, note in notes.iterrows():
    next_start = float(start + note['step'])
    end = float(start + note['duration'])
    note = pretty_midi.Note(
        velocity=velocity,
        pitch=int(note['pitch']),
        start=start,
        end=end,
    )
    instrument.notes.append(note)
    start = next_start

  pm.instruments.append(instrument)
  pm.write(out_file)
  return pm

# Function to convert a midi file into a sequence of "Notes"
def midi_to_notes(midi_file: str):
  midi_data = pretty_midi.PrettyMIDI(midi_file)
  instrument = midi_data.instruments[0]
  notes = collections.defaultdict(list)

  sorted_notes = sorted(instrument.notes, key=lambda note: note.start)
  prev_start = sorted_notes[0].start

  for note in sorted_notes:
    start = note.start
    end = note.end
    notes['pitch'].append(note.pitch)
    notes['note_name'].append(pretty_midi.note_number_to_name(note.pitch))
    notes['start'].append(start)
    notes['end'].append(end)
    notes['step'].append(start-prev_start)
    notes['duration'].append(end - start)
    prev_start = start

  return pd.DataFrame({name: np.array(value) for name, value in notes.items()})

def plot_piano_roll(notes: pd.DataFrame, count: Optional[int] = None, heading: Optional[str] = None):
  if count:
    title = f'First {count} notes'
  else:
    title = f'Whole track'
    count = len(notes['pitch'])
  if heading:
     title = heading
  plt.figure(figsize=(20, 4))
  plot_pitch = np.stack([notes['pitch'], notes['pitch']], axis=0)
  plot_start_stop = np.stack([notes['start'], notes['end']], axis=0)
  plt.plot(
      plot_start_stop[:, :count], plot_pitch[:, :count], color="b", marker=".")
  plt.xlabel('Time [s]')
  plt.ylabel('Pitch')
  _ = plt.title(title)

def plot_distributions(notes: pd.DataFrame, title='', drop_percentile=2.5):
  plt.figure(figsize=[15, 5])
  plt.subplot(1, 3, 1)
  sns.histplot(notes, x="pitch", bins=20)

  plt.subplot(1, 3, 2)
  plt.title(title)

  max_step = np.percentile(notes['step'], 100 - drop_percentile)
  sns.histplot(notes, x="step", bins=np.linspace(0, max_step, 21))
  
  plt.subplot(1, 3, 3)
  max_duration = np.percentile(notes['duration'], 100 - drop_percentile)
  sns.histplot(notes, x="duration", bins=np.linspace(0, max_duration, 21))

def plot_notes(
      notes: pd.DataFrame,
      t_len: int,
      heading: str):
  count = len(notes['pitch'])
  plt.figure(figsize=(20, 4))
  plot_pitch = np.stack([notes['pitch'].head(t_len), notes['pitch'].head(t_len)], axis=0)
  plot_start_stop = np.stack([notes['start'], notes['end']], axis=0)
  plt.plot(
      plot_start_stop[:, :t_len], plot_pitch[:, :t_len], color="b", marker=".")
  plot_pitch = np.stack([notes['pitch'].tail(count - t_len), notes['pitch'].tail(count - t_len)], axis=0)
  plot_start_stop = np.stack([notes['start'].tail(count - t_len), notes['end'].tail(count - t_len)], axis=0)
  plt.plot(
      plot_start_stop[:t_len, :count], plot_pitch[:t_len, :count], color="r", marker=".")
  plt.xlabel('Time [s]')
  plt.ylabel('Pitch')
  _ = plt.title(heading)

def select_data_sequences(
      pitchInst: pd.DataFrame, 
      get_list: list[int],
      seq_length: int,
      key_order: list[str]
) -> tf.data.Dataset:
    #REMOVE REMAINDERS FROM TRAINING SET HERE
    dfs = dict(tuple(pitchInst.groupby('melid')))

    # Empty array to build up
    train_subset = pd.DataFrame(None, columns=pitchInst.columns)
    # Loop through the solos
    for i, df in dfs.items():
        # skip the first 10 songs so they can be used for test genearation
        if (i in get_list):
            n = len(dfs[i])%seq_length  # leftovers
            # uncomment this next line to truncate each song to seq_lenght multiple
            dfs[i].drop(df.tail(n).index, inplace = True) # Drop the remnants
            train_subset = pd.concat([train_subset, dfs[i]], ignore_index=True)  # append to the set

    # n_notes will be used later to build batches
    return len(train_subset), np.stack([train_subset[key] for key in key_order], axis=1)
