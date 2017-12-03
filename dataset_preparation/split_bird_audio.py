import pydub as pd
import numpy as np
from pydub import AudioSegment
import matplotlib.pyplot as plt
import copy
import ntpath

window_lookback_ms = 200
chunk_width_ms = 2*1000
pct_song_max_threshold = 0.5


def split_bird_audio(wav_file, dest_folder=None, plot_folder=None):

  # Load wav file
  song = AudioSegment.from_wav(wav_file)

  # Downsample and set to 16 bit sample_width if necessary
  if not song.frame_rate == 16000 or not song.sample_width == 2:
    song = song.set_frame_rate(16000)
    song = song.set_sample_width(2)
    wav_filename_down = wav_file.replace('.wav', '_down.wav')
    #song.export(wav_filename_down , format="wav")

  # Find peak amplitude in each segment
  #millimax = [y.max for y in song]
  millimax = [y.rms for y in song]
  plt.plot(millimax)

  # Calculate mean of millimax in each 2 second window
  window_mean = np.hstack( (np.zeros(np.int(np.floor(window_lookback_ms/2)),),
                           np.array( [np.mean(millimax[i-window_lookback_ms:i]) for i in range(window_lookback_ms, len(song))]) ))
  #plt.plot(window_mean)
  window_mean_full = copy.deepcopy(window_mean)

  # Split file into chunks around peaks in window_mean
  counter = 0
  chunks = [None] * 5
  idx_chunks = np.zeros([5,2])
  song_max = song.max
  song_rms = song.rms
  while counter < 5 and np.max(window_mean) > song_rms: #'(song_max * pct_song_max_threshold):
      m = np.argmax(window_mean)
      idx_start = max(m - int(chunk_width_ms/2), 0)
      idx_end = min(m + int(chunk_width_ms/2), len(song))
      this_chunk = song[idx_start:idx_end]

      idx_chunks[counter, 0] = idx_start
      idx_chunks[counter, 1] = idx_end
      chunks[counter] = this_chunk

      plt.plot([idx_start, idx_end], [0, 0], color='r', linestyle='-', linewidth=2)
      plt.plot(range(idx_start, idx_end), millimax[idx_start:idx_end], color='r', linestyle='-')

      idx_kill_start = max(m - chunk_width_ms, 0)
      idx_kill_end = min(m + chunk_width_ms, len(song))
      window_mean[idx_kill_start:idx_kill_end] = 0

      # Save this chunk as a separate file
      if not dest_folder == None:
        fname = ntpath.basename(wav_file)
        #wav_filename_chunk = wav_file.replace('.wav', '_chunk_' + str(counter) + '.wav')
        wav_filename_chunk = dest_folder + fname.replace('.wav', '_chunk_' + str(counter) + '.wav')
        this_chunk.export(wav_filename_chunk, format="wav")

      counter += 1

  # Tidy up plot
  plt.plot(window_mean_full)
  plt.ylabel('Max in Segment')
  plt.xlabel('Milliseconds')
  plt.grid()
  plt.title(wav_file)

  if not plot_folder == None:
    fname = ntpath.basename(wav_file)
    plt.savefig(plot_folder + fname[0:-4] + '.png')

  plt.close()

#  wav_file = 'C:/Users/richa/Audio/71838.wav'
# split_bird_audio("C:/Users/richa/Audio/71838.wav")
def main():
  #split_bird_audio("C:/Users/richa/Audio/97375.wav")
  #split_bird_audio("C:/Users/richa/Audio/72827.wav", "C:/Users/richa/Audio/")
  #split_bird_audio("C:/Users/richa/Audio/4dd5046d-c962-4f02-a820.wav", "C:/Users/richa/Audio/")

  #split_bird_audio("C:/Users/richa/Audio/PC4_20100804_050000_0020.wav", "C:/Users/richa/Audio/")

  #split_bird_audio("C:/Users/richa/Audio/404-Door-woodcreaks.wav", "C:/Users/richa/Audio/")

  #split_bird_audio("C:/Users/richa/Audio/509-Fireworks.wav", "C:/Users/richa/Audio/")

  split_bird_audio("C:/Users/richa/Audio/dataset_v0_source/cat\\cat_99.wav", "C:/Users/richa/Audio/dataset_v0/cat/")


if __name__ == '__main__':
  main()
