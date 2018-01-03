
import glob
import pandas as pds
import numpy as np
import pydub
from pydub import AudioSegment
from pathlib import Path


#dataset_folder = 'C:/Users/richa/Audio/dataset_v0_source/'
#dataset_folder = 'C:/Users/richa/Audio/dataset_v0/'
#dataset_folder = 'C:/Users/richa/Audio/dataset_v1/'
dataset_folder = 'C:/Users/richa/Audio/T1-COMPGI23_DL_Group_tmp/data/2SecApprox/'

def list_wav_file_details(wav_folder=None):
  all_wav_files = glob.glob(wav_folder + '*.{}'.format('wav'))
  all_wav_filenames = [x[len(wav_folder):] for x in all_wav_files]

  num_to_show = len(all_wav_files)
  all_wav_duration = np.zeros((num_to_show, 1))
  all_wav_channels = np.zeros((num_to_show,1))
  all_wav_sample_width = np.zeros((num_to_show, 1))
  all_wav_frame_rate = np.zeros((num_to_show,1))
  all_wav_max_dBFS = np.zeros((num_to_show,1))
  all_wav_rms = np.zeros((num_to_show,1))

  for i in range(num_to_show):
      this_song = AudioSegment.from_wav(all_wav_files[i])
      all_wav_duration[i] = this_song.duration_seconds
      all_wav_channels[i] = this_song.channels
      all_wav_sample_width[i] = this_song.sample_width
      all_wav_frame_rate[i] = this_song.frame_rate
      all_wav_max_dBFS[i] = this_song.max_dBFS
      all_wav_rms[i] = this_song.rms

  df = pds.DataFrame(np.hstack((all_wav_channels, all_wav_sample_width, all_wav_frame_rate, all_wav_duration, all_wav_max_dBFS, all_wav_rms)), index=all_wav_filenames, columns=('Channels', 'SampleWidth', 'Frame Rate', 'Duration', 'max dBFS', 'RMS'))

  return df

def main():

  #p = Path(dataset_folder)
  #s = [x for x in p.iterdir() if x.is_dir()]
  writer = pds.ExcelWriter(dataset_folder + 'wav_file_details.xlsx')

  #g = glob.glob(dataset_folder + '/*/')
  #num_files = np.zeros((len(g),1))
  #for f in g:
  #print('Processing: ' + f + '...')
  df = list_wav_file_details(dataset_folder)
  #l = f[len(dataset_folder):].replace("\\", "")
  #df.to_excel(pds.ExcelWriter(, engine='xlsxwriter'), sheet_name=l)
  df.to_excel(writer, sheet_name = 'Possums')
  #num_files[[x == f for x in g]] = len(df)

  #data_folders = np.array([f[len(dataset_folder):].replace("\\", "") for f in g])
  #data_folders = np.append(data_folders, 'Total')
  #num_files = np.append(num_files, num_files[0:len(data_folders)].sum())
  #df_summary = pds.DataFrame(index=data_folders, data=num_files)
  #df_summary.to_excel(writer, sheet_name = 'Summary')#, columns='Samples')
  writer.save()

if __name__ == '__main__':
  main()
