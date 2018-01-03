import pydub as pd
import numpy as np
from pydub import AudioSegment
import matplotlib.pyplot as plt
import copy
import ntpath
import glob
import os
#run C:/Users/richa/Audio/T1-COMPGI23_DL_Group/simple_audio/tensorflow/examples/speech_command/playback
import dataset_preparation.playback as pb
import shutil as sh


from tkinter import *
from tkinter import messagebox
import random
import sqlite3 as sq3
import pandas.io.sql as pds
import pandas as p

dataset_folder = 'C:/Users/richa/Audio/dataset_v2_source/'
#folder = 'C:/Users/richa/Audio/dataset_v2_source/speech/'
#folder_to_check = 'bird_warblr_chunks'
folder_to_check = 'bird_ff_chunks'
dbName = 'dataset_v2_checks.db'

conn = sq3.connect(dataset_folder + dbName)
if False:
  sql = "CREATE TABLE WavFileCheck (Folder text, File text, Accept bit, Comment text)"
  conn.execute(sql)
if False:
  sql = "DELETE FROM WavFileCheck"
  conn.execute(sql)

sql = "select Folder, File, Accept, Comment from WavFileCheck where Folder = '" + folder_to_check + "'"
existing_checks = pds.read_sql(sql, conn)

wav_folder = os.path.join(dataset_folder, folder_to_check)
g = glob.glob(wav_folder + '/*.{}'.format('wav'))
all_wav_files = [a[1] for a in (os.path.split(x) for x in g)]

wav_files_to_check = list(set(all_wav_files) - set(existing_checks.File))

random.shuffle(wav_files_to_check)

for f in wav_files_to_check:
  song = AudioSegment.from_wav(os.path.join(wav_folder, f))
  print(f + ': <d> to delete')
  pb.play(song)
  a = input()

  c = str(True)
  if a == 'd':
    print('Marking as deleted')
    c = str(False)

  if a == 'x':
    break

  d = ''
  sql = "INSERT INTO WavFileCheck VALUES ('" + folder_to_check + "','" + f + "', '" + c + "', '" + d + "')"
  conn.execute(sql)
  conn.commit()



sql = "select Folder, File, Accept, Comment from WavFileCheck where Folder = '" + folder_to_check + "'"
existing_checks = pds.read_sql(sql, conn)

# Play all rejected files
print('Rejected: ' + str(sum(existing_checks.Accept == 'False')))
for f in existing_checks[existing_checks.Accept == 'False'].File:
  song = AudioSegment.from_wav(os.path.join(wav_folder, f))
  print(f + ": marked 'rejected'")
  pb.play(song)

# Play all accepted files
print('Accepted: ' + str(sum(existing_checks.Accept == 'True')))
for f in existing_checks[existing_checks.Accept == 'True'].File:
  song = AudioSegment.from_wav(os.path.join(wav_folder, f))
  print(f + ": marked 'accepted'")
  pb.play(song)


# Copy accepted files to file dataset folder

dataset_output_folder = 'C:/Users/richa/Audio/dataset_v2/'
bird_source_folders = ('bird_warblr_chunks', 'bird_ff_chunks')
bird_dest_folder = os.path.join(dataset_output_folder, 'bird')
os.makedirs(bird_dest_folder, exist_ok=True)

for folder in bird_source_folders:
  bird_source_folder = os.path.join(dataset_folder, folder)

  sql = "select Folder, File, Accept, Comment from WavFileCheck where Folder = '" + folder + "'"
  checks = pds.read_sql(sql, conn)
  files_to_copy = checks[checks.Accept == 'True'].File

  print("Copying {} accepted files for: {}...".format(str(len(files_to_copy)), folder))

  for f in files_to_copy:
    sh.copy(os.path.join(bird_source_folder, f), bird_dest_folder)





# Play final bird files...
for f in glob.glob(bird_dest_folder + '/*.{}'.format('wav')):
  song = AudioSegment.from_wav(f)
  print(f + ": copied into final dataset folder'")
  pb.play(song)


  #ans = messagebox.askyesnocancel(b, 'Delete the file?')
  #if ans == True:
  #  #os.remove(f)
  #  print('Removed ' + f)
  #elif ans == False:
   # print('Kept ' + f)
  #else:
  #  break


# Play final bird files...
import random
folder_to_play = 'possum'
g = glob.glob(os.path.join(dataset_output_folder, folder_to_play) + '/*.{}'.format('wav'))
random.shuffle(g)
for f in g:
  song = AudioSegment.from_wav(f)
  print(f + ": copied into final dataset folder'")
  pb.play(song)




