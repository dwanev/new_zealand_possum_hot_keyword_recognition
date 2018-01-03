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

from tkinter import *
from tkinter import messagebox
import random


#folder = 'C:/Users/richa/Audio/dataset_v2_source/speech/'
folder = 'C:/Users/richa/Audio/dataset_v2/bird_warblr/'
g = glob.glob(folder + '/*.{}'.format('wav'))
#random.shuffle(g)
for f in g:
  (a, b) = os.path.split(f)
  song = AudioSegment.from_wav(f)
  print(f + ': <d> to delete')
  pb.play(song)
  a = input()

  #ans = messagebox.askyesnocancel(b, 'Delete the file?')
  #if ans == True:
  #  #os.remove(f)
  #  print('Removed ' + f)
  #elif ans == False:
   # print('Kept ' + f)
  #else:
  #  break



