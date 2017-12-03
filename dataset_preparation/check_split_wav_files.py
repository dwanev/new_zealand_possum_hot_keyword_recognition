import pydub as pd
import numpy as np
from pydub import AudioSegment
import matplotlib.pyplot as plt
import copy
import ntpath
import glob
import os
run C:/Users/richa/Audio/T1-COMPGI23_DL_Group/simple_audio/tensorflow/examples/speech_command/playback


from tkinter import *
from tkinter import messagebox



folder = 'C:/Users/richa/Audio/dataset_v0/bird/'
for f in glob.glob(folder + '/*.{}'.format('wav')):
  song = AudioSegment.from_wav(f)
  play(song)
  ans = messagebox.askyesnocancel(f, 'Delete the file?')
  if ans == True:
    os.remove(f)
    print('Removed ' + f)
  elif ans == False:
    print('Kept ' + f)
  else:
    break





mainloop()



top = Tk()
top.geometry("300x800")
def hello():
   messagebox.showinfo("Say Hello", "Hello World")

def shut():
  messagebox.askyesno('test')
B1 = Button(top, text = "Say Hello", command = hello)
B2 = Button(top, text = "Close", command = shut)
B1.place(x = 35,y = 50)
B2.place(x = 35,y = 70)

top.mainloop()



root = Tk()

w = Label(root, text="Hello, world!")
w.pack()

root.mainloop()


