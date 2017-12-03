## RS 02-Dec-17
# Script to create dataset_v0

import os
import sys
import shutil as sh
import glob
import pandas as pds
import ntpath
import pydub
from pydub import AudioSegment

dataset_folder = 'C:/Users/richa/Audio/dataset_v0/'
dataset_source_folder = 'C:/Users/richa/Audio/dataset_v0_source/'

## STEP 1: Populate dataset_v0_source
print('Creating folder: ' + dataset_folder + '...')
if os.path.exists(dataset_folder):
  sh.rmtree(dataset_folder)
os.makedirs(dataset_folder, exist_ok=True)
os.makedirs(dataset_source_folder + '/plots/', exist_ok=True)

# Speech words
speech_source_folder = dataset_source_folder + 'speech/'
speech_dest_folder = dataset_folder + 'speech/'
print('Creating folder: ' + speech_dest_folder + '...')
os.makedirs(speech_dest_folder, exist_ok=True)
for f in glob.glob(speech_source_folder + '/*.{}'.format('wav')):
  sh.copy(f, speech_dest_folder)


# Cats
cat_source_folder = dataset_source_folder + 'cat/'
cat_dest_folder = dataset_folder + 'cat/'
print('Creating folder: ' + cat_dest_folder + '...')
os.makedirs(cat_dest_folder, exist_ok=True)
for f in glob.glob(cat_source_folder + '/*.{}'.format('wav')):
  #sh.copy(f, cat_dest_folder)
  split_bird_audio(f, dest_folder=cat_dest_folder, plot_folder='C:/Users/richa/Audio/dataset_v0_source/plots/')


## SUN: run sim on dataset; move to OneDrive; document; GIT upload code; message team


# Dogs
dog_source_folder = dataset_source_folder + 'dog/'
dog_dest_folder = dataset_folder + 'dog/'
print('Creating folder: ' + dog_dest_folder + '...')
os.makedirs(dog_dest_folder, exist_ok=True)
for f in glob.glob(dog_source_folder + '/*.{}'.format('wav')):
  #sh.copy(f, cat_dest_folder)
  split_bird_audio(f, dest_folder=dog_dest_folder, plot_folder='C:/Users/richa/Audio/dataset_v0_source/plots/')


# Possums
possum_source_folder = dataset_source_folder + 'possum/'
possum_dest_folder = dataset_folder + 'possum/'
print('Creating folder: ' + possum_dest_folder + '...')
os.makedirs(possum_dest_folder, exist_ok=True)
for f in glob.glob(possum_source_folder + '/*.{}'.format('wav')):
  #sh.copy(f, possum_dest_folder)
  possum = AudioSegment.from_wav(f)
  possum_channels = possum.split_to_mono()
  fname = ntpath.basename(f)
  possum_mono_wav = possum_dest_folder + fname[0:-4]  + '_mono.wav'
  possum_channels[0].export(possum_mono_wav, format="wav")

# Ground parrots
parrot_source_folder = dataset_source_folder + 'groundparrot/'
parrot_dest_folder = dataset_folder + 'groundparrot/'
print('Creating folder: ' + parrot_dest_folder + '...')
os.makedirs(parrot_dest_folder, exist_ok=True)
for f in glob.glob(parrot_source_folder + '/*.{}'.format('wav')):
  sh.copy(f, parrot_dest_folder)


# Random noises
random_source_folder = dataset_source_folder + 'random/'
random_dest_folder = dataset_folder + 'random/'
print('Creating folder: ' + random_dest_folder + '...')
os.makedirs(random_dest_folder, exist_ok=True)
for f in glob.glob(random_source_folder + '/*.{}'.format('wav')):
  #sh.copy(f, cat_dest_folder)
  split_bird_audio(f, dest_folder=random_dest_folder, plot_folder='C:/Users/richa/Audio/dataset_v0_source/plots/')


# Birds
bird_source_folder = dataset_source_folder + 'bird/'
bird_dest_folder = dataset_folder + 'bird/'
print('Creating folder: ' + bird_dest_folder + '...')
os.makedirs(bird_dest_folder, exist_ok=True)
for f in glob.glob(bird_source_folder + '/*.{}'.format('wav')):
  #sh.copy(f, cat_dest_folder)
  split_bird_audio(f, dest_folder=bird_dest_folder, plot_folder='C:/Users/richa/Audio/dataset_v0_source/plots/')


# Background noise
background_source_folder = dataset_source_folder + '_background_noise_/'
background_dest_folder = dataset_folder + '_background_noise_/'
print('Creating folder: ' + background_dest_folder + '...')
os.makedirs(background_dest_folder, exist_ok=True)
for f in glob.glob(background_source_folder + '/*.{}'.format('wav')):
  sh.copy(f, background_dest_folder)


##
#list_wav_file_details