## RS 02-Dec-17
# Script to create dataset_v0

import os
import sys
import shutil as sh
import glob
import pandas as pds
import numpy as np
import ntpath
import pydub
from pydub import AudioSegment

dataset_folder = 'C:/Users/richa/Audio/dataset_v2/'
dataset_source_folder = 'C:/Users/richa/Audio/dataset_v2_source/'

## STEP 1: Populate dataset_v2_source
print('Creating folder: ' + dataset_source_folder + '...')
if os.path.exists(dataset_source_folder):
  sh.rmtree(dataset_source_folder)
os.makedirs(dataset_source_folder, exist_ok=True)

###########################
# Speech words
# Copy 20 samples from each of the word categories
speech_folder = 'C:/Users/richa/Audio/dataset/'
max_files_per_word = 20

g = glob.glob(speech_folder + '/*/')
speech_word_folders = [f[len(speech_folder):].replace("\\", "") for f in g]
del(speech_word_folders[-1])
num_word_folders = len(speech_word_folders)
print("{} word folders".format(num_word_folders))
print(speech_word_folders)

word_dest_folder = dataset_source_folder + 'speech/'
print('Creating folder: ' + word_dest_folder + '...')
os.makedirs(word_dest_folder, exist_ok=True)
for w in speech_word_folders:
  word_orig_folder = speech_folder + w
  print('Copying: ' + w + '...')
  word_files = glob.glob(word_orig_folder + '/*.{}'.format('wav'))
  for f in word_files[0:max_files_per_word]:
    (a, b) = os.path.split(f)
    #sh.copy(f, word_dest_folder)
    sh.copy(f, os.path.join(word_dest_folder, w + '_' + b))


# Cats
cat_source_folder = dataset_source_folder + 'cat/'
print('Creating folder: ' + cat_source_folder + '...')
os.makedirs(cat_source_folder, exist_ok=True)
cat_folder_orig = ('C:/Users/richa/Documents/CSML/Term1/COMPGI23_IntroductionToDeepLearning/Project/KaggleCatsAndDogs/cats_dogs/train/cat/',
  'C:/Users/richa/Documents/CSML/Term1/COMPGI23_IntroductionToDeepLearning/Project/KaggleCatsAndDogs/cats_dogs/test/cats/')

for s in cat_folder_orig:
  for f in glob.glob(s + '/*.{}'.format('wav')):
    sh.copy(f, cat_source_folder)


# Dogs
dog_source_folder = dataset_source_folder + 'dog/'
print('Creating folder: ' + dog_source_folder + '...')
os.makedirs(dog_source_folder, exist_ok=True)
dog_folder_orig = ('C:/Users/richa/Documents/CSML/Term1/COMPGI23_IntroductionToDeepLearning/Project/KaggleCatsAndDogs/cats_dogs/train/dog/',
  'C:/Users/richa/Documents/CSML/Term1/COMPGI23_IntroductionToDeepLearning/Project/KaggleCatsAndDogs/cats_dogs/test/test/')

for s in dog_folder_orig:
  for f in glob.glob(s + '/*.{}'.format('wav')):
    sh.copy(f, dog_source_folder)


# Possums
possum_folder_orig = 'C:/Users/richa/Audio/T1-COMPGI23_DL_Group_tmp/data/2SecApprox/'
possum_source_folder = dataset_source_folder + 'possum/'
print('Creating folder: ' + possum_source_folder + '...')
os.makedirs(possum_source_folder, exist_ok=True)
for f in glob.glob(possum_folder_orig + '/*.left.{}'.format('wav')):
  sh.copy(f, possum_source_folder)

# Ground parrots
parrot_folder_orig = 'C:/Users/richa/Documents/CSML/Term1/COMPGI23_IntroductionToDeepLearning/Project/KaggleParrot/Training/GroundParrot/'
# also 'Others' folder - check it out
parrot_source_folder = dataset_source_folder + 'groundparrot/'
print('Creating folder: ' + parrot_source_folder + '...')
os.makedirs(parrot_source_folder, exist_ok=True)
for f in glob.glob(parrot_folder_orig + '/*.{}'.format('wav')):
  sh.copy(f, parrot_source_folder)

# Random noises
random_folder_orig = 'C:/Users/richa/Documents/CSML/Term1/COMPGI23_IntroductionToDeepLearning/Project/KaggleESC50/wav_concatenated_compressed/'
random_source_folder = dataset_source_folder + 'random/'
print('Creating folder: ' + random_source_folder + '...')
os.makedirs(random_source_folder, exist_ok=True)
for f in glob.glob(random_folder_orig + '/*.{}'.format('wav')):
  sh.copy(f, random_source_folder)


# Birds
bird_ff_folder_orig = 'C:/Users/richa/Documents/CSML/Term1/COMPGI23_IntroductionToDeepLearning/Project/Code/BirdListeningChallenge/ff1010bird_wav/wav/'
bird_source_folder = dataset_source_folder + 'bird_ff/'
print('Creating folder: ' + bird_source_folder + '...')
os.makedirs(bird_source_folder, exist_ok=True)

bird_metadata_file = 'C:/Users/richa/Documents/CSML/Term1/COMPGI23_IntroductionToDeepLearning/Project/Code/BirdListeningChallenge/ff1010bird_metadata.csv'

ff_meta = pds.read_csv(bird_metadata_file)
ff_meta_bird = ff_meta[ff_meta.hasbird == 1]
for f in ff_meta_bird.itemid:
  sh.copy(bird_ff_folder_orig + str(f) + '.wav', bird_source_folder)

# Background noise
background_folder_orig = 'C:/Users/richa/Audio/dataset/_background_noise_/'
background_source_folder = dataset_source_folder + '_background_noise_/'
print('Creating folder: ' + background_source_folder + '...')
os.makedirs(background_source_folder, exist_ok=True)
for f in glob.glob(background_folder_orig + '/*.{}'.format('wav')):
  sh.copy(f, background_source_folder)

# Background noise: British Library (.mp3)
background_folder_orig = 'C:/Users/richa/Audio/BritishLibrarySounds/'
background_source_folder = dataset_source_folder + '_background_noise_/'
print('Copying British Library background sounds: ' + background_source_folder + '...')
for f in glob.glob(background_folder_orig + '/*.{}'.format('mp3')):
  sound = AudioSegment.from_mp3(f)

  # Downsample and set to 16 bit sample_width if necessary
  if not sound.frame_rate == 16000 or not sound.sample_width == 2:
    sound = sound.set_frame_rate(16000)
    sound = sound.set_sample_width(2)

  sound_channels = sound.split_to_mono()

  fname = ntpath.basename(f)
  if sound.channels == 2:
    sound_mono_wav = background_source_folder + fname[0:-4]  + '_mono_l.wav'
    sound_channels[0].export(sound_mono_wav, format="wav")

    sound_mono_wav = background_source_folder + fname[0:-4]  + '_mono_r.wav'
    sound_channels[1].export(sound_mono_wav, format="wav")
  else:
    sound_mono_wav = background_source_folder + fname[0:-4] + '_mono.wav'
    sound_channels[0].export(sound_mono_wav, format="wav")




# Birds: Xeno Canto
# PROBLEM: files are in .flac format. Should be able to read them but can't
bird_xc_folder_orig = 'C:/Users/richa/Audio/british-birdsong-dataset/songs/songs/'
bird_xc_source_folder = dataset_source_folder + 'bird_xc/'
print('Creating folder: ' + bird_xc_source_folder + '...')
os.makedirs(bird_xc_source_folder, exist_ok=True)

bird_xc_metadata_file = 'C:/Users/richa/Audio/british-birdsong-dataset/birdsong_metadata.csv'

xc_meta = pds.read_csv(bird_xc_metadata_file)
file_id = ['xc' + str(x) + '.flac' for x in xc_meta.file_id]
for f in file_id:
  song = AudioSegment.from    (bird_xc_folder_orig + f)
  sh.copy(bird_xc_folder_orig + str(f) + '.wav', bird_source_folder)


# Birds: Kaggle
# PROBLEM: had some problems filtering out single-species audio. Audio also very quiet and low quality.
bird_kaggle_folder_orig = 'C:/Users/richa/KaggleBirdChallenge/mlsp_contest_dataset/mlsp_contest_dataset/essential_data/src_wavs/'
bird_kaggle_source_folder = dataset_source_folder + 'bird_kaggle/'
print('Creating folder: ' + bird_kaggle_source_folder + '...')
os.makedirs(bird_kaggle_source_folder, exist_ok=True)

bird_kaggle_metadata_file = 'C:/Users/richa/KaggleBirdChallenge/light_data/KaggleData.xlsx'

kaggle_meta = pds.read_excel(bird_kaggle_metadata_file)
f1 = kaggle_meta.Label=='10'
f2 = kaggle_meta.Label=='1'
f3 = kaggle_meta.Label=='9'
kaggle_meta_bird = kaggle_meta[kaggle_meta.Label=='10' or kaggle_meta.Label=='1' or kaggle_meta.Label=='9']
for f in kaggle_meta_bird.filename:
  sh.copy(bird_kaggle_folder_orig + str(f) + '.wav', bird_kaggle_source_folder)

# Birds: warblr
bird_warblr_folder_orig = 'C:/Users/richa/Documents/CSML/Term1/COMPGI23_IntroductionToDeepLearning/Project/Code/BirdListeningChallenge/warblrb10k_public_wav/wav/'
bird_warblr_source_folder = dataset_source_folder + 'bird_warblr/'
print('Creating folder: ' + bird_warblr_source_folder + '...')
os.makedirs(bird_warblr_source_folder, exist_ok=True)

bird_warblr_metadata_file = 'C:/Users/richa/Documents/CSML/Term1/COMPGI23_IntroductionToDeepLearning/Project/Code/BirdListeningChallenge/warblrb10k_public_metadata.csv'

warblr_meta = pds.read_csv(bird_warblr_metadata_file)
warblr_meta_bird = warblr_meta[warblr_meta.hasbird == 1]
for f in warblr_meta_bird.itemid:
  sh.copy(bird_warblr_folder_orig + str(f) + '.wav', bird_warblr_source_folder)


    ##
#list_wav_file_details.py