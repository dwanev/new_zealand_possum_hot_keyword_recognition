## RS 02-Dec-17
# Script to create dataset_v0

import os
import sys
import shutil as sh
import glob
import pandas as pds

dataset_folder = 'C:/Users/richa/Audio/dataset_v0/'
dataset_source_folder = 'C:/Users/richa/Audio/dataset_v0_source/'

## STEP 1: Populate dataset_v0_source
print('Creating folder: ' + dataset_source_folder + '...')
if os.path.exists(dataset_source_folder):
  sh.rmtree(dataset_source_folder)
os.makedirs(dataset_source_folder, exist_ok=True)

# Speech words
speech_folder = 'C:/Users/richa/Audio/dataset/'
speech_words = ('bed', 'on', 'seven', 'yes')
word_dest_folder = dataset_source_folder + 'speech/'
print('Creating folder: ' + word_dest_folder + '...')
os.makedirs(word_dest_folder, exist_ok=True)
for w in speech_words:
  word_orig_folder = speech_folder + w
  print('Copying: ' + w + '...')
  for f in glob.glob(word_orig_folder + '/*.{}'.format('wav')):
    sh.copy(f, word_dest_folder)


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
for f in glob.glob(possum_folder_orig + '/*.{}'.format('wav')):
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
bird_source_folder = dataset_source_folder + 'bird/'
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


##
#list_wav_file_details.py