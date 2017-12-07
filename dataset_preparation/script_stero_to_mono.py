import pydub
from pydub import AudioSegment
import os
from os import listdir
from os.path import isfile, join
import glob
import ntpath

#
# At conda command line, ran:
#     pip install pydub
#

#possum_in = 'C:/Users/richa/Audio/T1-COMPGI23_DL_Group_tmp/data/2SecApprox/1224941_0003_011 ( mp3cut.net)(7)-16khz.wav'
#possum_out = 'C:/Users/richa/Audio/T1-COMPGI23_DL_Group_tmp/data/2SecApprox/1224941_0003_011 ( mp3cut.net)(7)-16khz-mono.wav'



def split_audio(input_directory, left_output_directory, right_output_directory):
    if not os.path.exists(right_output_directory):
        os.makedirs(right_output_directory)
    if not os.path.exists(left_output_directory):
        os.makedirs(left_output_directory)

    for filename in glob.glob(input_directory+"/*.wav"):
        possum = AudioSegment.from_wav(filename)

        basename = ntpath.basename(filename)

        possum_channels = possum.split_to_mono()
        for channel in range(len(possum_channels)):
            if channel == 0:
                possum_channels[channel].export(left_output_directory + '/' + basename.replace(".wav","") + ".left.wav", format="wav")
            else:
                possum_channels[channel].export(right_output_directory + '/' + basename.replace(".wav","") + ".right.wav", format="wav")




split_audio("C:/projects/GitHub/T1-COMPGI23-DL/T1-COMPGI23_DL_Group/data/2SecApprox","c:/tmp/tmp/left","c:/tmp/tmp/right")










