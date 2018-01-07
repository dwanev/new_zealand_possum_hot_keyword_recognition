# Deep Networks for Possum Call Detection

Dwane van der Sluis     17127987     ucabdv1@ucl.ac.uk\
Richard Sterry          17115509     filip.svoboda.17@ucl.ac.uk \
Filip Svoboda           17044637     filip.svoboda.17@ucl.ac.uk \
Hadrien de Vaucleroy    13034248     hadrien.vaucleroy.17@ucl.ac.uk             
 

**To train a model and save it to disk use:**

simple_audio.tensorflow.examples.speech_commands.train.py

An example of the most cammon switches is:
simple_audio/tensorflow/examples/speech_commands/train.py --data_url= \
    --data_dir=/project/possum_project/dataset_v2/ \
    --wanted_words=possum,cat,dog,bird --clip_duration_ms=2000 \
    --train_dir=/project/possum_project/tmp/v2_de_01_train/ \
    --summaries_dir=/project/possum_project/tmp/v2_de_01_retrain_logs \
    --learning_rate=0.1,0.01,0.001 \
    --how_many_training_steps=10000,10000,10000 \
    --unknown_percentage=30.0 \
    --background_volume=0.5 \
    --model_architecture=deepear_v01



**To take a model and 'freeze' it so it can be used in realtime inferencing use:**

simple_audio.tensorflow.examples.speech_commands.freeze.py

**To run real-time inference use:**

simple_audio.tensorflow.examples.speech_realtime.inference_continuous.py


**Code Layout**

Data created by us is in /data /
Data collected from else where is in a separate file available here : TODO / 
Code to collect and collate data is in /datset_preparation /
Code to train networks is in /simple_audio   /
Code to create spectrograms is in / spectrograms 

**Dependencies**

tf-nightly (r1.5)
python 3.6.2

**iOS Setup**

Download correct tf-nightly version from here
https://pypi.python.org/pypi/tf-nightly

Active the conda environment you want then, then install the tf nightly file downloaded

(tf-simpleaudio) Dwanes-MacBook-Pro:tensorflow-master dsluis$ pip3 install ~/Downloads/tf_nightly-1.5.0.dev20171122-cp36-cp36m-macosx_10_11_x86_64.whl

To install pyaudio for realtime use
https://gist.github.com/jiaaro/9767512210a1d80a8a0d

Install portaudio using homebrew (or method of your choice)

brew install portaudio

vi ~/.pydistutils.cfg

[build_ext]
include_dirs=/usr/local/Cellar/portaudio/19.6.0/lib/
library_dirs=/usr/local/Cellar/portaudio/19.6.0/include/

Then in your virtualenv:

pip3 install pyaudio


**Windows Setup**

Look for the lastest tf-nightly here:
    https://ci.tensorflow.org/view/Nightly/job/nightly-win/M=windows,PY=36/
Download the whl file, something like:
    tf_nightly-1.5.0.dev20171115-cp36-cp36m-win_amd64.whl 
At the conda command prompt:
(C:\programs\Anaconda3) C:\Users\dwane\Documents>

Run:
pip install C:\Users\dwane\Downloads\tf_nightly-1.5.0.dev20171115-cp36-cp36m-win_amd64.whl

If you get a 
    tensorflow-1.4.0.dev0-cp36-cp36m-win_amd64.whl is not a supported wheel on this platform.
you need to check if you are running various versions of python. I was running 3.5, 3.6 and 2.x on the same machine 
from various directories. This went away after I deleted all but the latest anaconda3 version.    
    
If you get a:    
    parse() got an unexpected keyword argument 'transport_encoding'
Uninstall html5lib from conda with the following commands:
	conda remove html5lib
	conda install pip
	
**rasparian setup**
    To be completed
    
    
    