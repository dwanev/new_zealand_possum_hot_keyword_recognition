# GI23_GP


**To run real-time inference use:**

simple_audio.tensorflow.examples.speech_realtime.inference_continuous.py

**To train a model and save it to disk use:**

simple_audio.tensorflow.examples.speech_commands.train.py

**To take a model and 'freeze' it so it can be used in realtime inferencing use:**

simple_audio.tensorflow.examples.speech_commands.freeze.py


# Dependencies

tf-nightly 
python 3.6


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
	

    
    
    
    