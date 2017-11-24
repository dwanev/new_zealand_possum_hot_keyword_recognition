
This project requires tf-nightly and python 3.6

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