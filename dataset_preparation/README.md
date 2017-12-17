## Possum Dataset
***
### dataset_v2
* More possum data: 194 mono files
* More background noise data: added various additional noise files from the British Library Sound archive (rain, wind, hail, traffic, others)
* Better bird data: 2 sec chunks of warblr and freefield data based around amplitude peaks, then manually screened
* _nohash_ added to filenames for cats and dogs to ensure that chunks of individual source files are kept in the same train/validation/test splits
* Uploaded to SherlockML as dataset_v2

***
### dataset_v1
* Variant on v0 that I put together to deal with some problems in v0
* Bird data: reduced number of samples to reduce class imbalance; tried to improve quality by switching from the freefield1010 dataset to a mix of Kaggle and Warblr (still not sure how much better these are)
* Possum data: kept both channels (L & R) from the original stereo tracks to give a larger number of samples at the cost of them not being independent
* Not copied to OneDrive, but...
* On SherlockML: /projects/possum_project/dataset_v1 
* This will do for initial testing. Will be replaced with an improved dataset_v2, to include:
** More possum data (manually cut by Dwane)
** More background noise data
** Possibly better bird data if I can work out how to convert .flac -> .wav

***
### dataset_v0
* Created and copied to OneDrive 03-Dec-17
* .wav files | 2 seconds | mono | 16kHz | 16-bit
* wav_file_details.xlsx lists the files, file counts and formats

Summary of data folders and sources:

| Category |	Count | Source |
| ---------| -------| -------|
| bird | 5240 | Machine Listening Challenge freefield1010 dataset. Originally 10s, split into 2s |
| cat	| 414 | Kaggle. Originally different lengths, split into 2s |
| dog	| 235 | Kaggle. Originally different lengths, split into 2s |
| groundparrot	| 25 | Kaggle |
| possum	| 77 | Dwane. Manually split into ~2s chunks. Converted to mono based on left channel.|
| random	| 250 | Kaggle ESC50. Long recordings splits into 2s |
| speech	| 4086 | Google speech_commands. 1s |
| Total	| 10327 |  |

Problems with v0:
* **Class imbalance**: lots of samples for birds and speech, but very few for possums
* **Bird quality**: quality is mixed. Lots of samples labelled as birds don't contain birds. Lots of different types of birds are included, and they sound very different. Calls and songs are mixed up. I have lots of other bird datasets, but I don't think any of them are clearly better. Needs more investigation. If it comes to it I can generate a big list of candidate chunks, listen to them manually and delete the bad ones (I already wrote some code to automate this except for clicking an accept/reject button.)

Initial results of running the Google model on this dataset were poor.
* Accuracy stats were looking okay in general, but they were awful for possums
* Too many samples were being classified as birds, which became some sort of catch-all category because it's so large and has so much internal variation.
* I repeatedly reduced the number of samples in birds and speech to even out the classes. This helped a bit.


***
