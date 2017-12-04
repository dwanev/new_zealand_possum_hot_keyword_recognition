## Possum Dataset

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
