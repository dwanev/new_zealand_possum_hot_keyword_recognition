run("C:\Users\richa\Audio\T1-COMPGI23_DL_Group\simple_audio\tensorflow\examples\speech_commands\train.py" -- \
--data_url=
--data_dir=C:/Users/richa/Audio/dataset_v0/
--wanted_words=possum,cat,dog,bird
--clip_duration_ms=2000
--train_dir=/tmp/v0_commands_train
--summaries_dir=/tmp/v0_retrain_logs

freeze
--wanted_words=possum,cat,dog,bird
--clip_duration_ms=2000
--output_file=/tmp/my_frozen_graph_v0.pb
--start_checkpoint=/tmp/v0_commands_train/conv.ckpt-110

python tensorflow/examples/speech_commands/label_wav.py
--graph=/tmp/my_frozen_graph_v0.pb
--labels=/tmp/v0_commands_train/conv_labels.txt
#--wav=C:/Users/richa/Audio/dataset_v0/bird/518_chunk_0.wav


for e in tf.train.summary_iterator("C:\\tmp\\v1_retrain_logs\\validation\\events.out.tfevents.1512466763.DESKTOP-PJHTJ6U"):
    for v in e.summary.value:
        if v.tag == 'loss' or v.tag == 'accuracy':
            print(v.simple_value)