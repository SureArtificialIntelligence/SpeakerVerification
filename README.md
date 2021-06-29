# SpeakerVerification
Speaker Verification Using Deep Learning

This repository is created based on https://github.com/HarryVolek/PyTorch_Speaker_Verification

Instead of using TIMIT dataset, we aim at training the network with bigger data, Librispeech/VoxCeleb, to achieve better robustness and generalisation effect.

For best consistent with the original code above, we create audio list in the same format as TIMIT audio list, by using create_pkls.py script following "Preprocessing" part in above link. The created pickle file (audio list) is "create_audio_path_vox_lbr/audio_path_other.pkl". Note that the file path depends on your local machine path.

Without need of changing anything else, you can then run "train_speech_embedder.py" (default settings can be used, they follows the "Training" part in above link.)

So only data are preprocessed in the same way as for TIMIT, the rest are not changed (if I remember correctly!)

To Do: upload trained network.
