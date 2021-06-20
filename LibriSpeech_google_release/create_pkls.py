import csv
import os
import pickle as p
from os.path import join
from tqdm import tqdm

root_dir = '/nas/staff/data_work/Sure/LibriSpeech_GE2E_SV'
train_speakers, valid_speakers, test_speakers = [os.listdir(join(root_dir, split)) for split in ['train', 'valid', 'test']]
train_speakers_other, valid_speakers_other, test_speakers_other = [os.listdir(join(root_dir, split))
                                                                   for split in
                                                                   ['other_train', 'other_valid', 'other_test']]
# train_speakers.extend(train_speakers_other)
# valid_speakers.extend(valid_speakers_other)
# test_speakers.extend(test_speakers_other)

counter = 0
partitions = ['dev', 'train']
for partition in partitions:
    clean_paths, enrol_paths, inter_paths = [], [], []
    with open('/home/user/on_gpu/Speaeker_Verification/LibriSpeech_google_release/{}_tuples.csv'.format(partition)) as csv_file:
        contents = csv.reader(csv_file, delimiter=',')
        next(contents)
        for line in tqdm(contents):
            clean, enrollment, interference = line
            clean_speaker = clean.split('-')[0]
            enrol_speaker = enrollment.split('-')[0]
            inter_speaker = interference.split('-')[0]
            counter += 1
            assert clean_speaker == enrol_speaker

            if clean_speaker in train_speakers:
                clean_split = 'train'
            elif clean_speaker in valid_speakers:
                clean_split = 'valid'
            elif clean_speaker in test_speakers:
                clean_split = 'test'
            elif clean_speaker in train_speakers_other:
                clean_split = 'other_train'
            elif clean_speaker in valid_speakers_other:
                clean_split = 'other_valid'
            elif clean_speaker in test_speakers_other:
                clean_split = 'other_test'
            else:
                print('Did not find {}'.format(clean_speaker))

            if inter_speaker in train_speakers:
                inter_split = 'train'
            elif inter_speaker in valid_speakers:
                inter_split = 'valid'
            elif inter_speaker in test_speakers:
                inter_split = 'test'
            elif inter_speaker in train_speakers_other:
                inter_split = 'other_train'
            elif inter_speaker in valid_speakers_other:
                inter_split = 'other_valid'
            elif inter_speaker in test_speakers_other:
                inter_split = 'other_test'
            else:
                print('Did not find {}'.format(inter_speaker))

            clean_path = join(root_dir, clean_split, clean_speaker, clean+'.wav')
            enrol_path = join(root_dir, clean_split, clean_speaker, enrollment+'.wav')
            inter_path = join(root_dir, inter_split, inter_speaker, interference+'.wav')
            if os.path.exists(clean_path) and os.path.exists(enrol_path) and os.path.exists(inter_path):
                clean_paths.append([clean_path])
                enrol_paths.append([enrol_path])
                inter_paths.append([inter_path])
            else:
                print()
    print('# of samples: {}/{}'.format(len(clean_paths), counter))

    clean_pkl = '/home/user/on_gpu/Speaeker_Verification/LibriSpeech_google_release/{}_clean.pkl'.format(partition)
    enrol_pkl = '/home/user/on_gpu/Speaeker_Verification/LibriSpeech_google_release/{}_enrolment.pkl'.format(partition)
    inter_pkl = '/home/user/on_gpu/Speaeker_Verification/LibriSpeech_google_release/{}_interference.pkl'.format(partition)

    for pkl, paths in zip([clean_pkl, enrol_pkl, inter_pkl], [clean_paths, enrol_paths, inter_paths]):
        with open(pkl, 'wb') as pkl_file:
            p.dump(paths, pkl_file)
