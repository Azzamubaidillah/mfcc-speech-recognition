import librosa
import os
import json

DATASET_PATH = "dataset"
JSON_PATH = "data.json"
SAMPLES_TO_CONSIDER = 22050 # 1 sec worth of sound

def prepare_dataset(dataset_path, json_path, n_mfcc=13, hop_length=512, n_fft=2048):

    # data dictionary
    data = {
        "mappings": [],
        "labels": [],
        "MFCCs": [],
        "files": []
    }

    # loop thorugh all the sub-dirs
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):

        # ensure not at the root level
        if dirpath is not dataset_path:

            # update mappings
            category = dirpath.split("/")[-1] # dataset/down -> [dataset, down]
            data["mappings"].append(category)

            # loop through all the filenames and extract MFCCs
            for f in filenames:

                # get file path
                file_path = os.path.join(dirpath, f)

                # load audio file
                signal, sr = librosa.load(file_path)

                # ensure the audio file is at least 1 sec
                if len(signal) >= SAMPLES_TO_CONSIDER:
                    pass
