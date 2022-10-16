import keras.models
import numpy as np

MODEL_PATH = "model.h5"

class _Keyword_Spotting_Service:

    model = None
    _mappings: [
        "up",
        "right",
        "yes",
        "off",
        "no",
        "on",
        "stop",
        "left",
        "down"
    ]

    _istance = None

    def predict(self, file_path):

        # extract MFCCs
        MFCCs = self.preprocess(file_path) # (# segments, # coefficient)


        # convert 2d MFCCs array into 4d array -> (# samples, # segments, # coefficients, # channels)
        MFCCs = MFCCs[np.newaxis, ..., np.newaxis]

        # make prediction
        predictions = self.model.predict(MFCCs)
        predicted_index = np.argmax(predictions)
        predicted_keyword = self._mappings[predicted_index]

        return predicted_keyword

    def preprocess(self, file_path):
        pass


def Keyword_Spotting_Service():

    # ensure that we only have 1 instance of KSS
    if _Keyword_Spotting_Service._instance is None:
        _Keyword_Spotting_Service._instance = _Keyword_Spotting_Service()
        _Keyword_Spotting_Service._model = keras.models.load_model(MODEL_PATH)
    return _Keyword_Spotting_Service._instance

