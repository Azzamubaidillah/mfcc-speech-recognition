import keras.models

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

def Keyword_Spotting_Service():

    # ensure that we only have 1 instance of KSS
    if _Keyword_Spotting_Service._instance is None:
        _Keyword_Spotting_Service._instance = _Keyword_Spotting_Service()
        _Keyword_Spotting_Service._model = keras.models.load_model(MODEL_PATH)
    return _Keyword_Spotting_Service._instance

