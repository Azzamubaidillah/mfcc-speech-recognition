
DATA_PATH = "data.json"

LEARNING_RATE = 0.0001

def main():
    # load train/validation/test data splits
    X_train, X_validation, X_test, y_train, y_validation, y_test = get_data_splits(DATA_PATH)

    # build the CNN model
    input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3]) # (# segments, # coefficients 13, 1 )
    model = build_model(input_shape, LEARNING_RATE)

    # train the model


    # evaluate the model


    # save the model
