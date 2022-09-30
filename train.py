
DATA_PATH = "data.json"
SAVED_MODEL_PATH = "model.h5"

LEARNING_RATE = 0.0001
EPOCHS = 40
BATCH_SIZE = 32

def main():
    # load train/validation/test data splits
    X_train, X_validation, X_test, y_train, y_validation, y_test = get_data_splits(DATA_PATH)

    # build the CNN model
    input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3]) # (# segments, # coefficients 13, 1 )
    model = build_model(input_shape, LEARNING_RATE)

    # train the model
    model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE,
              validation_data=(X_validation, y_validation))

    # evaluate the model
    test_error, test_accuracy = model.evaluate(X_test, y_test)
    print(f"Test error: {test_error}, test accuracy: {test_accuracy}")

    # save the model
    model.save(SAVED_MODEL_PATH)
