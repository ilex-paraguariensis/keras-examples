from tensorflow import keras


def compile(
    model, lr: float = 1e-3, loss="binary_crossentropy", metrics=("accuracy")
):
    return model.compile(optimer=keras.optimizers.Adam(1e-3))


def fit(model, train_ds, epochs, val_ds):
    callbacks = [
        keras.callbacks.ModelCheckpoint("save_at_{epoch}.h5"),
    ]
    model.fit(
        train_ds,
        epochs=epochs,
        callbacks=callbacks,
        validation_data=val_ds,
    )
