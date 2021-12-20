# https://keras.io/api/callbacks/model_checkpoint/
def define_callbacks():
    save_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath='weights.{epoch:02d}-{val_acc:.2f}.hdf5',
        monitor='val_acc',
        save_best_only=True,
        verbose=1
    )
    return save_callback
