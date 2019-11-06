import os

from keras.optimizers import Adam
import keras.layers as KL
import keras.backend as K
from keras.models import Model
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, TensorBoard

from data_generator import MixedGenerator


def accuracy(y_true, y_pred):
    '''
    y_true : [None, 17]
    y_pred : [None, 17]
    '''
    y_true = K.argmax(y_true, axis=-1)
    y_pred = K.argmax(y_pred, axis=-1)
    acc = K.equal(y_true, y_pred)    # [None, 1]
    acc = K.cast(acc, K.floatx())
    acc = K.mean(acc)

    return acc


# 构建模型
def build(start_units=64):
    inputs_sen1 = KL.Input([32, 32, 8], name='in_sen1')
    inputs_sen2 = KL.Input([32, 32, 10], name='in_sen2')

    x1 = KL.Conv2D(start_units, (3, 3), padding='SAME')(inputs_sen1)
    x1 = KL.Activation('relu')(x1)
    x1 = KL.Conv2D(start_units, (3, 3), padding='SAME')(x1)
    x1 = KL.Activation('relu')(x1)
    x1 = KL.Dropout(0.1)(x1)
    x1 = KL.MaxPooling2D(padding='SAME')(x1)

    x2 = KL.Conv2D(start_units, (3, 3), padding='SAME')(inputs_sen2)
    x2 = KL.Activation('relu')(x2)
    x2 = KL.Conv2D(start_units, (3, 3), padding='SAME')(x2)
    x2 = KL.Activation('relu')(x2)
    x2 = KL.Dropout(0.1)(x2)
    x2 = KL.MaxPooling2D(padding='SAME')(x2)

    x = KL.Concatenate()([x1, x2])
    x = KL.Conv2D(start_units * 2, (3, 3), padding='SAME')(x)
    x = KL.Activation('relu')(x)
    x = KL.Dropout(0.1)(x)
    x = KL.Conv2D(start_units * 2, (3, 3), padding='SAME')(x)
    x = KL.Activation('relu')(x)
    x = KL.Dropout(0.1)(x)
    x = KL.MaxPooling2D(padding='SAME')(x)

    x = KL.Conv2D(start_units * 4, (3, 3), padding='SAME')(x)
    x = KL.Activation('relu')(x)
    x = KL.Dropout(0.1)(x)
    x = KL.Conv2D(start_units * 4, (3, 3), padding='SAME')(x)
    x = KL.Activation('relu')(x)
    x = KL.Dropout(0.1)(x)
    x = KL.Conv2D(start_units * 4, (3, 3), padding='SAME')(x)
    x = KL.Activation('relu')(x)
    x = KL.MaxPooling2D(padding='SAME')(x)

    x = KL.Flatten()(x)
    x = KL.Dropout(0.1)(x)
    x = KL.Dense(start_units * 8, activation='relu')(x)
    output = KL.Dense(17, activation='softmax', name='prediction')(x)

    train_model = Model(inputs=[inputs_sen1, inputs_sen2], outputs=output)

    return train_model


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    train_model = build()
    train_model.summary()

    train_model.compile(loss='categorical_crossentropy',
                        optimizer=Adam(lr=1e-4, clipnorm=0.001),
                        metrics=[accuracy])

    # 回调函数
    model_checkpoint = ModelCheckpoint("./snapshots/tsnet-9/tsnet_best.h5",
                                       verbose=1,
                                       save_best_only=True,
                                       monitor='val_accuracy',
                                       mode='max')
    reduce_lr = ReduceLROnPlateau(monitor='val_accuracy',
                                  mode='max',
                                  factor=0.1,
                                  patience=3,
                                  min_lr=1e-07,
                                  verbose=1)
    tensor_board = TensorBoard(log_dir="./snapshots/tsnet-9", batch_size=8)

    print('开始训练。。。')
    data_train = MixedGenerator('/home/lmzwhu/programs/DATAS/LCZ/', datatype='train', batch_size=16, split=0.1)
    data_val = MixedGenerator('/home/lmzwhu/programs/DATAS/LCZ/', datatype='val', batch_size=16, split=0.1)

    # train_model.load_weights('./snapshots/tsnet-9/tsnet_best.h5')
    train_model.fit_generator(data_train,
                              epochs=20,
                              steps_per_epoch=20000,
                              callbacks=[model_checkpoint, reduce_lr, tensor_board],
                              validation_data=data_val,
                              validation_steps=2000)
