import os
import random
import librosa
import pandas as pd
from tqdm import tqdm
import tensorflow as tf
import numpy as np
class_dim = 10
EPOCHS = 100
BATCH_SIZE = 32
init_model = None

'''创建训练数据,用于生成tensorflow'''
# 创建UrbanSound8K数据列表
def get_urbansound8k_list(path, urbansound8k_cvs_path):
    data_list = []
    data = pd.read_csv(urbansound8k_cvs_path)
    # 过滤掉长度少于3秒的音频
    valid_data = data[['slice_file_name', 'fold', 'classID', 'class']][data['end'] - data['start'] >= 3]
    valid_data['path'] = 'fold' + valid_data['fold'].astype('str') + '/' + valid_data['slice_file_name'].astype('str')
    for row in valid_data.itertuples():
        data_list.append([row.path, row.classID])

    f_train = open(os.path.join(path, 'train_list.txt'), 'w')
    f_test = open(os.path.join(path, 'test_list.txt'), 'w')

    for i, data in enumerate(data_list):
        sound_path = os.path.join('UrbanSound8K/', data[0])
        if i % 100 == 0:
            f_test.write('%s\t%d\n' % (sound_path, data[1]))
        else:
            f_train.write('%s\t%d\n' % (sound_path, data[1]))

    f_test.close()
    f_train.close()


'''生成TFRecord文件'''
# 获取浮点数组
def _float_feature(value):
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

# 获取整型数据
def _int64_feature(value):
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

# 把数据添加到TFRecord中
def data_example(data, label):
    feature = {
        'data': _float_feature(data),
        'label': _int64_feature(label),
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))

# 开始创建tfrecord数据
def create_data_tfrecord(data_list_path, save_path):
    with open(data_list_path, 'r') as f:
        data = f.readlines()
    with tf.io.TFRecordWriter(save_path) as writer:
        for d in tqdm(data):
            try:
                path, label = d.replace('\n', '').split('\t')
                wav, sr = librosa.load(path, sr=16000)
                intervals = librosa.effects.split(wav, top_db=20)
                wav_output = []
                wav_len = int(16000 * 2.04)
                for sliced in intervals:
                    wav_output.extend(wav[sliced[0]:sliced[1]])
                for i in range(5):
                    if len(wav_output) > wav_len:
                        l = len(wav_output) - wav_len
                        r = random.randint(0, l)
                        wav_output = wav_output[r:wav_len + r]
                    else:
                        wav_output.extend(np.zeros(shape=[wav_len - len(wav_output)], dtype=np.float32))
                    wav_output = np.array(wav_output)
                    # 转成梅尔频谱
                    ps = librosa.feature.melspectrogram(y=wav_output, sr=sr, hop_length=256).reshape(-1).tolist()
                    if len(ps) != 128 * 128: continue
                    tf_example = data_example(ps, int(label))
                    writer.write(tf_example.SerializeToString())
                    if len(wav_output) <= wav_len:
                        break
            except Exception as e:
                print(e)

'''读取TFRecord文件数据'''


def _parse_data_function(example):

    data_feature_description = {
        'data': tf.io.FixedLenFeature([16384], tf.float32),
        'label': tf.io.FixedLenFeature([], tf.int64),
    }
    return tf.io.parse_single_example(example, data_feature_description)


def train_reader_tfrecord(data_path, num_epochs, batch_size):
    raw_dataset = tf.data.TFRecordDataset(data_path)
    train_dataset = raw_dataset.map(_parse_data_function)
    train_dataset = train_dataset.shuffle(buffer_size=1000) \
        .repeat(count=num_epochs) \
        .batch(batch_size=batch_size) \
        .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return train_dataset


def test_reader_tfrecord(data_path, batch_size):
    raw_dataset = tf.data.TFRecordDataset(data_path)
    test_dataset = raw_dataset.map(_parse_data_function)
    test_dataset = test_dataset.batch(batch_size=batch_size)
    return test_dataset


if __name__ == '__main__':

    '''生成数据列表'''
    get_urbansound8k_list('UrbanSound8K/', 'UrbanSound8K.csv')

    '''生成tfrecord'''
    create_data_tfrecord('UrbanSound8K/train_list.txt', 'UrbanSound8K/train.tfrecord')
    create_data_tfrecord('UrbanSound8K/test_list.txt', 'UrbanSound8K/test.tfrecord')

    '''训练'''
    model = tf.keras.models.Sequential([
        tf.keras.applications.ResNet50V2(include_top=False, weights=None, input_shape=(128, None, 1)),
        tf.keras.layers.ActivityRegularization(l2=0.5),
        tf.keras.layers.Dropout(rate=0.5),
        tf.keras.layers.GlobalMaxPooling2D(),
        tf.keras.layers.Dense(units=class_dim, activation=tf.nn.softmax)
    ])

    model.summary()

    # 定义优化方法
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

    train_dataset = train_reader_tfrecord('UrbanSound8K/train.tfrecord', EPOCHS, batch_size=BATCH_SIZE)
    test_dataset = test_reader_tfrecord('UrbanSound8K/test.tfrecord', batch_size=BATCH_SIZE)

    if init_model:
        model.load_weights(init_model)

    for batch_id, data in enumerate(train_dataset):

        sounds = data['data'].numpy().reshape((-1, 128, 128, 1))
        labels = data['label']
        # 执行训练
        with tf.GradientTape() as tape:
            predictions = model(sounds)
            # 获取损失值
            train_loss = tf.keras.losses.sparse_categorical_crossentropy(labels, predictions)
            train_loss = tf.reduce_mean(train_loss)
            # 获取准确率
            train_accuracy = tf.keras.metrics.sparse_categorical_accuracy(labels, predictions)
            train_accuracy = np.sum(train_accuracy.numpy()) / len(train_accuracy.numpy())

        # 更新梯度
        gradients = tape.gradient(train_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        if batch_id % 20 == 0:
            print("Batch %d, Loss %f, Accuracy %f" % (batch_id, train_loss.numpy(), train_accuracy))

        if batch_id % 200 == 0 and batch_id != 0:
            test_losses = list()
            test_accuracies = list()
            for d in test_dataset:

                test_sounds = d['data'].numpy().reshape((-1, 128, 128, 1))
                test_labels = d['label']

                test_result = model(test_sounds)
                # 获取损失值
                test_loss = tf.keras.losses.sparse_categorical_crossentropy(test_labels, test_result)
                test_loss = tf.reduce_mean(test_loss)
                test_losses.append(test_loss)
                # 获取准确率
                test_accuracy = tf.keras.metrics.sparse_categorical_accuracy(test_labels, test_result)
                test_accuracy = np.sum(test_accuracy.numpy()) / len(test_accuracy.numpy())
                test_accuracies.append(test_accuracy)

            print('=================================================')
            print("Test, Loss %f, Accuracy %f" % (
                sum(test_losses) / len(test_losses), sum(test_accuracies) / len(test_accuracies)))
            print('=================================================')

            # 保存模型
            model.save(filepath='files/resnet50.h5')
            model.save_weights(filepath='files/model_weights.h5')

