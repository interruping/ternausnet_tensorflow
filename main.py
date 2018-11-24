import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm, trange
import argparse

from model import Unet11
from utils import get_img_resizer, RLenc
from metrics import get_iou_vector

if __name__ == '__main__':

    args = argparse.ArgumentParser()
    args.add_argument('--mode', type=str, default='train')
    args.add_argument('--epochs', type=int, default=500)
    args.add_argument('--batch_size', type=int, default=20)
    args.add_argument('--checkpoint', type=str, default=None)
    config = args.parse_args()

    # load data
    train_df = pd.read_csv("./input/train.csv", index_col="id", usecols=[0])
    depths_df = pd.read_csv("./input/depths.csv", index_col="id")
    train_df = train_df.join(depths_df)
    test_df = depths_df[~depths_df.index.isin(train_df.index)]

    train_df["images"] = [np.array(load_img("./input/train/images/{}.png".format(idx), grayscale=True)) / 255 for idx in tqdm(train_df.index, desc='load train images...')]
    train_df["masks"] = [np.array(load_img("./input/train/masks/{}.png".format(idx), grayscale=True)) / 255 for idx in tqdm(train_df.index, desc='load mask images...')]
    train_df["coverage"] = train_df.masks.map(np.sum) / pow(101, 2)

    def cov_to_class(val):    
        for i in range(0, 11):
            if val * 10 <= i :
                return i
            
    train_df["coverage_class"] = train_df.coverage.map(cov_to_class)
    
    # data split for train and valid 
    img_size_target = 128
    upsample = get_img_resizer(img_size_target)

    x_train, x_valid, y_train, y_valid = train_test_split(
    np.array(train_df.images.map(upsample).tolist()).reshape(-1, img_size_target, img_size_target, 1), 
    np.array(train_df.masks.map(upsample).tolist()).reshape(-1, img_size_target, img_size_target, 1), 
    test_size=0.2, stratify=train_df.coverage_class, random_state=1337)

    # data argumentation
    x_train = np.append(x_train, [np.fliplr(x) for x in x_train], axis=0)
    y_train = np.append(y_train, [np.fliplr(x) for x in y_train], axis=0)
    
    # extend for feed 3 channel input
    x_train = np.repeat(x_train,3,axis=3)
    x_valid = np.repeat(x_valid,3,axis=3)

    EPOCHS = config.epochs
    BATCH_SIZE = config.batch_size

    with tf.Session() as sess:
        # build model
        model = Unet11(sess, img_size_target)

        # model initialize
        if config.checkpoint:
            model.restore(config.checkpoint)
        else:
            model.restore_latest()

        # train mode
        if config.mode == 'train':
            epochs_tqdm = tqdm(range(EPOCHS))
            # validation hook that when called when training.
            def validation_hook():
                valid_loss, valid_iou = model.validate(x_valid, y_valid, BATCH_SIZE)
                epochs_tqdm.set_postfix(valid_loss=valid_loss, valid_iou=valid_iou)        
            model.set_train_hook(every_n=100, hook=validation_hook)
            # training
            for epoch in epochs_tqdm:
                epochs_tqdm.set_description('epoch {}'.format(epoch))

                mean_loss, mean_iou = model.train(x_train, y_train, BATCH_SIZE, 0.9, 1e-3)
                model.save()
                epochs_tqdm.set_postfix(mean_loss=mean_loss, mean_iou=mean_iou)
        # test mode
        elif config.mode == 'test':
            # calcurate for find best threshold
            valid_logits = model.predict(x_valid, BATCH_SIZE)
            thresholds = np.linspace(0, 1, 50)
            ious = np.array([get_iou_vector(y_valid, np.int32(valid_logits > threshold)) for threshold in tqdm(thresholds, desc='find best thresold...')])
            threshold_best_index = np.argmax(ious[9:-10]) + 9
            iou_best = ious[threshold_best_index]
            threshold_best = thresholds[threshold_best_index]

            print('best_threshold:', threshold_best)

            # model test 
            x_test = np.array([upsample(np.array(load_img("./input/images/{}.png".format(idx), grayscale=True))) / 255 for idx in tqdm(test_df.index, desc='load test images...')]).reshape(-1, img_size_target, img_size_target, 1)
            x_test = np.repeat(x_test,3,axis=3)

            downsample = get_img_resizer(101)

            test_logits = model.predict(x_test, BATCH_SIZE)
            pred_dict = {idx: RLenc(np.round(downsample(test_logits[i]) > threshold_best)) for i, idx in enumerate(tqdm(test_df.index.values, desc='post processing...'))}
            sub = pd.DataFrame.from_dict(pred_dict,orient='index')
            sub.index.names = ['id']
            sub.columns = ['rle_mask']
            sub.to_csv('submission.csv')

            print('test complete with create submission.csv')
        else:
            raise ValueError('Invalid mode. Please select mode between train and test.')