"""
Train LightGBM and xgboost to diversify results from NNs.
based on https://www.kaggle.com/the1owl/natural-growth-patterns-fractals-of-nature

@author: PermanentPon
"""

from multiprocessing import Pool
from tqdm import tqdm
import gc
import random
import numpy as np
import pandas as pd
import datetime as dt
from math import isnan
from itertools import combinations
from scipy.stats import kurtosis, skew
from scipy.ndimage import laplace, sobel
from sklearn.model_selection import StratifiedShuffleSplit
import xgboost as xgb
import lightgbm as lgb

def read_json(file='', loc='../input/'):
    """
    Initial data loading
    :param file: name of file to load
    :param loc: folder path
    :return: data to train or test as ndarray
    """
    df = pd.read_json('{}{}'.format(loc, file))
    df['inc_angle'] = df['inc_angle'].replace('na', -1).astype(float)

    band1 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in df["band_1"]])
    band2 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in df["band_2"]])
    df = df.drop(['band_1', 'band_2'], axis=1)

    bands = np.stack((band1, band2, 0.5 * (band1 + band2)), axis=-1)
    del band1, band2

    return df, bands

def run_lgb(params={}, lgb_train=None, lgb_valid=None, lgb_test=None, test_ids=None, nr_round=2000, min_round=100,
            file=''):
    """
    Run LightGBM training
    :return: predictions as ndarray
    """
    print('\nLightGBM: {}'.format(params['boosting']))
    model2 = lgb.train(params,
                       lgb_train,
                       nr_round,
                       lgb_valid,
                       verbose_eval=50, early_stopping_rounds=min_round)

    pred = model2.predict(lgb_test, num_iteration=model2.best_iteration)
    subm = pd.DataFrame({'id': test_ids, 'is_iceberg': pred})
    subm.to_csv(file, index=False, float_format='%.6f')
    df = pd.DataFrame({'feature': model2.feature_name(), 'importances': model2.feature_importance()})

    return pred, df, model2.best_score['valid_0']['binary_logloss']


# forked from
# https://www.kaggle.com/the1owl/planet-understanding-the-amazon-from-space/natural-growth-patterns-fractals-of-nature/notebook
def img_to_stats(paths):
    img_id, img = paths[0], paths[1]

    # ignored error
    np.seterr(divide='ignore', invalid='ignore')

    bins = 20
    scl_min, scl_max = -50, 50
    opt_poly = True
    # opt_poly = False

    try:
        st = []
        st_interv = []
        hist_interv = []
        for i in range(img.shape[2]):
            img_sub = np.squeeze(img[:, :, i])

            # median, max and min
            sub_st = []
            sub_st += [np.mean(img_sub), np.std(img_sub), np.max(img_sub), np.median(img_sub), np.min(img_sub)]
            sub_st += [(sub_st[2] - sub_st[3]), (sub_st[2] - sub_st[4]), (sub_st[3] - sub_st[4])]
            sub_st += [(sub_st[-3] / sub_st[1]), (sub_st[-2] / sub_st[1]),
                       (sub_st[-1] / sub_st[1])]  # normalized by stdev
            st += sub_st
            # Laplacian, Sobel, kurtosis and skewness
            st_trans = []
            st_trans += [laplace(img_sub, mode='reflect', cval=0.0).ravel().var()]  # blurr
            sobel0 = sobel(img_sub, axis=0, mode='reflect', cval=0.0).ravel().var()
            sobel1 = sobel(img_sub, axis=1, mode='reflect', cval=0.0).ravel().var()
            st_trans += [sobel0, sobel1]
            st_trans += [kurtosis(img_sub.ravel()), skew(img_sub.ravel())]

            if opt_poly:
                st_interv.append(sub_st)
                #
                st += [x * y for x, y in combinations(st_trans, 2)]
                st += [x + y for x, y in combinations(st_trans, 2)]
                st += [x - y for x, y in combinations(st_trans, 2)]

                # hist
            # hist = list(cv2.calcHist([img], [i], None, [bins], [0., 1.]).flatten())
            hist = list(np.histogram(img_sub, bins=bins, range=(scl_min, scl_max))[0])
            hist_interv.append(hist)
            st += hist
            st += [hist.index(max(hist))]  # only the smallest index w/ max value would be incl
            st += [np.std(hist), np.max(hist), np.median(hist), (np.max(hist) - np.median(hist))]

        if opt_poly:
            for x, y in combinations(st_interv, 2):
                st += [float(x[j]) * float(y[j]) for j in range(len(st_interv[0]))]

            for x, y in combinations(hist_interv, 2):
                hist_diff = [x[j] * y[j] for j in range(len(hist_interv[0]))]
                st += [hist_diff.index(max(hist_diff))]  # only the smallest index w/ max value would be incl
                st += [np.std(hist_diff), np.max(hist_diff), np.median(hist_diff),
                       (np.max(hist_diff) - np.median(hist_diff))]

        # correction
        nan = -999
        for i in range(len(st)):
            if isnan(st[i]) == True:
                st[i] = nan

    except:
        print('except: ')

    return [img_id, st]


def extract_img_stats(paths):
    imf_d = {}
    p = Pool(8)  # (cpu_count())
    ret = p.map(img_to_stats, paths)
    for i in tqdm(range(len(ret)), miniters=100):
        imf_d[ret[i][0]] = ret[i][1]

    ret = []
    fdata = [imf_d[i] for i, j in paths]
    return np.array(fdata, dtype=np.float32)


def process(df, bands):
    data = extract_img_stats([(k, v) for k, v in zip(df['id'].tolist(), bands)]);
    gc.collect()
    #new_feats = trainer.get_leak_features(df['inc_angle'])
    data = np.concatenate([data, df['inc_angle'].values[:, np.newaxis]], axis=-1);
    #data = np.concatenate([data, new_feats], axis=1);
    gc.collect()

    print(data.shape)
    return data


def save_blend(preds, score, loc='./'):
    target = 'is_iceberg'

    w_total = 0.0
    blend = None
    df_corr = None
    print('\nBlending...')
    for k, v in preds.items():
        if blend is None:
            blend = pd.read_csv('{0}/{1}'.format(loc, k))
            print('load: {0}, w={1}'.format(k, v))

            df_corr = pd.DataFrame({'id': blend['id'].tolist()})
            df_corr[k[16:-4]] = blend[target]

            w_total += v
            blend[target] = blend[target] * v

        else:
            preds_tmp = pd.read_csv('{0}/{1}'.format(loc, k))
            preds_tmp = blend[['id']].merge(preds_tmp, how='left', on='id')
            #print('load: {0}, w={1}'.format(k, v))
            df_corr[k[16:-4]] = preds_tmp[target]

            w_total += v
            blend[target] += preds_tmp[target] * v
            del preds_tmp

    #print('\n{}'.format(df_corr.corr()), flush=True)
    # write submission
    blend[target] = blend[target] / w_total
    #print('\nPreview: \n{}'.format(blend.head()), flush=True)
    blend.to_csv('../results/trees/{:.5f}trees_blend{:03d}_{}.csv'.format(score, len(preds), tmp), index=False, float_format='%.6f')


if __name__ == '__main__':

    np.random.seed(104)
    target = 'is_iceberg'

    # Load data
    train, train_bands = read_json(file='train.json', loc='../data/processed/')
    test, test_bands = read_json(file='test.json', loc='../data/processed/')

    train_X = process(df=train, bands=train_bands)
    train_y = train[target].values

    test_X = process(df=test, bands=test_bands)

    # results
    freq = pd.DataFrame()
    subms = []

    # training
    test_ratio = 0.2
    nr_runs = 3


    for i in range(1, 50):
        split_seed = random.randint(1, 1000)
        kf = StratifiedShuffleSplit(n_splits=nr_runs, test_size=test_ratio, train_size=None, random_state=split_seed)
        cv_log_loss = []
        for r, (train_index, test_index) in enumerate(kf.split(train_X, train_y)):
            print('\nround {:04d} of {:04d}, seed={}'.format(r + 1, nr_runs, split_seed))

            tmp = dt.datetime.now().strftime("%Y-%m-%d-%H-%M")

            x1, x2 = train_X[train_index], train_X[test_index]
            y1, y2 = train_y[train_index], train_y[test_index]
            # x1, x2, y1, y2 = train_test_split(train_X, train_y, test_size=test_ratio, random_state=split_seed + r)
            print('splitted: {0}, {1}'.format(x1.shape, x2.shape), flush=True)
            test_X_dup = test_X.copy()

            # XGB
            xgb_train = xgb.DMatrix(x1, y1)
            xgb_valid = xgb.DMatrix(x2, y2)
            #
            watchlist = [(xgb_train, 'train'), (xgb_valid, 'valid')]
            params = {'objective': 'binary:logistic', 'seed': 99, 'silent': True}
            params['eta'] = 0.04
            params['max_depth'] = 6
            params['subsample'] = 0.9
            params['eval_metric'] = 'logloss'
            params['colsample_bytree'] = 0.9
            params['colsample_bylevel'] = 0.85
            params['max_delta_step'] = 3
            # params['gamma'] = 5.0
            # params['labmda'] = 1
            params['scale_pos_weight'] = 1.0
            params['seed'] = split_seed + r
            nr_round = 2000
            min_round = 100

            model1 = xgb.train(params,
                               xgb_train,
                               nr_round,
                               watchlist,
                               verbose_eval=50,
                               early_stopping_rounds=min_round)
            cv_log_loss.append(model1.best_score)
            pred_xgb = model1.predict(xgb.DMatrix(test_X_dup), ntree_limit=model1.best_ntree_limit + 45)

            #
            file = '../results/temp/subm_{}_xgb_{:02d}.csv'.format(tmp, r + 1)
            subm = pd.DataFrame({'id': test['id'].values, target: pred_xgb})
            subm.to_csv(file, index=False, float_format='%.6f')
            subms.append(file)

            ##LightGBM
            lgb_train = lgb.Dataset(x1, label=y1, free_raw_data=False)
            lgb_valid = lgb.Dataset(x2, label=y2, reference=lgb_train, free_raw_data=False)
            # gbdt
            params = {'learning_rate': 0.02, 'max_depth': 4, 'boosting': 'gbdt', 'objective': 'binary',
                      'is_training_metric': False, 'seed': 99}
            params['boosting'] = 'gbdt'
            params['metric'] = 'binary_logloss'
            params['learning_rate'] = 0.02
            params['max_depth'] = 7
            params['num_leaves'] = 16  # higher number of leaves
            params['feature_fraction'] = 0.8  # Controls overfit
            params['bagging_fraction'] = 0.95
            params['bagging_freq'] = 3
            params['seed'] = split_seed + r
            #
            params['verbose'] = -1

            file = '../results/temp/subm_{}_lgb_{}_{:02d}.csv'.format(tmp, params['boosting'], r + 1)
            subms.append(file)

            pred, f_tmp, best_score = run_lgb(params=params,
                                  lgb_train=lgb_train,
                                  lgb_valid=lgb_valid,
                                  lgb_test=test_X_dup,
                                  test_ids=test['id'].values,
                                  nr_round=nr_round,
                                  min_round=min_round,
                                  file=file)
            cv_log_loss.append(best_score)
            ##LightGBM
            # dart
            params = {'learning_rate': 0.01, 'max_depth': 3, 'boosting': 'gbdt', 'objective': 'binary',
                      'is_training_metric': False, 'seed': 99}
            params['boosting'] = 'dart'
            params['metric'] = 'binary_logloss'
            params['learning_rate'] = 0.02
            params['max_depth'] = 7
            params['num_leaves'] = 16  # higher number of leaves
            params['feature_fraction'] = 0.85  # Controls overfit
            params['bagging_fraction'] = 0.9
            params['bagging_freq'] = 3
            params['seed'] = split_seed + r
            # dart
            params['drop_rate'] = 0.1
            params['skip_drop'] = 0.5
            params['max_drop'] = 10
            params['verbose'] = -1

            file = '../results/temp/subm_{}_lgb_{}_{:02d}.csv'.format(tmp, params['boosting'], r + 1)
            subms.append(file)

            pred, f_tmp, best_score = run_lgb(params=params,
                                  lgb_train=lgb_train,
                                  lgb_valid=lgb_valid,
                                  lgb_test=test_X_dup,
                                  test_ids=test['id'].values,
                                  nr_round=nr_round,
                                  min_round=min_round,
                                  file=file)
        cv_log_loss.append(best_score)
        # blending
        preds = {k: 1.0 for k in subms}
        score = sum(cv_log_loss) / len(cv_log_loss)
        save_blend(preds=preds, score = score)

        print("Average best_score = {}".format(score))
