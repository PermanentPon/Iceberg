"""
Baysian hyperparameter optimization [https://github.com/fmfn/BayesianOptimization]
based on Vladimir Iglovikov work
"""

import pandas as pd
import xgboost as xgb
from bayes_opt import BayesianOptimization
from scipy.stats import kurtosis, skew
from scipy.ndimage import laplace, sobel
import numpy as np
from multiprocessing import Pool
from tqdm import tqdm
import gc
from itertools import combinations
from math import isnan

def read_json(file='', loc='../input/'):
    df = pd.read_json('{}{}'.format(loc, file))
    df['inc_angle'] = df['inc_angle'].replace('na', -1).astype(float)
    # print(df['inc_angle'].value_counts())

    band1 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in df["band_1"]])
    band2 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in df["band_2"]])
    df = df.drop(['band_1', 'band_2'], axis=1)

    bands = np.stack((band1, band2, 0.5 * (band1 + band2)), axis=-1)
    del band1, band2

    return df, bands

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
    data = np.concatenate([data, df['inc_angle'].values[:, np.newaxis]], axis=-1);
    gc.collect()

    print(data.shape)
    return data


def xgb_evaluate(min_child_weight,
                 colsample_bytree,
                 colsample_bylevel,
                 max_delta_step,
                 max_depth,
                 subsample,
                 gamma,
                 alpha):
    print("xgb_evaluate starting...")

    params['min_child_weight'] = int(min_child_weight)
    params['cosample_bytree'] = max(min(colsample_bytree, 1), 0)
    params['colsample_bylevel'] = max(min(colsample_bylevel, 1), 0)
    params['max_delta_step'] = int(max_delta_step)
    params['max_depth'] = int(max_depth)
    params['subsample'] = max(min(subsample, 1), 0)
    params['gamma'] = max(gamma, 0)
    params['alpha'] = max(alpha, 0)

    params['scale_pos_weight'] = 1.0

    cv_result = xgb.cv(params, xgtrain, num_boost_round=num_rounds, nfold=5,
             seed=random_state,
             callbacks=[xgb.callback.early_stop(50)])

    return -cv_result['test-logloss-mean'].values[-1]


def prepare_data():

    np.random.seed(104)
    target = 'is_iceberg'

    # Load data
    train, train_bands = read_json(file='train.json', loc='../data/processed/')
    test, test_bands = read_json(file='test.json', loc='../data/processed/')

    train_X = process(df=train, bands=train_bands)
    train_y = train[target].values

    test_X = process(df=test, bands=test_bands)

    xgb_train = xgb.DMatrix(train_X, train_y)
    xgb_valid = xgb.DMatrix(test_X)

    return xgb_train


if __name__ == '__main__':
    xgtrain = prepare_data()

    num_rounds = 3000
    random_state = 2016
    num_iter = 25
    init_points = 5
    watchlist = [(xgtrain, 'train')]
    params = {
        'eta': 0.03,
        'silent': 1,
        'eval_metric': 'logloss',
        'objective': 'binary:logistic',
        'verbose_eval': True,
        'seed': random_state
    }

    xgbBO = BayesianOptimization(xgb_evaluate, {'min_child_weight': (1, 20),
                                                'colsample_bytree': (0.6, 0.95),
                                                'colsample_bylevel': (0.6, 0.95),
                                                'max_delta_step': (1, 5),
                                                'max_depth': (2, 7),
                                                'subsample': (0.7, 0.95),
                                                'gamma': (0, 10),
                                                'alpha': (0, 10),
                                                })

    xgbBO.maximize(init_points=init_points, n_iter=num_iter)
