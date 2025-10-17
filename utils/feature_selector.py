from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
from boruta import BorutaPy
import numpy as np
# pd.set_option('display.max_columns', None)


def principalca(df, n=8):
    """
    Performs Principal Component Analysis on the DataFrame.
    :param df: The DataFrame to perform PCA on.
    :param n: Number of components to return.
    :return: The DataFrame with the principal components.
    """
    df = df.fillna(0)
    x = df.drop('Close',axis=1).values
    sc = StandardScaler()
    scaled_x = sc.fit_transform(x)
    del x, sc, df

    pca = PCA(n_components=n)
    principal_components = pca.fit_transform(scaled_x)
    principal_df = pd.DataFrame(data=principal_components,columns=[f'pc{i+1}' for i in range(np.shape(principal_components)[1])])

    print('The principal components: ')
    print(principal_df.head(10))
    print(f'The explained variance from {n} components is {pca.explained_variance_ratio_}')
    del pca

    return principal_df


def Boruta_py(df, n_estimators='auto', random_state=69420, max_depth=5):
    """
    Performs Boruta feature selection on the DataFrame.
    :param df: The DataFrame to perform Boruta on.
    :param n_estimators: The number of estimators to use. (use auto to allow Boruta to decide)
    :param random_state: The random state.
    :param max_depth: The maximum depth of the Random Forest Regressor.
    :return: The DataFrame with the selected features.
    """
    np.int = np.int32
    np.float = np.float64
    np.bool = np.bool_
    df = df.fillna(0)
    x = df.drop('Close', axis=1).values
    y = df['Close'].values
    del df
    rf = RandomForestRegressor(n_jobs=-1, max_depth=max_depth)
    feature_selector = BorutaPy(rf, n_estimators=n_estimators, verbose=1, random_state=random_state)
    feature_selector.fit(x,y)
    del rf, y
    print(feature_selector.support_)
    print(feature_selector.ranking_)
    return1 = feature_selector.transform(x)
    ranking = feature_selector.ranking_
    return (return1, ranking)


# def get_importances(X, y):
#     rf = RandomForestRegressor(max_depth=20)
#     rf.fit(X,y)
#     importances = {feature_name: f_importance for feature_name, f_importance in zip(X.columns, rf.feature_importances_)}
#     only_shadow_feat_importance = {key:value for key,value in importances.items() if "shadow" in key}
#     highest_shadow_feature = list(dict(sorted(only_shadow_feat_importance.items(), key=lambda item: item[1], reverse=True)).values())[0]
#     selected_features = [key for key, value in importances.items() if value > highest_shadow_feature]
#     return selected_features
#
#
# def get_tail_items(pmf):
#     total = 0
#     for i, x in enumerate(pmf):
#         total += x
#         if total > 0.05:
#             break
#     return i
#
#
# def choose_features(feature_hits, thresh, trials=50):
#     green_zone = trials-thresh
#     blue_upper = green_zone
#     blue_lower = thresh
#     green = [key for key, value in feature_hits.items() if value >= green_zone]
#     blue = [key for key, value in feature_hits.items() if value >= blue_lower and value < blue_upper]
#     return green, blue
#
#
# def boruta(df,trials=50):
#     x = df.drop('Close', axis=1)
#     y = df['Close']
#
#     for col in x.columns:
#         x['shadow_col_{col}'] = x[col].sample(frac=1).reset_index(drop=True)
#
#     feature_hits = {i : 0 for i in df.columns}
#
#     for _ in (range(trials)):
#         imp_features = get_importances(x,y)
#         for key, _ in feature_hits.items():
#             if key in imp_features:
#                 feature_hits[key] += 1
#
#     print(feature_hits)
#     pmf = [sp.stats.binom.pmf(x, trials, 0.5) for x in range(trials+1)]
#     thresh = get_tail_items(pmf)
#     green, blue = choose_features(feature_hits=feature_hits, thresh=thresh, trials=trials)
#
#     return green, blue












