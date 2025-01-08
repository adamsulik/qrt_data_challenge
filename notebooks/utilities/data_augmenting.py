import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform


def fast_ewm_correlation_matrix(df, halflife, min_periods=1):
    # Compute alpha (smoothing factor) from span
    alpha = 1 - np.exp(-np.log(2) / halflife)

    # Normalize the data for EWM computation
    def _ewm_normalize(x):
        # Compute exponential weights
        weights = np.power(1 - alpha, np.arange(len(x))[::-1])
        weights /= weights.sum()

        # Weighted mean
        mean = np.sum(x * weights)

        # Weighted variance and standard deviation
        var = np.sum(weights * (x - mean) ** 2)
        return mean, np.sqrt(var)

    # Vectorized correlation computation
    def _vectorized_ewm_corr(data):
        n = data.shape[1]
        corr_matrix = np.eye(n)

        # Precompute means and standard deviations
        means = np.zeros(n)
        stds = np.zeros(n)
        for i in range(n):
            means[i], stds[i] = _ewm_normalize(data[:, i])

        # Compute correlations
        for i in range(n):
            for j in range(i + 1, n):
                # Compute weighted covariance
                x = data[:, i] - means[i]
                y = data[:, j] - means[j]
                weights = np.power(1 - alpha, np.arange(len(x))[::-1])
                weights /= weights.sum()

                # Weighted covariance
                cov = np.sum(weights * x * y)

                # Correlation
                corr = cov / (stds[i] * stds[j])

                corr_matrix[i, j] = corr
                corr_matrix[j, i] = corr

        return corr_matrix

    # Convert to NumPy array and compute correlation
    data = df.values
    corr_matrix = _vectorized_ewm_corr(data)

    # Convert back to DataFrame
    return pd.DataFrame(corr_matrix, index=df.columns, columns=df.columns)


def distance_func(corr, method='simple'):
    assert method in ['simple', 'dist']
    if method == 'simple':
        distance_matrix = 1 - corr
    else:
        distance_matrix = np.sqrt(2 * (1 - corr))
    return distance_matrix


def corr2clusters(corr, threshold=2.0, distance_method='simple'):
    corr = corr.fillna(0)
    distance_matrix = distance_func(corr, method=distance_method)
    Z = linkage(squareform(distance_matrix), 'ward')
    threshold = 2.0
    clusters = fcluster(Z, threshold, criterion='distance')
    return clusters


def process_date_group(group_df, halflife, numret_cols=5, cluster_threshold=2.0, distance_method='simple'):
    # compute corr and groups
    ret_columns = [c for c in group_df.columns if c.startswith('RET_')]
    corr = fast_ewm_correlation_matrix(group_df.set_index('STOCK')[ret_columns].T.iloc[::-1], halflife=halflife)
    groups = corr2clusters(corr, threshold=cluster_threshold, distance_method=distance_method)

    # Add avg values from group weighted by first eigen vector
    group_series = pd.Series(groups, index=group_df['STOCK'])
    groups_dict = group_series.groupby(group_series).apply(lambda x: x.index.to_list()).to_dict()

    submatrices = {}
    first_eigvecs = {}
    grouped_avg = pd.DataFrame(index=group_df['STOCK'], columns=[f'Group_ret_{i}' for i in range(1, numret_cols + 1)])

    for group_name, indices in groups_dict.items():
        # get sub corr matrix and compute its eigenvectors
        submatrices[group_name] = corr.loc[indices, indices]
        eigvals, eigvecs = np.linalg.eig(submatrices[group_name].fillna(0))
        first_eigvecs[group_name] = np.absolute(eigvecs[:, 0])

        # apply local averaging based on eigenvector weights
        sub_df = group_df.loc[group_df['STOCK'].isin(indices)][['STOCK'] + ret_columns[:numret_cols]].set_index('STOCK')
        weights = pd.Series(first_eigvecs[group_name], index=indices)
        _local_avg = pd.DataFrame(index=weights.index, columns=grouped_avg.columns)

        # normalize local returns by standard deviation
        _local_stds = group_df.loc[
                          group_df['STOCK'].isin(indices)
                      ][[c for c in group_df.columns
                         if c.startswith('RET_')]].iloc[:, ::-1].T.ewm(halflife=halflife).std().T.iloc[:, ::-1]
        _local_stds.index = sub_df.index
        sub_df = sub_df / _local_stds[sub_df.columns]
        for i, c in enumerate(_local_avg.columns):
            _local_avg[c] = np.average(sub_df[f'RET_{i + 1}'], weights=weights)
        grouped_avg.loc[_local_avg.index] = _local_avg

    # compute global eigenvectors
    eigvals, eigvecs = np.linalg.eig(corr.fillna(0))
    first_eigvec = np.absolute(eigvecs[:, 0])
    second_eigvec = np.absolute(eigvecs[:, 1])
    for eig_id, _name in zip([0, 1], ['First', 'Second']):
        print(f'{_name} eigvec explains {np.absolute(eigvals[eig_id]) / np.sum(np.absolute(eigvals)) * 100:.2f}%')

    ret_cols = [c for c in group_df.columns if c.startswith('RET_')]
    group_df['eig_weight_1'] = first_eigvec
    group_df['eig_weight_2'] = second_eigvec
    group_df['group'] = groups
    group_df['ret_std'] = group_df[ret_cols].std(axis=1)

    group_df['ret_1_norm'] = group_df['RET_1'] / group_df['ret_std']
    for eig_id in [1, 2]:
        group_df[f'global_avg_{eig_id}'] = np.average(group_df['ret_1_norm'].fillna(0),
                                                      weights=group_df[f'eig_weight_{eig_id}'])
    group_df = group_df.merge(grouped_avg, on='STOCK')

    ret_cols.sort(key=lambda x: int(x.split('_')[-1]))
    group_df.loc[:, ret_cols] = group_df[ret_colsutlda].values / group_df['ret_std'].fillna(1).values[:, np.newaxis]

    return group_df
