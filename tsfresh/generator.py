import numpy as np
import pandas as pd
from tqdm import tqdm #tqdm_notebook as tqdm
from joblib import Parallel, delayed
import os
import gc
import xgboost as xgb
from sklearn.model_selection import KFold
import scipy as sp
from sklearn import metrics
import sys
from tsfresh.feature_extraction import feature_calculators
import mmap

"""

Feature extraction for tsfresh

Author: Nanxin Chen 

"""

def get_num_lines(file_path):
    fp = open(file_path, "r+")
    buf = mmap.mmap(fp.fileno(), 0)
    lines = 0
    while buf.readline():
        lines += 1
    return lines

class FeatureGenerator(object):
    def __init__(self, file_list, output_file):
        self.file_list = file_list
        self.output_file = output_file

    def features(self, x, prefix):
        feature_dict = dict()

        # create features here
        # numpy
        feature_dict[prefix+'_'+'mean'] = np.mean(x)
        feature_dict[prefix+'_'+'max'] = np.max(x)
        feature_dict[prefix+'_'+'min'] = np.min(x)
        feature_dict[prefix+'_'+'std'] = np.std(x)
        feature_dict[prefix+'_'+'var'] = np.var(x)
        feature_dict[prefix+'_'+'ptp'] = np.ptp(x)
        feature_dict[prefix+'_'+'percentile_10'] = np.percentile(x, 10)
        feature_dict[prefix+'_'+'percentile_20'] = np.percentile(x, 20)
        feature_dict[prefix+'_'+'percentile_30'] = np.percentile(x, 30)
        feature_dict[prefix+'_'+'percentile_40'] = np.percentile(x, 40)
        feature_dict[prefix+'_'+'percentile_50'] = np.percentile(x, 50)
        feature_dict[prefix+'_'+'percentile_60'] = np.percentile(x, 60)
        feature_dict[prefix+'_'+'percentile_70'] = np.percentile(x, 70)
        feature_dict[prefix+'_'+'percentile_80'] = np.percentile(x, 80)
        feature_dict[prefix+'_'+'percentile_90'] = np.percentile(x, 90)

        # scipy
        feature_dict[prefix+'_'+'skew'] = sp.stats.skew(x)
        feature_dict[prefix+'_'+'kurtosis'] = sp.stats.kurtosis(x)
        feature_dict[prefix+'_'+'kstat_1'] = sp.stats.kstat(x, 1)
        feature_dict[prefix+'_'+'kstat_2'] = sp.stats.kstat(x, 2)
        feature_dict[prefix+'_'+'kstat_3'] = sp.stats.kstat(x, 3)
        feature_dict[prefix+'_'+'kstat_4'] = sp.stats.kstat(x, 4)
        feature_dict[prefix+'_'+'moment_1'] = sp.stats.moment(x, 1)
        feature_dict[prefix+'_'+'moment_2'] = sp.stats.moment(x, 2)
        feature_dict[prefix+'_'+'moment_3'] = sp.stats.moment(x, 3)
        feature_dict[prefix+'_'+'moment_4'] = sp.stats.moment(x, 4)
        
        feature_dict[prefix+'_'+'abs_energy'] = feature_calculators.abs_energy(x)
        feature_dict[prefix+'_'+'abs_sum_of_changes'] = feature_calculators.absolute_sum_of_changes(x)
        feature_dict[prefix+'_'+'count_above_mean'] = feature_calculators.count_above_mean(x)
        feature_dict[prefix+'_'+'count_below_mean'] = feature_calculators.count_below_mean(x)
        feature_dict[prefix+'_'+'mean_abs_change'] = feature_calculators.mean_abs_change(x)
        feature_dict[prefix+'_'+'mean_change'] = feature_calculators.mean_change(x)
        feature_dict[prefix+'_'+'var_larger_than_std_dev'] = feature_calculators.variance_larger_than_standard_deviation(x)
        feature_dict[prefix+'_'+'range_minf_m4000'] = feature_calculators.range_count(x, -np.inf, -4000)
        feature_dict[prefix+'_'+'range_m4000_m3000'] = feature_calculators.range_count(x, -4000, -3000)
        feature_dict[prefix+'_'+'range_m3000_m2000'] = feature_calculators.range_count(x, -3000, -2000)
        feature_dict[prefix+'_'+'range_m2000_m1000'] = feature_calculators.range_count(x, -2000, -1000)
        feature_dict[prefix+'_'+'range_m1000_0'] = feature_calculators.range_count(x, -1000, 0)
        feature_dict[prefix+'_'+'range_0_p1000'] = feature_calculators.range_count(x, 0, 1000)
        feature_dict[prefix+'_'+'range_p1000_p2000'] = feature_calculators.range_count(x, 1000, 2000)
        feature_dict[prefix+'_'+'range_p2000_p3000'] = feature_calculators.range_count(x, 2000, 3000)
        feature_dict[prefix+'_'+'range_p3000_p4000'] = feature_calculators.range_count(x, 3000, 4000)
        feature_dict[prefix+'_'+'range_p4000_pinf'] = feature_calculators.range_count(x, 4000, np.inf)

        feature_dict[prefix+'_'+'ratio_unique_values'] = feature_calculators.ratio_value_number_to_time_series_length(x)
        feature_dict[prefix+'_'+'first_loc_min'] = feature_calculators.first_location_of_minimum(x)
        feature_dict[prefix+'_'+'first_loc_max'] = feature_calculators.first_location_of_maximum(x)
        feature_dict[prefix+'_'+'last_loc_min'] = feature_calculators.last_location_of_minimum(x)
        feature_dict[prefix+'_'+'last_loc_max'] = feature_calculators.last_location_of_maximum(x)
        feature_dict[prefix+'_'+'time_rev_asym_stat_10'] = feature_calculators.time_reversal_asymmetry_statistic(x, 10)
        feature_dict[prefix+'_'+'time_rev_asym_stat_100'] = feature_calculators.time_reversal_asymmetry_statistic(x, 100)
        feature_dict[prefix+'_'+'time_rev_asym_stat_1000'] = feature_calculators.time_reversal_asymmetry_statistic(x, 1000)
        feature_dict[prefix+'_'+'autocorrelation_1'] = feature_calculators.autocorrelation(x, 1)
        feature_dict[prefix+'_'+'autocorrelation_2'] = feature_calculators.autocorrelation(x, 2)
        feature_dict[prefix+'_'+'autocorrelation_3'] = feature_calculators.autocorrelation(x, 3)
        feature_dict[prefix+'_'+'autocorrelation_4'] = feature_calculators.autocorrelation(x, 4)
        feature_dict[prefix+'_'+'autocorrelation_5'] = feature_calculators.autocorrelation(x, 5)
        feature_dict[prefix+'_'+'autocorrelation_6'] = feature_calculators.autocorrelation(x, 6)
        feature_dict[prefix+'_'+'autocorrelation_7'] = feature_calculators.autocorrelation(x, 7)
        feature_dict[prefix+'_'+'autocorrelation_8'] = feature_calculators.autocorrelation(x, 8)
        feature_dict[prefix+'_'+'autocorrelation_9'] = feature_calculators.autocorrelation(x, 9)
        feature_dict[prefix+'_'+'autocorrelation_10'] = feature_calculators.autocorrelation(x, 10)
        feature_dict[prefix+'_'+'autocorrelation_50'] = feature_calculators.autocorrelation(x, 50)
        feature_dict[prefix+'_'+'autocorrelation_100'] = feature_calculators.autocorrelation(x, 100)
        feature_dict[prefix+'_'+'autocorrelation_1000'] = feature_calculators.autocorrelation(x, 1000)
        feature_dict[prefix+'_'+'c3_1'] = feature_calculators.c3(x, 1)
        feature_dict[prefix+'_'+'c3_2'] = feature_calculators.c3(x, 2)
        feature_dict[prefix+'_'+'c3_3'] = feature_calculators.c3(x, 3)
        feature_dict[prefix+'_'+'c3_4'] = feature_calculators.c3(x, 4)
        feature_dict[prefix+'_'+'c3_5'] = feature_calculators.c3(x, 5)
        feature_dict[prefix+'_'+'c3_10'] = feature_calculators.c3(x, 10)
        feature_dict[prefix+'_'+'c3_100'] = feature_calculators.c3(x, 100)
        for c in range(1, 34):
            feature_dict[prefix+'_'+'fft_{0}_real'.format(c)] = list(feature_calculators.fft_coefficient(x, [{'coeff': c, 'attr': 'real'}]))[0][1]
            feature_dict[prefix+'_'+'fft_{0}_imag'.format(c)] = list(feature_calculators.fft_coefficient(x, [{'coeff': c, 'attr': 'imag'}]))[0][1]
            feature_dict[prefix+'_'+'fft_{0}_ang'.format(c)] = list(feature_calculators.fft_coefficient(x, [{'coeff': c, 'attr': 'angle'}]))[0][1]
        feature_dict[prefix+'_'+'long_strk_above_mean'] = feature_calculators.longest_strike_above_mean(x)
        feature_dict[prefix+'_'+'long_strk_below_mean'] = feature_calculators.longest_strike_below_mean(x)
        feature_dict[prefix+'_'+'cid_ce_0'] = feature_calculators.cid_ce(x, 0)
        feature_dict[prefix+'_'+'cid_ce_1'] = feature_calculators.cid_ce(x, 1)
        feature_dict[prefix+'_'+'binned_entropy_5'] = feature_calculators.binned_entropy(x, 5)
        feature_dict[prefix+'_'+'binned_entropy_10'] = feature_calculators.binned_entropy(x, 10)
        feature_dict[prefix+'_'+'binned_entropy_20'] = feature_calculators.binned_entropy(x, 20)
        feature_dict[prefix+'_'+'binned_entropy_50'] = feature_calculators.binned_entropy(x, 50)
        feature_dict[prefix+'_'+'binned_entropy_80'] = feature_calculators.binned_entropy(x, 80)
        feature_dict[prefix+'_'+'binned_entropy_100'] = feature_calculators.binned_entropy(x, 100)

        feature_dict[prefix+'_'+'num_crossing_0'] = feature_calculators.number_crossing_m(x, 0)
        feature_dict[prefix+'_'+'num_peaks_1'] = feature_calculators.number_peaks(x, 1)
        feature_dict[prefix+'_'+'num_peaks_3'] = feature_calculators.number_peaks(x, 3)
        feature_dict[prefix+'_'+'num_peaks_5'] = feature_calculators.number_peaks(x, 5)
        feature_dict[prefix+'_'+'num_peaks_10'] = feature_calculators.number_peaks(x, 10)
        feature_dict[prefix+'_'+'num_peaks_50'] = feature_calculators.number_peaks(x, 50)
        feature_dict[prefix+'_'+'num_peaks_100'] = feature_calculators.number_peaks(x, 100)
        feature_dict[prefix+'_'+'num_peaks_500'] = feature_calculators.number_peaks(x, 500)

        feature_dict[prefix+'_'+'spkt_welch_density_1'] = list(feature_calculators.spkt_welch_density(x, [{'coeff': 1}]))[0][1]
        feature_dict[prefix+'_'+'spkt_welch_density_2'] = list(feature_calculators.spkt_welch_density(x, [{'coeff': 2}]))[0][1]
        feature_dict[prefix+'_'+'spkt_welch_density_5'] = list(feature_calculators.spkt_welch_density(x, [{'coeff': 5}]))[0][1]
        feature_dict[prefix+'_'+'spkt_welch_density_8'] = list(feature_calculators.spkt_welch_density(x, [{'coeff': 8}]))[0][1]
        feature_dict[prefix+'_'+'spkt_welch_density_10'] = list(feature_calculators.spkt_welch_density(x, [{'coeff': 10}]))[0][1]
        feature_dict[prefix+'_'+'spkt_welch_density_50'] = list(feature_calculators.spkt_welch_density(x, [{'coeff': 50}]))[0][1]
        feature_dict[prefix+'_'+'spkt_welch_density_100'] = list(feature_calculators.spkt_welch_density(x, [{'coeff': 100}]))[0][1]

        feature_dict[prefix+'_'+'time_rev_asym_stat_1'] = feature_calculators.time_reversal_asymmetry_statistic(x, 1)
        feature_dict[prefix+'_'+'time_rev_asym_stat_2'] = feature_calculators.time_reversal_asymmetry_statistic(x, 2)
        feature_dict[prefix+'_'+'time_rev_asym_stat_3'] = feature_calculators.time_reversal_asymmetry_statistic(x, 3)
        feature_dict[prefix+'_'+'time_rev_asym_stat_4'] = feature_calculators.time_reversal_asymmetry_statistic(x, 4)
        feature_dict[prefix+'_'+'time_rev_asym_stat_10'] = feature_calculators.time_reversal_asymmetry_statistic(x, 10)
        feature_dict[prefix+'_'+'time_rev_asym_stat_100'] = feature_calculators.time_reversal_asymmetry_statistic(x, 100)        

        for r in range(20):
            feature_dict[prefix+'_'+'symmetry_looking_'+str(r)] = feature_calculators.symmetry_looking(x, [{'r': r * 0.05}])[0][1]

        for r in range(1, 20):
            feature_dict[prefix+'_'+'large_standard_deviation_'+str(r)] = feature_calculators.large_standard_deviation(x, r*0.05)

        for r in range(1, 10):
            feature_dict[prefix+'_'+'quantile_'+str(r)] = feature_calculators.quantile(x, r*0.1)

        for r in ['mean', 'median', 'var']:
            feature_dict[prefix+'_'+'agg_autocorr_'+r] = feature_calculators.agg_autocorrelation(x, [{'f_agg': r, 'maxlag':40}])[0][-1]


        #for r in range(1, 6):
        #    feature_dict[prefix+'_'+'number_cwt_peaks_'+str(r)] = feature_calculators.number_cwt_peaks(x, r)

        for r in range(1, 10):
            feature_dict[prefix+'_'+'index_mass_quantile_'+str(r)] = feature_calculators.index_mass_quantile(x, [{'q': r}])[0][1]


        #for ql in [0., .2, .4, .6, .8]:
        #    for qh in [.2, .4, .6, .8, 1.]:
        #        if ql < qh:
        #            for b in [False, True]:
        #                for f in ["mean", "var"]:
        #                    feature_dict[prefix+'_'+'change_quantiles_'+str(ql)+'_'+str(qh)+'_'+str(b)+'_'+str(f)] = feature_calculators.change_quantiles(x, ql, qh, b, f)

        #for r in [.1, .3, .5, .7, .9]:
        #    feature_dict[prefix+'_'+'approximate_entropy_'+str(r)] = feature_calculators.approximate_entropy(x, 2, r)

        feature_dict[prefix+'_'+'max_langevin_fixed_point'] = feature_calculators.max_langevin_fixed_point(x, 3, 30)

        for r in ['pvalue', 'rvalue', 'intercept', 'slope', 'stderr']:
            feature_dict[prefix+'_'+'linear_trend_'+str(r)] = feature_calculators.linear_trend(x, [{'attr': r}])[0][1]

        for r in ['pvalue', 'teststat', 'usedlag']:
            feature_dict[prefix+'_'+'augmented_dickey_fuller_'+r] = feature_calculators.augmented_dickey_fuller(x, [{'attr': r}])[0][1]


        for r in [0.5, 1, 1.5, 2, 2.5, 3, 5, 6, 7, 10]:
            feature_dict[prefix+'_'+'ratio_beyond_r_sigma_'+str(r)] = feature_calculators.ratio_beyond_r_sigma(x, r)

        #for attr in ["pvalue", "rvalue", "intercept", "slope", "stderr"]:
        #    feature_dict[prefix+'_'+'linear_trend_timewise_'+attr] = feature_calculators.linear_trend_timewise(x, [{'attr': attr}])[0][1]
        #for attr in ["rvalue", "intercept", "slope", "stderr"]:
        #    for i in [5, 10, 50]:
        #        for f in ["max", "min", "mean", "var"]:
        #            feature_dict[prefix+'_'+'agg_linear_trend_'+attr+'_'+str(i)+'_'+f] = feature_calculators.agg_linear_trend(x, [{'attr': attr, 'chunk_len': i, 'f_agg': f}])[0][-1]
        #for width in [2, 5, 10, 20]:
        #    for coeff in range(15):
        #        for w in [2, 5, 10, 20]:
        #            feature_dict[prefix+'_'+'cwt_coefficients_'+str(width)+'_'+str(coeff)+'_'+str(w)] = list(feature_calculators.cwt_coefficients(x, [{'widths': width, 'coeff': coeff, 'w': w}]))[0][1]
        #for r in range(10):
        #    feature_dict[prefix+'_'+'partial_autocorr_'+str(r)] = feature_calculators.partial_autocorrelation(x, [{'lag': r}])[0][1]
        # "ar_coefficient": [{"coeff": coeff, "k": k} for coeff in range(5) for k in [10]],
        # "fft_coefficient": [{"coeff": k, "attr": a} for a, k in product(["real", "imag", "abs", "angle"], range(100))],
        # "fft_aggregated": [{"aggtype": s} for s in ["centroid", "variance", "skew", "kurtosis"]],
        # "value_count": [{"value": value} for value in [0, 1, -1]],
        # "range_count": [{"min": -1, "max": 1}, {"min": 1e12, "max": 0}, {"min": 0, "max": 1e12}],
        # "friedrich_coefficients": (lambda m: [{"coeff": coeff, "m": m, "r": 30} for coeff in range(m + 1)])(3),
        #  "energy_ratio_by_chunks": [{"num_segments": 10, "segment_focus": i} for i in range(10)],
        return feature_dict
 
    def process(self, last=None):
        dtypes = {'Timestamp': pd.np.float32,
          'X': pd.np.float32,
          'Y': pd.np.float32,
          'Z': pd.np.float32}
        ret = []
        for f in tqdm(open(self.file_list, 'r'), total=get_num_lines(self.file_list)):
        #for f in open(self.file_list):
            m = pd.read_csv(f.rstrip(), dtype=dtypes)
            if last is not None and last.isnumeric() and m.shape[0] > last:
                last = int(last)
                delete = (m.shape[0] - last) // 2
                m = m[delete:-delete]
            elif type(last) is str:
                filename = f.rstrip().split('/')[-1]
                mask = pd.read_csv(last + '/' + filename, header=None)
                m = m.iloc[mask.values[:,0].astype(np.bool)]
            fea_x = self.features(m.X.to_numpy(), 'X')
            fea_y = self.features(m.Y.to_numpy(), 'Y')
            fea_z = self.features(m.Z.to_numpy(), 'Z')
            fea_x2 = self.features(np.abs(np.diff(m.X.to_numpy())), 'X2')
            fea_y2 = self.features(np.abs(np.diff(m.Y.to_numpy())), 'Y2')
            fea_z2 = self.features(np.abs(np.diff(m.Z.to_numpy())), 'Z2')
            fea = {**fea_x, **fea_y, **fea_z, **fea_x2, **fea_y2, **fea_z2}
            fea['measurement_id'] = f.rstrip().split('/')[-1][:-4]
            ret.append(fea)
        pd.DataFrame(ret).to_csv(self.output_file, index=False)

if __name__ == '__main__':
    m = FeatureGenerator(sys.argv[1], sys.argv[2])
    if len(sys.argv) > 3:
        m.process(sys.argv[3])
    else:
        m.process()
