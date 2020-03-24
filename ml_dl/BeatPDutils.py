import numpy as np
from collections import OrderedDict

def get_class_weights(data_frame_in):
    pids = data_frame_in['subject_id'].values
    class_weights = np.zeros(np.shape(pids))
    uniq_pid = np.unique(pids)
    freq_pid = np.zeros(uniq_pid.shape)
    for i in range(len(uniq_pid)):
        freq_pid[i] = np.sqrt(np.sum(pids == uniq_pid[i]))
        class_weights[pids == uniq_pid[i]] = 1 / freq_pid[i]
    return class_weights


def sort_dict(in_dict):
	out_dict = OrderedDict()
	if in_dict:
		sorted_keys = sorted(in_dict.keys())
		for key in sorted_keys:
			out_dict[key] = in_dict[key]
	return out_dict	     