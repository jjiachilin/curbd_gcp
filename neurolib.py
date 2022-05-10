import numpy as np
import csv
from scipy import io

def loadmat(filename):
    '''
    this function should be called instead of direct spio.loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects
    '''
    data = io.loadmat(filename, struct_as_record=False, squeeze_me=True)
    return _check_keys(data)

def _check_keys(dict):
    '''
    checks if entries in dictionary are mat-objects. If yes
    todict is called to change them to nested dictionaries
    '''
    for key in dict:
        if isinstance(dict[key], io.matlab.mio5_params.mat_struct):
            dict[key] = _todict(dict[key])
    return dict        

def _todict(matobj):
    '''
    A recursive function which constructs from matobjects nested dictionaries
    '''
    dict = {}
    for strg in matobj._fieldnames:
        elem = matobj.__dict__[strg]
        if isinstance(elem, io.matlab.mio5_params.mat_struct):
            dict[strg] = _todict(elem)
        else:
            dict[strg] = elem
    return dict

def get_behavioral(path, columns, phase):
    '''Get behavioral data from csv file and save to a numpy array
    
    Params
    ------
    path: string
        path to behavioral data csv file
    columns: list of strings
        list of columns to retrieve
    '''
    
    infile = open(path)
    reader = csv.DictReader(infile)

    reach = []
    out = []
    prev = ''
    for row in reader:
        if row['date'] != 'd1':
            out.append(reach)
            break
        if row['reach'] != prev and prev != '':
            out.append(reach)
            reach = []
        if phase:
            if row[phase] == '1':
                reach.append([float(row[key]) for key in columns])
        else:
                reach.append([float(row[key]) for key in columns])

        prev = row['reach']
    return out

def downsample_behav(behav_dt, neural_dt, behavioral_data):
    '''downsample behavioral_data to match temporal resolution of neural data, neural_dt/behav_dt should be an integer
    
    Params
    ------
    behav_dt: int
        the change in time between behavioral data points
    neural_dt: int
        the change in time between the neural data points
    behavioral_data: ragged tensor
        behavioral data
    
    Returns
    -------
    downsampled_behav_data: ragged tensor 
        behavioral data downsampled according to temporal resolution factor
    '''
    
    downsampled_behav_data = []
    for reach in behavioral_data:
        new_reach = []
        # take average of every neural_dt/behav_dt time points
        for i in range(0, len(reach), neural_dt//behav_dt):
            if i+5 < len(reach):
                new_timebin = np.mean(np.array([reach[i], reach[i+1], reach[i+2], reach[i+3], reach[i+4]], dtype=float), axis=0)
            else:
                count = 0
                while i+count < len(reach):
                    count += 1
                new_timebin = np.mean(np.array([reach[i+c] for c in range(count)], dtype=float), axis=0)
            new_reach.append(new_timebin)
        downsampled_behav_data.append(new_reach)
    return downsampled_behav_data


def trim_neural(neural_data, behavioral_data, reach_start, neural_dt, bins_before, lag=0):
    '''trim firing rates to match behavioral_data
    
    Params
    ------
    neural_data: tensor of shape (k, t, n)
        firing rates
    behavioral_data: ragged tensor
        kinematic data
    reach_start: int
        the starting time in ms of the behavior
    neural_dt: int
        change in time in ms between data points in neural data
    bins_before: int
        number of time bins before the current time point to consider when decoding
    lag: int
        number of time bins to lag the neural data by ie. lag=10 means we are shifting the neural data by 10 timepoints to the left such that 
        we are decoding the behavior at time t with neural data at time t-10
        
    Returns
    -------
    trimmed_rates: ragged tensor
        neural data trimmed according to the varying time length of behavior
    '''
    reach_start //= neural_dt
    trimmed_rates = []
    for i in range(len(neural_data)):
        trimmed_rate = neural_data[i][int(reach_start-bins_before-lag):int(reach_start+len(behavioral_data[i])-lag),:].tolist()
        trimmed_rates.append(trimmed_rate)
    return trimmed_rates


def bin_neural(bins_before, neural_data, bins_current=0):
    '''bin the neural data into so that we can decode behavioral data point from previous n time bins of neural data
    
    Params
    ------
    bins_before: int
        number of bins before the current timepoint to consider when decoding behavior
    bins_current: bool (0 or 1)
        whether or not to consider the current bin, default to 0 because that's cheating
    neural_data: ragged tensor
        trimmed neural data
    
    Returns
    -------
    binned_data: ragged tensor
        data points binned according to bins_before and bins_current
    '''
    binned_data = []
    num_bins = bins_before+bins_current
    for example in neural_data:
        new_example = []
        for i in range(num_bins, len(example)):
            new_example.append(example[i-num_bins:i])
        binned_data.append(new_example)
    binned_data = np.array(binned_data)
    return binned_data


def successful_reaches(neural_data, behavioral_data, success):
    ''' return neural_data and behavioral_data with only successful reaches
    
    Params
    ------
    neural_data:
        binned neural data
    behavioral_data:
        behavioral data
    success:
        one hot encoding of successful reaches where the index is the reach number
    
    Returns
    -------
    success_neural_data:
        neural data of only successful reaches
    success_behavioral_data:
        behavioral data of only successful reaches
    '''
    num_reaches = neural_data.shape[0]
    success_neural_data, success_behavioral_data = [], []
    for i in range(num_reaches):
        if success[i] == 1:
            success_neural_data.append(neural_data[i])
            success_behavioral_data.append(behavioral_data[i])
    return np.array(success_neural_data), np.array(success_behavioral_data)


def n_loop_reaches(neural_data, behavioral_data, loops, n_loops):
    num_reaches = neural_data.shape[0]
    n_loops_neural_data, n_loops_behavioral_data = [], []
    for i in range(num_reaches):
        if loops[i] <= n_loops:
            n_loops_neural_data.append(neural_data[i])
            n_loops_behavioral_data.append(behavioral_data[i])
    return np.array(n_loops_neural_data), np.array(n_loops_behavioral_data)


def sign_early_loop(neural_data, behavioral_data, sign, early_loop):
    num_reaches = neural_data.shape[0]
    success_neural_data, success_behavioral_data = [], []
    for i in range(num_reaches):
        if sign[i] == 1 and early_loop[i] == 0:
            success_neural_data.append(neural_data[i])
            success_behavioral_data.append(behavioral_data[i])
    return np.array(success_neural_data), np.array(success_behavioral_data)
    
def sign_early_loop2(neural_data, sign, early_loop):
    num_reaches = neural_data.shape[2]
    success_neural_data = []
    for i in range(num_reaches):
        if sign[i] == 1 and early_loop[i] == 0:
            success_neural_data.append(neural_data[:,:,i])
    return np.swapaxes(np.swapaxes((np.array(success_neural_data)),0,2),0,1)

"""def soft_norm(rates, softenNorm=5):
    # format rates to jPCA format
    for i in range(rates.shape[2]):
       continue """
    
