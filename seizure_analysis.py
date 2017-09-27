import os.path
import numpy as np
import scipy.io
import common.time as time
import matplotlib
from matplotlib import cm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import example_1_transforms as transforms
# process all of one type of the competition mat data
# data_type is one of ('ictal', 'interictal', 'test')
SAMPLE_FREQUENCY = 1526/8
#reference period for standardlizing the slope
REF_TIME_END = 32.5
REF_TIME_START = 2.5
#time to calculate and display on the figures
START_TIME = 0
END_TIME = 100
#lowest and highest frequency to calculate
START_FREQ = 1
END_FREQ = 50
#time axis shift for behavior video mismatch
BEHAVIOR_SHIFT = 0
#number of seizure status
STATUS_NUM = 7 
#smoothing peroid for calculation of change of slope and correlation
SMOOTHING_PERIOD = 5

WINDOW_RANGE = SAMPLE_FREQUENCY
SLIDING_PERIOD = WINDOW_RANGE#SAMPLE_FREQUENCY
#number of col and num in the figures
COL_NUM = 1
ROW_NUM = 4 #useless

#min and max of the colorbar(for normalized figures)
COLOR_MIN = -4
COLOR_MAX = 4
#number of channels
PROBLEM_CH = [3] #Note: have to be descendant (ex. [4, 3, 1], not [1, 3, 4])
#STFT_CH = 1
STFT_PERIOD = SAMPLE_FREQUENCY/4
TOTAL_CH_NUM = 8
CH_NUM = TOTAL_CH_NUM - len(PROBLEM_CH)

#threshold of normalized slope to be seizure channel
SLOPE_THRESHOLD = 2.5

def print_variables():
    print
    print '2016 Ya-Liang Chang'
    print
    print '====================='
    print '= Default variables ='
    print '====================='
    print
    print '==============='
    print '= About Data  ='
    print '==============='

    print 'Sampling frequency: ', SAMPLE_FREQUENCY
    print 'Reference time: ',REF_TIME_END, 's ~ ', REF_TIME_START, 's'
    print 'Calculation time for data: ', START_TIME, 's ~ ', END_TIME, 's'
    print 'Shifting time(for video lag adjustment):', BEHAVIOR_SHIFT
    print 
    print '============================'
    print '= About Slope Calculation  ='
    print '============================'
    print 'Smoothing period for slope calculation: ', SMOOTHING_PERIOD
    print 'Size(data points) for sliding window: ',WINDOW_RANGE
    print 'Time(data points for each move of sliding window', SLIDING_PERIOD
    print 'Threshold for slope: ', SLOPE_THRESHOLD
    print
    print '=================='
    print '= About the Plot ='
    print '=================='
    print 'Number of columns and rows of the plot: (col ',COL_NUM ,', row ', ROW_NUM,')'
    print 'Range of colorbar(max and min of the figure): max ',COLOR_MIN, ', min', COLOR_MAX
    print
    print '=============='
    print '= About STFT ='
    print '=============='
    print 'Channel with problem(Note: have to be descending): ', PROBLEM_CH
    print 'Period for STFT window moving: ', STFT_PERIOD
    print 'Total channel number: ', TOTAL_CH_NUM
    print 'Real channel num(because of ignoring problem channel)', CH_NUM
    print
    print '================================================================='
    print

def read_mat_data(filename):
    if os.path.exists(filename):
        mat_data = scipy.io.loadmat(filename)
    else:
        raise Exception("file %s not found" % filename)
    return mat_data
 

def report_time(start):
    print '(Used %dsec)' % (time.get_seconds() - start)
    new_start = time.get_seconds()
    return new_start

# for each data point in ictal, interictal and test,
# generate (X, <y>, <latency>) per channel
def get_data(mat_data, data_type = 'data', problem_channels = []):
    print 'Loading data',

    if 'data_behavior' in mat_data:
        dataKey = 'data_behavior'
    elif 'data_3sFIR' in mat_data:
        dataKey = 'data_3sFIR'
    else:
        dataKey = 'data'
    print "mat:", mat_data[dataKey].shape
    data = mat_data[dataKey][0:TOTAL_CH_NUM,:]
    if len(problem_channels)!=0:
        for each_channel in problem_channels:
            data = np.delete(data, each_channel-1, axis = 0)
    if data_type == 'data':
        print 'Data:', data.shape
        return data
    elif data_type == 'latencies':
        if mat_data[dataKey].shape[0] > TOTAL_CH_NUM:
            latencies = mat_data[dataKey][TOTAL_CH_NUM, :]
        else:
            latencies = np.zeros(len(data[0]))
        print 'Latencies:', latencies
        return latencies


def plot_data(data, plot_name, start_time, end_time, period = 1.0/SAMPLE_FREQUENCY):
    """
    Plot out abitrary data.
    """
    print 'Plotting out figure:', plot_name
    print '==========================================================='
    print '==                                                       =='
    print '== Please check if it is the right period to calculate.  =='
    print '== If yes, use ctrl+W to close the window.               =='
    print '== If not, please answer \'no\' in the folloing question   =='
    print '==  and restart the program.                             =='
    print '==                                                       =='
    print '==========================================================='
    print  
    plt.figure()
    plt.title(plot_name)
    x1 = np.arange(start_time, end_time, period)
    col = data.shape[0]
    for i in range(0, col):
        if not(x1.shape == data[i].shape):
            print 'Error: time out of range(',
            print 't = ', x1.shape, ', data = ', data.shape, ' )'
            break
        if i==0:
            plt.title(plot_name)
        plt.subplot(col, 1, i+1)
        plt.plot(x1, data[i])
        plt.ylim(-0.0015, 0.0015)
    plt.show()


class do_calculation:
    def __init__(self, start_time = START_TIME, end_time = END_TIME, start_freq = START_FREQ, end_freq = END_FREQ):
        self.s = start_time
        self.e = end_time
        self.start_f = start_freq
        self.end_f = end_freq
        self.do_slope = False
        self.do_eigen = False
        self.do_stft = False

    def calculate_eigenvalue_ref(self, data, data_type = 'None'):
        """
        Using sliding window to calculate the change of eigenvalue
        with time.
        """
        print 'Calculating reference change of eigenvalue in ', data_type, ' domain .. (may take some time)'
        #the change of eigenvalue with time in frequency/time domain
        eigen_ref = []
        for i in range(int(REF_TIME_START*SAMPLE_FREQUENCY), int(REF_TIME_END*SAMPLE_FREQUENCY)):
            if data_type == 'Time':
                data_correlation = transforms.TimeCorrelation_whole(self.end_f, 'usf').apply(data[:, i:i+WINDOW_RANGE])
            elif data_type == 'Frequency':
                data_correlation = transforms.FreqCorrelation_whole(self.start_f, self.end_f, 'usf').apply(data[:, i:i+WINDOW_RANGE])
            w = transforms.Eigenvalues().apply(data_correlation)
            eigen_ref.append(w)
     
        eigen_ref = np.array(eigen_ref)
        eigen_ref = np.swapaxes(eigen_ref, 0, 1)
        print data_type, ' eigen ref:', eigen_ref.shape
        return eigen_ref

    def calculate_eigen_change(self, data, ref_mean, ref_std, data_type = 'none'):
        print 'Calculating change of eigenvalue in ', data_type, ' domain .. (may take some time)'
        eigen_change = []
        for i in range(int(self.s*SAMPLE_FREQUENCY), int(self.e*SAMPLE_FREQUENCY), SAMPLE_FREQUENCY/4):
            if (data_type == 'Time'):
                data_correlation = transforms.TimeCorrelation_whole(self.end_f, 'usf').apply(data[:, i:i+WINDOW_RANGE])
            elif (data_type == 'Frequency'):
                data_correlation = transforms.FreqCorrelation_whole(self.start_f, self.end_f, 'usf').apply(data[:, i:i+WINDOW_RANGE])
 
            w = transforms.Eigenvalues().apply(data_correlation)
            eigen_change.append(w)
 
        eigen_change = np.array(eigen_change)
        eigen_change = np.swapaxes(eigen_change, 0, 1)
        print data_type,' change:',  eigen_change.shape
        for i in range (0, eigen_change.shape[1]):
            for j in range(0, CH_NUM):
                eigen_change[j][i] = (eigen_change[j][i] - ref_mean[j]) / ref_std[j][0]
        print data_type, 'eigen change normalized:', eigen_change.shape
        self.do_eigen = True
        return eigen_change

    def calculate_stft(self, data, ref = []):
        print 'Calculating STFT for all channels ... (may take some time)'
        stft_change_ref = []
        stft_change_ref_mean = []
        stft_change_ref_std = []
        #stft_change_ref = transforms.STFT(start, end).apply(ref)
        for i in range(0, CH_NUM):
            ch_stft_change = []
            ch_stft_change_mean = []
            ch_stft_change_std = []
            for j in range(0, ref.shape[1], STFT_PERIOD):
                ref_stft = transforms.STFT(self.start_f, self.end_f).apply(ref[i, j:j+WINDOW_RANGE])
                ch_stft_change.append(ref_stft)
            stft_change_ref.append(ch_stft_change)
            ch_stft_change = np.array(ch_stft_change[0:120][0:120])
            #print 'ch change',  ch_stft_change.shape
            #print ch_stft_change
            ch_stft_change_mean = np.average(ch_stft_change, axis = 0)
            ch_stft_change = np.swapaxes(ch_stft_change,0,1)
            ch_stft_change_std = transforms.Stats().apply(ch_stft_change)
 
            stft_change_ref_mean.append(ch_stft_change_mean)
            stft_change_ref_std.append(ch_stft_change_std)
 
        stft_change_ref = np.array(stft_change_ref)
        print 'stft_change_ref', stft_change_ref.shape
        #print stft_change_ref
        #stft_change_ref = np.swapaxes(stft_change_ref, 0, 1)
        #stft_change_ref_mean = np.average(stft_change_ref, axis = 1)
        stft_change_ref_mean = np.array(stft_change_ref_mean)
        print 'stft_change_ref_mean', stft_change_ref_mean.shape
        stft_change_ref_std = np.array(stft_change_ref_std)
        print 'stft_change_ref_std', stft_change_ref_std.shape
 
        stft_change = []
        for i in range(0, CH_NUM):
            ch_stft_change = []
            for j in range(int(self.s*SAMPLE_FREQUENCY), int(self.e*SAMPLE_FREQUENCY), STFT_PERIOD):
                data_stft = transforms.STFT(self.start_f, self.end_f).apply(data[i, j:j+WINDOW_RANGE])
                for k in range(data_stft.shape[0]):
                    data_stft[k] = (data_stft[k] - stft_change_ref_mean[i][k])/stft_change_ref_std[i][k][0]
                ch_stft_change.append(data_stft)
            stft_change.append(ch_stft_change)
        stft_change = np.array(stft_change)
        stft_change = np.swapaxes(stft_change, 0, 1)
        print 'stft change:',  stft_change.shape
        self.do_stft = True
        return stft_change


    def calculate_corr_ref(self, data, data_type = 'None'):
        print 'Calculating reference change of corr in ', data_type, ' domain'
        corr_ref = []
        for i in range(int(REF_TIME_START*SAMPLE_FREQUENCY), int(REF_TIME_END*SAMPLE_FREQUENCY), SLIDING_PERIOD):
            if data_type == 'Time':
                data_correlation = transforms.TimeCorrelation_whole(self.end_f, 'usf').apply(data[:, i:i+WINDOW_RANGE])
            elif data_type == 'Frequency':
                data_correlation = transforms.FreqCorrelation_whole(self.start_f, self.end_f, 'usf').apply(data[:, i:i+WINDOW_RANGE])
            corr = []
            for j in range(data_correlation.shape[0]):
                sum = 0.0
                for k in range(data_correlation.shape[1]):
                    sum += abs(data_correlation[j][k])
                corr.append(sum)

            corr_ref.append(corr)

        corr_ref = np.array(corr_ref)
        corr_ref = np.swapaxes(corr_ref, 0, 1)
        print data_type, ' corr ref:', corr_ref.shape
        return corr_ref
    def calculate_corr_change(self, data, ref_mean, ref_std, data_type = 'none'):
        corr_change = []
        for i in range(int(self.s*SAMPLE_FREQUENCY), int(self.e*SAMPLE_FREQUENCY), SLIDING_PERIOD):
            if (data_type == 'Time'):
                data_correlation = transforms.TimeCorrelation_whole(self.end_f, 'usf').apply(data[:, i:i+WINDOW_RANGE])
            elif (data_type == 'Frequency'):
                data_correlation = transforms.FreqCorrelation_whole(self.start_f, self.end_f, 'usf').apply(data[:, i:i+WINDOW_RANGE])
            if i == self.s*SAMPLE_FREQUENCY:
                print 'data_corr:', data_correlation.shape
            corr = []
            for j in range(data_correlation.shape[0]):
                sum = 0.0
                for k in range(data_correlation.shape[1]):
                    sum += abs(data_correlation[j][k])
                corr.append(sum)

            w = transforms.Eigenvalues().apply(data_correlation)
            corr_change.append(corr)

        corr_change = np.array(corr_change)
        corr_change = np.swapaxes(corr_change, 0, 1)

        print data_type,' change:',  corr_change.shape
        for i in range (0, corr_change.shape[1]):
            for j in range(0, CH_NUM):
                corr_change[j][i] = (corr_change[j][i] - ref_mean[j]) / ref_std[j][0]
        print data_type, 'corr change normalized:', corr_change.shape
        
        return corr_change


    def calculate_slope_ref(self, data):
        """
        Calculate the standard deviation and normalized slope to define seizures.
        """
        print 'Calculating the reference slope and change of slope ... '
        #reference slope
        slope_stats = []
        for i in range(int(REF_TIME_START*SAMPLE_FREQUENCY), int(REF_TIME_END*SAMPLE_FREQUENCY)):
            slopes = []
            for j in range(0, CH_NUM):
                slope = (data[j, i+1] - data[j, i] ) * SAMPLE_FREQUENCY
                slopes.append(slope)
            slope_stats.append(slopes)
        slope_stats = np.array(slope_stats)
        slope_stats = np.swapaxes(slope_stats, 0 ,1)
        slope_stats = transforms.Stats().apply(slope_stats)
        print "Slope stats(std, min, max):", slope_stats.shape
        print slope_stats
        return slope_stats

    def calculate_slope_change(self, data, slope_stats, data_type = 'change'):
        #change of slope
        #note: smoothed by SMOOTHING_PERIOD s average, calculated for each sec
        slope_change = []
        seizure_num_by_slope = []
        for i in range(int(self.s*SAMPLE_FREQUENCY), int(self.e*SAMPLE_FREQUENCY), SAMPLE_FREQUENCY):
            seizure_channels_by_slope = 0
            slopes = []
            for j in range(0, CH_NUM):
                average_slope = 0.0
                for k in range(0, int(SMOOTHING_PERIOD*SAMPLE_FREQUENCY)):
                    slope = (data[j, i+1+k] - data[j, i+k] ) * SAMPLE_FREQUENCY
                    average_slope += abs(slope)
                average_slope /= SMOOTHING_PERIOD*SAMPLE_FREQUENCY
                slope_normalized = abs(average_slope / slope_stats[j][0])
                #slope_normalized = abs(slope / slope_stats[j][0])
                if (slope_normalized > SLOPE_THRESHOLD):
                    seizure_channels_by_slope += 1
                slopes.append(slope_normalized)
            slope_change.append(slopes)
            seizure_num_by_slope.append(seizure_channels_by_slope)
            
        if data_type == 'change':
            slope_change = np.array(slope_change)
            print 'slope change of each channel', slope_change.shape
            print slope_change
            return slope_change
        elif data_type == 'num':
            seizure_num_by_slope = np.array(seizure_num_by_slope)
            print 'seizure_num_by_slope', seizure_num_by_slope
            self.do_slope = True
            return seizure_num_by_slope


    def plot_figures(self, latencies = [], seizure_num_by_slope = [], slope_change = [], 
           t_eigen_change = [], f_eigen_change = [], stft_change = [], number_of_figures = 0, stft_ch = -1, corr_change = []):
        if (number_of_figures==0): return False
        print 'Plotting out the figures.. ', 
        print 'Use ctrl+W to close the window'
        print
        i = 0
        fig = plt.figure()
        #STFT for STFT_CH
        if (len(stft_change)!=0):
            plt.subplot2grid((number_of_figures, COL_NUM), (i,0), rowspan = 2)
            i+=2
            stft_change = np.swapaxes(stft_change, 0 ,1)
            im = plt.imshow(stft_change, origin = 'lower',
                    aspect = 'auto', extent = [self.s,self.e,self.start_f-1,self.end_f],
                    interpolation = 'none')
            plt.title('Normalized STFT, ch%d'%stft_ch)
            plt.xlabel('time(s)')
            plt.ylabel('frequency(Hz)')
            plt.tight_layout() #adjust the space between plots
            fig.subplots_adjust(right = 0.93)
            plt.clim(COLOR_MIN, COLOR_MAX)
            cbax = fig.add_axes([0.94, 0.82, 0.01,0.12])
            fig.colorbar(im, cax = cbax)
        
        if len(slope_change) != 0:
            i+=1
            #slope change of each channel
            slope_change = np.array(slope_change)
            slope_change = np.swapaxes(slope_change, 0, 1)
            plt.subplot(number_of_figures, COL_NUM, i)
            plt.title('Slope change of each channel(moving average by 5 sec)')
            im = plt.imshow(slope_change, origin = 'lower',
                    aspect = 'auto', extent = [self.s,self.e,1,CH_NUM],
                    interpolation = 'none')
            plt.ylabel('channel')
            fig.subplots_adjust(right = 0.93)
            plt.clim(COLOR_MIN, COLOR_MAX)
            cbax = fig.add_axes([0.94, 0.82, 0.01,0.12])
            fig.colorbar(im, cax = cbax)
 
        if len(t_eigen_change)!=0:
            i+=1
            #time correlation
            plt.subplot(number_of_figures, COL_NUM, i)
            plt.title('Time Domain Correlation Analysis (Normalized)')
            im = plt.imshow(t_eigen_change, origin = 'lower',
                    aspect = 'auto', extent = [self.s,self.e,0,CH_NUM],
                    interpolation = 'none')
            plt.ylabel('eigenvalues')
            plt.clim(COLOR_MIN, COLOR_MAX)
            plt.tight_layout() #adjust the space between plots
            fig.subplots_adjust(right = 0.93)
            cbax = fig.add_axes([0.94, 0.82, 0.01,0.12])
            fig.colorbar(im, cax = cbax)

        if len(f_eigen_change)!=0:
            i+=1
            #phase correlation
            plt.subplot(number_of_figures, COL_NUM, i)
            plt.title('Frequency Domain Correlation Analysis (Normalized)')
            im = plt.imshow(f_eigen_change, origin = 'lower',
                    aspect = 'auto', extent = [self.s,self.e,0,CH_NUM],
                    interpolation = 'none')
            plt.ylabel('eigenvalues')
            plt.tight_layout() #adjust the space between plots
            fig.subplots_adjust(right = 0.93)
            plt.clim(COLOR_MIN, COLOR_MAX)
            cbax = fig.add_axes([0.94, 0.82, 0.01,0.12])
            fig.colorbar(im, cax = cbax)

        if len(corr_change)!=0:
            i+=1
            #correlation sum
            plt.subplot(number_of_figures, COL_NUM, i)
            plt.title('Correlation Sum Analysis (Normalized)')
            im = plt.imshow(corr_change, #origin = 'lower',
                    aspect = 'auto', extent = [self.s, self.e,0,CH_NUM], interpolation = 'none')
            plt.ylabel('correlation sum')
            plt.clim(COLOR_MIN, COLOR_MAX)
            cbax = fig.add_axes([0.94, 0.82, 0.01,0.12])
            fig.colorbar(im, cax = cbax)
 
        #seizure onset by observation
        if len(latencies) != 0:
            i+=1
            plt.subplot(number_of_figures, COL_NUM, i)
            plt.title('Seizure Time by Behavior')
            x2 = np.arange(self.s+BEHAVIOR_SHIFT, self.e+BEHAVIOR_SHIFT, 1.0/SAMPLE_FREQUENCY)
            plt.plot(x2, latencies[self.s*SAMPLE_FREQUENCY:self.e*SAMPLE_FREQUENCY])
            plt.axis([self.s, self.e, 0, STATUS_NUM])
            plt.xlabel('time(s)')
            plt.ylabel('seizure status')
        if len(seizure_num_by_slope) != 0:
            i+=1
            #seizure onset by slope_normalized > 2.5
            plt.subplot(number_of_figures, COL_NUM, i)
            plt.title('Seizure Time by (Normalized Slope > 2.5) num ')
            #x3 = np.arange(self.s, self.e, 1.0/SAMPLE_FREQUENCY)
            x3 = np.arange(self.s, self.e, 1)
            #plt.plot(x3, slope_change)
            plt.plot(x3, seizure_num_by_slope)
            plt.axis([self.s, self.e, 0, CH_NUM])
            plt.xlabel('time(s)')
            plt.ylabel('# of (sn > 2.5)')

        plt.show()
        return True

def input_filename(current_path, input_type = 'Test'):
    print input_type, 'mat file (should be under current path): ',
    file_path = raw_input()
    file_path = os.path.join(current_path, file_path)
    while not os.path.exists(file_path):
        print 'Error: ', file_path, 'not found.'
        print
        print input_type, ' mat file (should be under current path): ',
        file_path = raw_input()
        file_path = os.path.join(current_path, file_path)
    return file_path

def input_variable(var_name):
    while True:
        try:
            new_v = input('Set %s: '%var_name)
            return new_v
            break
        except:
            print 'Error: it must be a number'
            print

def input_yes_or_no(question_name):
    yes_or_no = raw_input('%s (y/n): '%question_name)
    while not (yes_or_no == 'y'
            or yes_or_no == 'Y'
            or yes_or_no == 'N'
            or yes_or_no == 'n'):

        print 'Error: it must be y/n'
        yes_or_no = raw_input('%s? (y/n): '%question_name)
    if yes_or_no == 'Y' or yes_or_no == 'y':
        return True
    else:
        return False

def restart():
    if input_yes_or_no('Restart the program?'):
        return True
    else:
        return False

def main():
    current_path = os.path.dirname(os.path.abspath(__file__))
    print 'Current path: ', current_path
    file_path = input_filename(current_path)
    ref = input_filename(current_path, 'Ref')

    
    print 'Test file: ', file_path
    print 'Reference file: ', ref
    print
    start_time = input_variable('start_time to calculate')
    end_time = input_variable('end_time to calculate')

    start = time.get_seconds()
    initial_start = time.get_seconds()
    mat_data = read_mat_data(file_path)
    data = get_data(mat_data, problem_channels = PROBLEM_CH)
    start = report_time(start)
    plot_data(data[:, start_time*SAMPLE_FREQUENCY:end_time*SAMPLE_FREQUENCY],start_time = start_time, end_time = end_time, plot_name = 'Exp EEG')
    #plot_data(data[STFT_CH-1:STFT_CH,self.s*SAMPLE_FREQUENCY:self.e*SAMPLE_FREQUENCY], plot_name = 'Exp EEG')
    start = report_time(start)

    latencies = get_data(mat_data, 'latencies')

    ref_mat_data = read_mat_data(ref)
    ref_data = get_data(ref_mat_data, problem_channels = PROBLEM_CH)
    plot_data(ref_data[:,REF_TIME_START*SAMPLE_FREQUENCY:REF_TIME_END*SAMPLE_FREQUENCY], plot_name = 'Reference EEG', start_time = REF_TIME_START, end_time = REF_TIME_END)
    start = report_time(start)
    
    do_slope = do_eigen = do_stft = do_corr =False
    do_slope = input_yes_or_no('If do slope?')
    do_eigen = input_yes_or_no('If do correlation structure(eigenvalues)?')
    do_stft = input_yes_or_no('If do STFT?')
    do_corr = input_yes_or_no('If do correlation sum?')
    if (do_slope == False and do_eigen == False and do_stft == False and do_corr == False):
        print 'Nothing to calculate.'
        return restart()
    if (do_corr == True):
        print '============================================================================'
        print '==                                                                        =='
        print '== Note:                                                                  =='
        print '== \'Correlation sum\' is invented by the author of this program,           =='
        print '==  no reference paper,                                                   =='
        print '==  not sure if there is a similar method by others,                      =='
        print '==  not sure if it works well.                                            =='
        print '==  Please check the reliability and inform the author for further usage. =='
        print '==                                                                        =='
        print '============================================================================'
        if not (input_yes_or_no('Read and agree the above?')):
            print 'Not agree.'
            return restart()


    c = do_calculation(start_time, end_time)
    if (do_slope):
        print
        print '=============='
        print '== Do slope =='
        print '=============='
        slope_ref = c.calculate_slope_ref(ref_data)
        #slope_change = calculate_slope_change(data, slope_ref, 'change')
        slope_num = c.calculate_slope_change(data, slope_ref, 'num')
        start = report_time(start)
        print 'ref_data', ref_data.shape
        #stft_change_ref = calculate_stft(ref_data[:, REF_TIME_START*SAMPLE_FREQUENCY:REF_TIME_END*SAMPLE_FREQUENCY], 1, 50)
    if (do_eigen):
        print
        print '=============='
        print '== Do eigen =='
        print '=============='
  
        t_eigen_ref = c.calculate_eigenvalue_ref(ref_data, data_type = "Time")
        start = report_time(start)
        t_eigen_ref_std = transforms.Stats().apply(t_eigen_ref)
        print 'ref std:', t_eigen_ref_std.shape
        start = report_time(start)
        t_eigen_ref_mean = np.average(t_eigen_ref, axis = 1)
        print 'ref avg:', t_eigen_ref_mean.shape
        start = report_time(start)
        t_eigen_change = c.calculate_eigen_change(data, t_eigen_ref_mean, t_eigen_ref_std, data_type = 'Time')
        start = report_time(start)
 
        f_eigen_ref = c.calculate_eigenvalue_ref(ref_data, data_type = 'Frequency')
        f_eigen_ref_std = transforms.Stats().apply(f_eigen_ref)
        f_eigen_ref_mean = np.average(f_eigen_ref, axis = 1)
        f_eigen_change = c.calculate_eigen_change(data, f_eigen_ref_mean, f_eigen_ref_std, data_type = 'Frequency')

    if (do_stft):
        print
        print '============='
        print '== Do STFT =='
        print '============='
        stft_change = c.calculate_stft(data, ref = ref_data[:, REF_TIME_START*SAMPLE_FREQUENCY:REF_TIME_END*SAMPLE_FREQUENCY])
        start = report_time(start)
        stft_change = np.swapaxes(stft_change, 0 , 1)
        print 'stft change swap', stft_change.shape

    if (do_corr):
        print
        print '========================'
        print '== Do correlation sum =='
        print '========================'
        corr_change_ref = c.calculate_corr_ref(ref_data, data_type = 'Time')
        corr_change_ref_std = transforms.Stats().apply(corr_change_ref)
        corr_change_ref_mean = np.average(corr_change_ref, axis = 1)
        start = report_time(start)
        corr_change = c.calculate_corr_change(data, corr_change_ref_mean, corr_change_ref_std, data_type = 'Time')


    print
    if input_yes_or_no('If plot out the results?'):
        def check_plot_options():
            latencies_t = slope_num_t =  slope_change_t = t_eigen_change_t = f_eigen_change_t = stft_change_t = corr_change_t = []
            stft_ch_t = -1
            num_of_figures = 0
            if input_yes_or_no('If show behavior on the plot?'):
                latencies_t = latencies
                num_of_figures += 1
            if do_slope:
                if input_yes_or_no('If show slope?'):
                    slope_num_t = slope_num
                    num_of_figures += 1
            if do_eigen:
                if input_yes_or_no('If show time domain correlation structure?'):
                    t_eigen_change_t = t_eigen_change
                    num_of_figures += 1
                if input_yes_or_no('If show frequency domain correlation strucure?'):
                    f_eigen_change_t = f_eigen_change
                    num_of_figures += 1
            if do_stft:
                if input_yes_or_no('If show Short time Fourier transform(STFT)?\n Note: cannot print STFT with correlatioin structure.'):
                    stft_ch_t = input_variable('STFT channel to show:')
                    stft_change_t = stft_change[stft_ch_t-1]
                    num_of_figures += 2
            if do_corr:
                if input_yes_or_no('If show correlation sum?'):
                    corr_change_t = corr_change
                    num_of_figures += 1

            p = c.plot_figures(latencies = latencies_t, seizure_num_by_slope = slope_num_t,
                   slope_change = slope_change_t,
                   t_eigen_change = t_eigen_change_t,
                   f_eigen_change = f_eigen_change_t,
                   stft_change = stft_change_t,
                   stft_ch = stft_ch_t,
                   corr_change = corr_change_t,
                   number_of_figures = num_of_figures
                    )
            print
            if not p :
                print 'Plotted nothing'
            print 
            if input_yes_or_no('Plot again?'):
                return check_plot_options()
            else:
                return False
        check_plot_options()
    if do_slope:
        if (input_yes_or_no('Save slope num data?')):
            filename = os.path.basename(file_path)
            savefilename = os.path.join(current_path, '%s_slope_%ds_%ds'%(filename, start_time, end_time))
            scipy.io.savemat(savefilename, {'slope_num':slope_num, 'start_time':start_time, 'end_time':end_time})
            print 'Saved file:%s.mat' % savefilename
            print
    if do_eigen:
        if (input_yes_or_no('Save correlation structure data?')):
            filename = os.path.basename(file_path)
            savefilename = os.path.join(current_path, '%s_correlation_structure_%ds_%ds'%(filename, start_time, end_time))
            scipy.io.savemat(savefilename, {'time_corr_struct':t_eigen_change, 'freq_corr_struct': f_eigen_change, 
            'start_time':start_time, 'end_time':end_time})
            print 'Saved file:%s.mat' % savefilename
            print
    if do_stft:
        if (input_yes_or_no('Save STFT data?')):
            filename = os.path.basename(file_path)
            savefilename = os.path.join(current_path, '%s_stft_%ds_%ds'%(filename, start_time, end_time))
            scipy.io.savemat(savefilename, {'stft':stft_change, 'start_time':start_time, 'end_time':end_time})
            print 'Saved file:%s.mat' % savefilename
            print
    if do_corr:
        if (input_yes_or_no('Save correlation sum data?')):
            filename = os.path.basename(file_path)
            savefilename = os.path.join(current_path, '%s_corr_%ds_%ds'%(filename, start_time, end_time))
            scipy.io.savemat(savefilename, {'corr':corr_change, 'start_time':start_time, 'end_time':end_time})
            print 'Saved file:%s.mat' % savefilename
            print


    print
    print '======================'
    print 'Total time:', 
    print
    start = report_time(initial_start)
    print
    return restart()


def parse_command():
    print
    

if __name__ == "__main__":
    print_variables()
    if_stop = main()
    while (if_stop):
        if_stop = main()
    print
    print '[End]'
    print
