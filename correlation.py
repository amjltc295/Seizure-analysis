import os.path
import numpy as np
import scipy.io
import common.time as time
import matplotlib
import matplotlib.pyplot as plt
import example_1_transforms as transforms
# process all of one type of the competition mat data
# data_type is one of ('ictal', 'interictal', 'test')
SAMPLE_FREQUENCY = 1526/8
#reference period for standardlizing the slope
REF_TIME_END = 32.5
REF_TIME_START = 2.5
#time to calculate and display on the figures
START_TIME = 600
END_TIME = 620

#smoothing peroid for calculation of change of slope and correlation
SMOOTHING_PERIOD = 5

WINDOW_RANGE = SAMPLE_FREQUENCY
SLIDING_PERIOD = WINDOW_RANGE/4#SAMPLE_FREQUENCY
#number of col and num in the figures
COL_NUM = 1
ROW_NUM = 2

#min and max of the colorbar
COLOR_MIN = -4
COLOR_MAX = 4
#number of channels
#PROBLEM_CH = [4,3,2,1]
PROBLEM_CH = []
TOTAL_CH_NUM = 8
CH_NUM = TOTAL_CH_NUM - len(PROBLEM_CH)

#threshold of normalized slope to be seizure channel
SLOPE_THRESHOLD = 2.5

def parse_input_data(filename, ref = 'None'):
    def read_mat_data(filename):
        if os.path.exists(filename):
            mat_data = scipy.io.loadmat(filename)
        else:
            raise Exception("file %s not found" % filename)
        return mat_data
     

    def report_time(start):
        print '(%ds)' % (time.get_seconds() - start)
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
            print 'Data:', data.shape, data
            return data
        elif data_type == 'latencies':
            if mat_data[dataKey].shape[0] > TOTAL_CH_NUM:
                latencies = mat_data[dataKey][TOTAL_CH_NUM, :]
            else:
                latencies = np.zeros(len(data[0]))
            print 'Latencies:', latencies
            return latencies


    def plot_EEG(data):
        """
        Plot out the original EEG signals.
        """
        print 'Plotting out the original EEG signals... ',
        channels_fig = plt.figure()
        x1 = np.arange(START_TIME, END_TIME, 1.0/SAMPLE_FREQUENCY)
        for i in range(0,CH_NUM):
            plt.subplot(CH_NUM, 1, i+1)
            plt.plot(x1, data[i,START_TIME*SAMPLE_FREQUENCY:END_TIME*SAMPLE_FREQUENCY])
        plt.show()

    def plot_data(data, plot_name = 'None', s = START_TIME, e = END_TIME, period = 1.0/SAMPLE_FREQUENCY):
        """
        Plot out abitrary data.
        """
        print 'Plotting out figure:', plot_name
        plt.figure()
        plt.title(plot_name)
        x1 = np.arange(s, e, period)
        col = data.shape[0]
        for i in range(0, col):
            if i==1:
                plt.title(plot_name)
            plt.subplot(col, 1, i+1)
            plt.plot(x1, data[i])
            plt.ylim(-0.001, 0.001)
        plt.show()


    def calculate_eigenvalue_ref(data, data_type = 'None'):
        """
        Using sliding window to calculate the change of eigenvalue
        with time.
        """
        print 'Calculating reference change of eigenvalue in ', data_type, ' domain'
        #the change of eigenvalue with time in frequency/time domain
        eigen_ref = []
        for i in range(int(REF_TIME_START*SAMPLE_FREQUENCY), int(REF_TIME_END*SAMPLE_FREQUENCY)):
            if data_type == 'Time':
                data_correlation = transforms.TimeCorrelation_whole(50, 'usf').apply(data[:, i:i+WINDOW_RANGE])
            elif data_type == 'Frequency':
                data_correlation = transforms.FreqCorrelation_whole(1, 50, 'usf').apply(data[:, i:i+WINDOW_RANGE])
            w = transforms.Eigenvalues().apply(data_correlation)
            eigen_ref.append(w)

        eigen_ref = np.array(eigen_ref)
        eigen_ref = np.swapaxes(eigen_ref, 0, 1)
        print data_type, ' eigen ref:', eigen_ref.shape
        print eigen_ref
        return eigen_ref

    def calculate_eigen_change(data, ref_mean, ref_std, data_type = 'none'):
        eigen_change = []
        for i in range(int(START_TIME*SAMPLE_FREQUENCY), int(END_TIME*SAMPLE_FREQUENCY), SAMPLE_FREQUENCY/4):
            if (data_type == 'Time'):
                data_correlation = transforms.TimeCorrelation_whole(50, 'usf').apply(data[:, i:i+WINDOW_RANGE])
            elif (data_type == 'Frequency'):
                data_correlation = transforms.FreqCorrelation_whole(1, 50, 'usf').apply(data[:, i:i+WINDOW_RANGE])

            w = transforms.Eigenvalues().apply(data_correlation)
            eigen_change.append(w)

        eigen_change = np.array(eigen_change)
        eigen_change = np.swapaxes(eigen_change, 0, 1)
        print data_type,' change:',  eigen_change
        for i in range (0, eigen_change.shape[1]):
            for j in range(0, CH_NUM):
                if i < 2:
                    print i, j
                    print eigen_change[j][i],
                    print t_eigen_ref_mean[j],
                    print t_eigen_ref_std[j][0]
                eigen_change[j][i] = (eigen_change[j][i] - ref_mean[j]) / ref_std[j][0]
        print data_type, 'eigen change normalized:', eigen_change.shape
        print eigen_change
        return eigen_change

    def calculate_slope_ref(data):
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
        print "Slope stats:", slope_stats.shape
        print slope_stats
        return slope_stats

    def calculate_slope_change(data, slope_stats, data_type = 'change'):
        #change of slope
        #note: smoothed by SMOOTHING_PERIOD s average, calculated for each sec
        slope_change = []
        seizure_num_by_slope = []
        for i in range(int(START_TIME*SAMPLE_FREQUENCY), int(END_TIME*SAMPLE_FREQUENCY), SAMPLE_FREQUENCY):
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
            return seizure_num_by_slope

    
    def plot_figures(latencies = [], seizure_num_by_slope = [], slope_change = [], 
            t_eigen_change = [], f_eigen_change = []):
        #Plot out the seizure period and correlation structure.
        print 'Plotting out the other figures.. ', 
        #seizure onset by observation
        fig = plt.figure()
        """
        plt.subplot(ROW_NUM, COL_NUM, 3)
        plt.title('Seizure Time by Behavior')
        x2 = np.arange(START_TIME, END_TIME, 1.0/SAMPLE_FREQUENCY)
        plt.plot(x2, latencies[START_TIME*SAMPLE_FREQUENCY:END_TIME*SAMPLE_FREQUENCY])
        plt.axis([START_TIME, END_TIME, 0, 5])
        plt.xlabel('time(s)')
        plt.ylabel('seizure status')
        """
        #seizure onset by slope_normalized > 2.5
        plt.subplot(ROW_NUM, COL_NUM, 2)
        plt.title('Seizure Time by (Normalized Slope > 2.5) num ')
        #x3 = np.arange(START_TIME, END_TIME, 1.0/SAMPLE_FREQUENCY)
        x3 = np.arange(START_TIME, END_TIME, 1)
        #plt.plot(x3, slope_change)
        plt.plot(x3, seizure_num_by_slope)
        plt.axis([START_TIME, END_TIME, 0, 8])
        plt.ylabel('# of (sn > 2.5)')
        
        if len(slope_change) != 0:
            #slope change of each channel
            slope_change = np.array(slope_change)
            slope_change = np.swapaxes(slope_change, 0, 1)
            plt.subplot(ROW_NUM, COL_NUM, 1)
            plt.title('Slope change of each channel(moving average by 5 sec)')
            im = plt.imshow(slope_change, origin = 'lower',
                    aspect = 'auto', extent = [START_TIME,END_TIME,1,CH_NUM])#,interpolation = 'none')
            plt.ylabel('channel')
            fig.subplots_adjust(right = 0.93)
            plt.clim(COLOR_MIN, COLOR_MAX)
            cbax = fig.add_axes([0.94, 0.82, 0.01,0.12])
            fig.colorbar(im, cax = cbax)
        else:
            #time correlation
            plt.subplot(ROW_NUM, COL_NUM, 1)
            plt.title('Time Domain Correlation Analysis (Normalized)')
            im = plt.imshow(t_eigen_change, origin = 'lower',
                    aspect = 'auto', extent = [START_TIME,END_TIME,0,CH_NUM])#, interpolation = 'none')
            plt.ylabel('eigenvalues')
            plt.clim(COLOR_MIN, COLOR_MAX)
         
            """
            #phase correlation
            #f_eigen_change = np.array(f_eigen_change)
            #f_eigen_change = np.swapaxes(f_eigen_change, 0, 1)
            print "f eigen change", f_eigen_change.shape
            plt.subplot(ROW_NUM, COL_NUM, 2)
            plt.title('Frequency Domain Correlation Analysis (Normalized)')
            im = plt.imshow(f_eigen_change, origin = 'lower',
                    aspect = 'auto', extent = [START_TIME,END_TIME,0,7])#,                interpolation = 'none')
            #plt.colorbar()
            """
            plt.tight_layout() #adjust the space between plots
            fig.subplots_adjust(right = 0.93)
            plt.clim(COLOR_MIN, COLOR_MAX)
            cbax = fig.add_axes([0.94, 0.82, 0.01,0.12])
            fig.colorbar(im, cax = cbax)

        plt.show()

    start = time.get_seconds()
    initial_start = time.get_seconds()
    mat_data = read_mat_data(filename)
    #data = get_data(mat_data)
    data = get_data(mat_data, problem_channels = PROBLEM_CH)
    start = report_time(start)
    plot_data(data[:,START_TIME*SAMPLE_FREQUENCY:END_TIME*SAMPLE_FREQUENCY], plot_name = 'EEG')
    #plot_EEG(data)
    start = report_time(start)
    if ref!='None':
        print "Reference Data:", ref
        ref_mat_data = read_mat_data(ref)
        ref_data = get_data(ref_mat_data, problem_channels = PROBLEM_CH)
        plot_data(ref_data[:,REF_TIME_START*SAMPLE_FREQUENCY:REF_TIME_END*SAMPLE_FREQUENCY], plot_name = 'Reference EEG', s = REF_TIME_START, e = REF_TIME_END)
    else:
        print "Reference Data:", filename
        ref_data = data


    slope_ref = calculate_slope_ref(ref_data)
    #slope_change = calculate_slope_change(data, slope_ref, 'change')
    slope_num = calculate_slope_change(data, slope_ref, 'num')
    start = report_time(start)
  
    t_eigen_ref = calculate_eigenvalue_ref(ref_data, data_type = "Time")
    start = report_time(start)
    t_eigen_ref_std = transforms.Stats().apply(t_eigen_ref)
    print 'ref std:'
    print t_eigen_ref_std
    start = report_time(start)
    t_eigen_ref_mean = np.average(t_eigen_ref, axis = 1)
    print 'ref avg:'
    print t_eigen_ref_mean
    start = report_time(start)
    t_eigen_change = calculate_eigen_change(data, t_eigen_ref_mean, t_eigen_ref_std, data_type = 'Time')
    start = report_time(start)

    f_eigen_ref = calculate_eigenvalue_ref(ref_data, data_type = 'Frequency')
    f_eigen_ref_std = transforms.Stats().apply(f_eigen_ref)
    f_eigen_ref_mean = np.average(f_eigen_ref, axis = 1)
    f_eigen_change = calculate_eigen_change(data, f_eigen_ref_mean, f_eigen_ref_std, data_type = 'Frequency')
    start = report_time(start)
    plot_figures(latencies = get_data(mat_data, 'latencies'), seizure_num_by_slope = slope_num,
           # slope_change = slope_change,
            t_eigen_change = t_eigen_change, f_eigen_change = f_eigen_change )
    """
    plot_figures(latencies = get_data(mat_data, 'latencies'), seizure_num_by_slope = slope_num,
            slope_change = slope_change)

    """
    print '======================'
    print 'Total time:', 
    start = report_time(initial_start)
    print


if __name__ == "__main__":
    current_path = os.path.dirname(os.path.abspath(__file__))
    ref = os.path.join(current_path, 'seizure-data/seizure03/160803sei03_ctrl.mat')
    #filename = os.path.join(current_path, 'seizure-data/seizure03/160803sei03_ctrl.mat')
    filename = os.path.join(current_path, 'seizure-data/seizure03/160803sei03_ptz_behavior.mat')
    """
    ref = os.path.join(current_path, 'seizure-data/seizure03/160812sei03_ctrl.mat')
    filename = os.path.join(current_path, 'seizure-data/seizure03/160812sei03_ptz_behavior.mat')
    """
    """
    ref = os.path.join(current_path, 'seizure-data/seizure04/160805sei04_ctrl.mat')
    filename = os.path.join(current_path, 'seizure-data/seizure04/160805sei04_ptz_behavior.mat')
    #filename = os.path.join(current_path, 'seizure-data/160805seizure/sei04_ctrl.mat')
    """
    
    parse_input_data(filename, ref = ref)
    #parse_input_data(filename, ref = 'None')
