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
REF_TIME_END = 132.5
REF_TIME_START = 2.5
#time to calculate and display on the figures
START_TIME = 0
END_TIME = 100

#smoothing peroid for calculation of change of slope and correlation
SMOOTHING_PERIOD = 5

WINDOW_RANGE = SAMPLE_FREQUENCY
SLIDING_PERIOD = WINDOW_RANGE#SAMPLE_FREQUENCY
#number of col and num in the figures
COL_NUM = 1
ROW_NUM = 4

#number of channels
CH_NUM = 8

#threshold of normalized slope to be seizure channel
SLOPE_THRESHOLD = 2.5


def parse_input_data(filename):
    if os.path.exists(filename):
        mat_data = scipy.io.loadmat(filename)
    else:
        raise Exception("file %s not found" % filename)
    ictal = 'ictal'

    # for each data point in ictal, interictal and test,
    # generate (X, <y>, <latency>) per channel
    def process_raw_data(mat_data, with_latency):
        start = time.get_seconds()
        initial_start = time.get_seconds()
        print 'Loading data',

        if 'data_behavior' in mat_data:
            dataKey = 'data_behavior'
        elif 'data_3sFIR' in mat_data:
            dataKey = 'data_3sFIR'
        else:
            dataKey = 'data'
        print "mat:", mat_data[dataKey].shape
        data = mat_data[dataKey][0:CH_NUM,:]
        print data.shape,data
        if mat_data[dataKey].shape[0] > CH_NUM:
            latencies = mat_data[dataKey][CH_NUM, :]
        else:
            latencies = np.zeros(len(data[0]))
        print latencies

        """
        Plot out the original EEG signals.
        """
        print 'Plotting out the original EEG signals... ',
        channels_fig = plt.figure()
        x1 = np.arange(START_TIME, END_TIME, 1.0/SAMPLE_FREQUENCY)
        for i in range(0,CH_NUM):
            plt.subplot(CH_NUM, 1, i+1)
            plt.plot(x1, data[i,START_TIME*SAMPLE_FREQUENCY:END_TIME*SAMPLE_FREQUENCY])
        print '(%ds)' % (time.get_seconds() - start)
        start = time.get_seconds()

        """
        Using sliding window to calculate the change of eigenvalue
        with time.
        """
        print 'Calculating change of eigenvalue in frequncy and time domain ... ',
        #the change of eigenvalue with time in frequency/time domain
        t_eigen_ref = []
        for i in range(int(REF_TIME_START*SAMPLE_FREQUENCY), int(REF_TIME_END*SAMPLE_FREQUENCY)):
            data_tc = transforms.TimeCorrelation_whole(50, 'usf').apply(data[:, i:i+WINDOW_RANGE])
            w = transforms.Eigenvalues().apply(data_tc)
            t_eigen_ref.append(w)

        t_eigen_ref = np.array(t_eigen_ref)
        t_eigen_ref = np.swapaxes(t_eigen_ref, 0, 1)
        t_eigen_ref_std = transforms.Stats().apply(t_eigen_ref)
        t_eigen_ref_mean = np.average(t_eigen_ref, axis = 1)
        print 't eigen ref', t_eigen_ref
        print "t eigen ref std", t_eigen_ref_std
        print 't eigen ref mean', t_eigen_ref_mean

        f_eigen_change = []
        t_eigen_change = [] 
        for i in range(int(START_TIME*SAMPLE_FREQUENCY), int(END_TIME*SAMPLE_FREQUENCY)):
            data_tc = transforms.TimeCorrelation_whole(50, 'usf').apply(data[:, i:i+WINDOW_RANGE])
            w = transforms.Eigenvalues().apply(data_tc)
            t_eigen_change.append(w)
            data_fc = transforms.FreqCorrelation_whole(1, 50, 'usf').apply(data[:, i:i+WINDOW_RANGE])
            w = transforms.Eigenvalues().apply(data_fc)
            f_eigen_change.append(w)

        t_eigen_change = np.array(t_eigen_change)
        t_eigen_change = np.swapaxes(t_eigen_change, 0, 1)
        print 't eigen change', t_eigen_change
        for i in range (0, t_eigen_change.shape[1]):
            for j in range(0, CH_NUM):
                t_eigen_change[j][i] = (t_eigen_change[j][i] - t_eigen_ref_mean[j]) / t_eigen_ref_std[j][0]
                if i < 2:
                    print i, j
                    print t_eigen_change[j][i]
                    print t_eigen_ref_mean[j]
                    print t_eigen_ref_std[j][0]
        print 't eigen change normalized', t_eigen_change
        """
        for i in range(0, len(f_eigen_change)):
            f_avg = 0
            t_avg = 0
            for j in range(0, SMOOTHING_PERIOD):
                f_avg += f_eigen_change[]
        """        
        print '(%ds)' % (time.get_seconds() - start)
        start = time.get_seconds()

        """
        Calculate the standard deviation and normalized slope to define seizures.
        """
        print 'Calculating the reference slope and change of slope ... ',
        #reference slope
        slope_stats = []
        for i in range(int(REF_TIME_START*SAMPLE_FREQUENCY), int(REF_TIME_END*SAMPLE_FREQUENCY)):
            slopes = []
            for j in range(0, CH_NUM):
                slope = (data[j, i+1] - data[j, i] ) * SAMPLE_FREQUENCY
                slopes.append(slope)
            slope_stats.append(slopes)
        slope_stats = np.array(slope_stats)
        slope_stats = transforms.Stats().apply(slope_stats)
        print "slope stats:", slope_stats.shape

        #change of slope
        #note: smoothed by SMOOTHING_PERIOD s average, calculated for each sec
        slope_change = []
        seizure_num_by_slope = []
        for i in range(int(START_TIME*SAMPLE_FREQUENCY), int(END_TIME*SAMPLE_FREQUENCY), SAMPLE_FREQUENCY):
            seizure_channels_by_slope = 0
            slopes = []
            for j in range(0, CH_NUM):
                average_slope = 0.0
                for k in range(0, SMOOTHING_PERIOD*SAMPLE_FREQUENCY):
                    slope = (data[j, i+1+k] - data[j, i+k] ) * SAMPLE_FREQUENCY
                    average_slope += slope
                average_slope /= SMOOTHING_PERIOD
                slope_normalized = abs(average_slope / slope_stats[j][0])
                #slope_normalized = abs(slope / slope_stats[j][0])
                if (slope_normalized > SLOPE_THRESHOLD):
                    seizure_channels_by_slope += 1
                slopes.append(slope_normalized)
            slope_change.append(slopes)
            seizure_num_by_slope.append(seizure_channels_by_slope)
            
        slope_change = np.array(slope_change)
        print 'slope change of each channel', slope_change.shape
        seizure_num_by_slope = np.array(seizure_num_by_slope)
        print 'seizure_num_by_slope', seizure_num_by_slope

        print '(%ds)' % (time.get_seconds() - start)
        start = time.get_seconds()


        #Plot out the seizure period and correlation structure.
        print 'Plotting out the other figures.. ', 
        #seizure onset by observation
        fig = plt.figure()
        plt.subplot(ROW_NUM, COL_NUM, 3)
        plt.title('Seizure Time by Behavior')
        x2 = np.arange(START_TIME, END_TIME, 1.0/SAMPLE_FREQUENCY)
        plt.plot(x2, latencies[START_TIME*SAMPLE_FREQUENCY:END_TIME*SAMPLE_FREQUENCY])
        plt.axis([START_TIME, END_TIME, 0, 5])
        plt.xlabel('time(s)')
        plt.ylabel('seizure status')

        #seizure onset by slope_normalized > 2.5
        plt.subplot(ROW_NUM, COL_NUM, 4)
        plt.title('Seizure Time by (Normalized Slope > 2.5) num ')
        #x3 = np.arange(START_TIME, END_TIME, 1.0/SAMPLE_FREQUENCY)
        x3 = np.arange(START_TIME, END_TIME, 1)
        #plt.plot(x3, slope_change)
        plt.plot(x3, seizure_num_by_slope)
        plt.axis([START_TIME, END_TIME, 0, 8])
        plt.ylabel('# of (sn > 2.5)')
        
        #slope change of each channel
        slope_change = np.array(slope_change)
        slope_change = np.swapaxes(slope_change, 0, 1)
        plt.subplot(ROW_NUM, COL_NUM, 1)
        plt.title('Slope change of each channel(moving average by 5 sec)')
        plt.imshow(slope_change, origin = 'lower',
                aspect = 'auto', extent = [START_TIME,END_TIME,1,CH_NUM],
                interpolation = 'none')
        plt.ylabel('channel')
        #plt.colorbar()


        """
        #time correlation
        plt.subplot(ROW_NUM, COL_NUM, 1)
        plt.title('Time Domain Correlation Analysis')
        plt.imshow(t_eigen_change, origin = 'lower',
                aspect = 'auto', extent = [START_TIME,END_TIME,0,7])#, interpolation = 'none')
        plt.ylabel('eigenvalues')
        plt.colorbar()

        #phase correlation
        f_eigen_change = np.array(f_eigen_change)
        f_eigen_change = np.swapaxes(f_eigen_change, 0, 1)
        print "f eigen change", f_eigen_change.shape
        plt.subplot(ROW_NUM, COL_NUM, 2)
        plt.title('Frequency Domain Correlation Analysis')
        plt.imshow(f_eigen_change, origin = 'lower',
                aspect = 'auto', extent = [START_TIME,END_TIME,0,7],
                interpolation = 'none')
        #plt.colorbar()

        print '(%ds)' % (time.get_seconds() - start)
        start = time.get_seconds()

        latencies = np.array(latencies)
        print latencies
        """

        plt.tight_layout() #adjust the space between plots
        plt.show()

    data = process_raw_data(mat_data, with_latency=ictal)


if __name__ == "__main__":
    current_path = os.path.dirname(os.path.abspath(__file__))
    #filename = os.path.join(current_path, 'seizure-data/Rat0614/ptz.mat')
    #filename = os.path.join(current_path, 'seizure-data/Rat0614/blank1.mat')
    #filename = os.path.join(current_path, 'seizure-data/160801seizure/control1-50Hz.mat')
    #filename = os.path.join(current_path, 'seizure-data/160801seizure/ptz1-50Hz.mat')
    #filename = os.path.join(current_path, 'seizure-data/160803seizure/control_1-50_incage.mat')
    #filename = os.path.join(current_path, 'seizure-data/160803seizure/ptz_1-50_incage.mat')
    #filename = os.path.join(current_path, 'seizure-data/160803seizure/ptz_1-50_incage_behavior.mat')
    filename = os.path.join(current_path, 'seizure-data/160805seizure/sei04_ctrl.mat')
    parse_input_data(filename)
