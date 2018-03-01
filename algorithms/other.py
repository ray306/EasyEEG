# from ..default import *
# from ..condition import *
# from .. import draw
# from ..group_template import group_template


# def Latency(data,groups,window,sig_limit=0.1,func='max'):
#     def peak(x):
#         if func=='min':
#             index = signal.argrelmin(x, order=3)[0]
#             if(len(index)==0):
#                 return 0
#             extremeValue = min([x[i] for i in index])
#             individual_peaks = np.apply_along_axis(minP, 2, individual_win)
#         elif func=='max':
#             index = signal.argrelmax(x, order=3)[0]
#             if(len(index)==0):
#                 return 0
#             extremeValue = max([x[i] for i in index])
#         extremePoint = list(filter(lambda i:x[i]==extremeValue,index))[0]
#         return extremePoint
    
#     func1 = lambda sub: sub.data.mean(0)
#     func2 = lambda sub_1,sub_2: sub_1.data.mean(0)-sub_2.data.mean(0)
    
#     lines_group = group_template(data,groups,'ERP',func1,func2)
#     win = range((window[0]-epoch_time[0])*sr//1000,(window[1]-epoch_time[0])*sr//1000+1)

#     for subject_lines in lines_group: 
#         individual_win = np.array(subject_lines[2])[:,:,win]
#         individual_peaks = np.apply_along_axis(peak, 2, individual_win)
#         #deal with miss values
#         individual_peaks = np.array([individual_peaks[:,i] for i in range(individual_peaks.shape[1]) if min(individual_peaks[:,i])>0]).T
#         #t-test
#         T_result = scipy.stats.ttest_rel(individual_peaks[0],individual_peaks[1])
#         if T_result[1]<sig_limit:
#             print(subject_lines[0],subject_lines[1],window,[round(i,2) for i in individual_peaks.mean(1)],round(T_result[1],4))
#     #     sns.tsplot(individual_peaks.T,err_style="ci_bars",interpolate=False
    
# def Amplitude(data,groups,window,sig_limit=0.1,func=np.mean):
#     func1 = lambda sub: sub.data.mean(0)
#     func2 = lambda sub_1,sub_2: sub_1.data.mean(0)-sub_2.data.mean(0)
    
    
    
#     lines_group = group_template(data,groups,'ERP',func1,func2)
#     win = range((window[0]-epoch_time[0])*sr//1000,(window[1]-epoch_time[0])*sr//1000+1)

#     for subject_lines in lines_group: 
#         individual_win = np.array(subject_lines[2])[:,:,win]
#         individual_peaks = np.apply_along_axis(func, 2, individual_win)
#         #t-test
#         T_result = scipy.stats.ttest_rel(individual_peaks[0],individual_peaks[1])
#         if T_result[1]<sig_limit:
#             print(str(func).split(' ')[2][:-1],subject_lines[0][4:],window,subject_lines[1],
#                   ['%.2f' %i for i in individual_peaks.mean(1)],'%.4f' %T_result[1])
#     #     sns.tsplot(individual_peaks.T,err_style="ci_bars",interpolate=False
