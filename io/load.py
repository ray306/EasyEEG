from ..default import *
from ..structure import Epochs, Analyzed_data
import pickle
from scipy.io import loadmat

def from_mne_epochs(subj_datas,subjIDs=[],montage_path='standard-10-5-cap385'):
    '''
    Parameters:
        subj_datas: mne.epochs.Epochs or list of mne.epochs.Epochs,
    '''
    if isinstance(subj_datas, mne.epochs.Epochs):
        subj_datas = [subj_datas]

    if subjIDs == []:
        subjIDs = [i+1 for i in range(len(subj_datas))]

    trial_num_for_subjects = dict((s,0) for s in subjIDs)
    epochs_data = []
    # load each subject's eeg data respectively 
    for subjID,epoch_raw in zip(subjIDs,subj_datas): 
        sr = epoch_raw.info['sfreq']
        channel_names = epoch_raw.info['ch_names']

        epochs_data_each_subj = epoch_raw.to_data_frame().stack('signal').unstack('time')
        index_new = [(subjID, condition, trial_num_for_subjects[subjID] + trial, channel) for condition, trial, channel in epochs_data_each_subj.index.tolist()]
        trial_num_for_subjects[subjID] = max([trial for subjID, condition, trial, channel in index_new])

        epochs_data_each_subj.index = pd.MultiIndex.from_tuples(index_new, names=['subject','condition','trial','channel'])

        # td = pd.to_timedelta(epochs_data_each_subj.columns,unit='ms')
        # epochs_data_each_subj.columns = [['data']*len(td), td]
        epochs_data_each_subj.columns.name = 'time'
        
        epochs_data.append(epochs_data_each_subj)
        
        print(subjID,': ')
        print('Condition count',[(k,v//len(channel_names)) for k,v in Counter(epochs_data_each_subj.index.get_level_values('condition')).items()])
        
    # combine eeg data of all the subjects
    epochs_data = pd.concat(epochs_data)    
    epochs_data.sort_index(inplace=True)

    return Epochs(epochs_data, montage_path=montage_path, info={'sample_rate':sr}   )

def load_epochs(files=[],path='',subjIDs=[],montage_path='standard-10-5-cap385',baseline_time=[],
            extremum_in_all=None,extremum_in_baseline=None,
            events_dict=dict(),drop_channels=[],compress=False):

    filepaths = infer_paths(files, path)

    if filepaths[0].endswith('.pickle'):
        return load_dataframe_pickle(files,path,subjIDs,montage_path,compress)
    elif filepaths[0].endswith('.h5'):
        return load_dataframe_hdf5(files,path,subjIDs,montage_path,compress)
    elif filepaths[0].endswith('.fif'):
        return load_mne_fif(files,path,subjIDs,montage_path)
    elif filepaths[0].endswith('.mat'):
        return load_eeglab_mat(files,path,subjIDs,montage_path,baseline_time,
            extremum_in_all,extremum_in_baseline,events_dict,drop_channels)
    else:
        raise ValueError('Only supported pickle files (".pickle"), HDF5 files (".h5"), MNE epochs files(".fif"), or EEGLAB files (".mat")')

def infer_paths(files, path):
    if isinstance(files, str):
        files = [files]
    
    if len(files)==0:
        if path!='':
            filepaths = os.listdir(path)
        else:
            filepaths = os.listdir('data')
    elif len(files)>0:
        if path!='':
            filepaths = [path+'/'+fn for fn in files]
        else:
            filepaths = files

    return filepaths

def load_dataframe_pickle(files=[],path='',subjIDs=[],montage_path='standard-10-5-cap385',compress=False):
    filepaths = infer_paths(files, path)

    sys.modules['pandas.tseries.tdi'] = pd
    sys.modules['pandas.indexes'] = pd.core.indexes

    epochs_data = []
    for filepath in filepaths:
        try:
            df = pd.read_pickle(filepath)
            if isinstance(df, pd.DataFrame):
                if compress:
                    conditions_list = list(df.index.get_level_values('condition').unique())
                    df_averaged = df.mean(level=list(np.setdiff1d(df.index.names, ['trial'])))
                    df_averaged.index = pd.MultiIndex.from_tuples(
                                        [(subject,condition,conditions_list.index(condition),channel) 
                                        for (channel,condition,subject) in df_averaged.index],
                                        names=['subject','condition','trial','channel'])
                    epochs_data.append(df_averaged)
                else:
                    epochs_data.append(df)
            else:
                print('"%s" is not a DataFrame file' %filename)
        except:
            print('"%s" cannot be loaded as pickle file' %filename)

    epochs_data = pd.concat(epochs_data)
    epochs_data.sort_index(inplace=True)

    try:
        epochs_data.columns = [int(i.to_timedelta64())//1000000 for i in epochs_data['data'].columns]
    except:
        pass

    return Epochs(epochs_data, montage_path=montage_path)

def load_dataframe_hdf5(files=[],path='',subjIDs=[],montage_path='standard-10-5-cap385',compress=False):
    def preprocess(df):
        if isinstance(df, pd.DataFrame):
            if compress:
                conditions_list = list(df.index.get_level_values('condition').unique())
                df_averaged = df.mean(level=list(np.setdiff1d(df.index.names, ['trial'])))
                df_averaged.index = pd.MultiIndex.from_tuples(
                                    [(subject,condition,conditions_list.index(condition),channel) 
                                    for (channel,condition,subject) in df_averaged.index],
                                    names=['subject','condition','trial','channel'])
                return df_averaged
            else:
                return df
        else:
            raise Exception(f'{sub_id} in "{filepath}" is not a DataFrame')

    filepaths = infer_paths(files, path)

    epochs_data = []
    info = dict()
    for filepath in filepaths:
        try:
            store = pd.HDFStore(filepath)
            if list(store.keys())==['/all']:
                df = store['/all']
                info = store.get_storer('/all').attrs['info']
                epochs_data.append(preprocess(df))
            for sub_id in store.keys():
                print(f'loading subject{sub_id}')
                df = store[sub_id]
                info = store.get_storer(sub_id).attrs['info']
                epochs_data.append(preprocess(df))
            store.close()
        except:
            raise Exception('"%s" cannot be loaded as a hdf5 file' %filepath)

    if len(epochs_data)>1:
        epochs_data = pd.concat(epochs_data)

    epochs_data.sort_index(inplace=True)

    '''for a unknown bug of pd.HDFStore. del it here and in save.py if fixed'''
    info['trials'] = dict()
    info['trials']['all'] = list(epochs_data.index.get_level_values('trial').unique())
    for subj_id,subj_data in epochs_data.groupby(level=['subject']):
        info['trials'][str(subj_id)] = sorted(list(subj_data.index.get_level_values('trial').unique()))
    info['timepoints'] = {'all':list(epochs_data.columns)}
    '''for a unknown bug of pd.HDFStore'''

    return Epochs(epochs_data, montage_path=montage_path, info=info)

def load_dataframe_hdf5(files=[],path='',subjIDs=[],montage_path='standard-10-5-cap385',compress=False):
    def preprocess(df):
        if isinstance(df, pd.DataFrame):
            if compress:
                conditions_list = list(df.index.get_level_values('condition').unique())
                df_averaged = df.mean(level=list(np.setdiff1d(df.index.names, ['trial'])))
                df_averaged.index = pd.MultiIndex.from_tuples(
                                    [(subject,condition,conditions_list.index(condition),channel) 
                                    for (channel,condition,subject) in df_averaged.index],
                                    names=['subject','condition','trial','channel'])
                return df_averaged
            else:
                return df
        else:
            raise Exception(f'{subjID} in "{filepath}" is not a DataFrame')

    filepaths = infer_paths(files, path)

    epochs_data = []
    info = dict()
    for filepath in filepaths:
        try:
            store = pd.HDFStore(filepath)
            if list(store.keys())==['/all']:
                df = store['/all']
                info = store.get_storer('/all').attrs['info']
                epochs_data.append(preprocess(df))
                print
            if subjIDs==[]:
                subjIDs = store.keys()
            for subjID in subjIDs:
                if subjID != '/supplementary':
                    print(f'reading {subjID[1:]}')
                    df = store[subjID]
                    epochs_data.append(preprocess(df))
                else:
                    info = store.get_storer('/supplementary').attrs['info']
            store.close()
        except Exception as inst:
            raise Exception(f'{inst}. So "{filepath}" cannot be loaded as a hdf5 file.')

    if len(epochs_data)>1:
        epochs_data = pd.concat(epochs_data)
        print('Concatenated.')

    epochs_data.sort_index(inplace=True)

    return Epochs(epochs_data, montage_path=montage_path, info=info)

def load_mne_fif(files=[],path='',subjIDs=[],montage_path='standard-10-5-cap385'):
    '''
    Parameters:
        paths: FIF filepath or list of FIF filepath
    '''
    filepaths = infer_paths(files, path)

    if isinstance(filepaths[0], str):
        subj_datas = [mne.read_epochs(path) for path in filepaths]
    else:
        raise ValueError('Parameter [files] should be path(s)')

    return from_mne_epochs(subj_datas,subjIDs=[],montage_path='standard-10-5-cap385')

def load_eeglab_mat(files=[],path='',subjIDs=[],montage_path='standard-10-5-cap385',baseline_time=[],
            extremum_in_all=None,extremum_in_baseline=None,
            events_dict=dict(),drop_channels=[]):

    filepaths = infer_paths(files, path)

    if subjIDs == []:
        subjIDs = [i+1 for i in range(len(filepaths))]

    epochs_data = []
    # load each subject's eeg data respectively 
    for subjID,filepath in zip(subjIDs,filepaths): 
        # load .mat file
        mat_file = loadmat(filepath)['EEG'][0][0]
        ## get information in mat
        # chanlocs_raw = mat_file[21][0]
        # chanlocs = dict()
        # for l in chanlocs_raw:
        #     chan_name = l[0][0]
        #     # if chan_name in drop_channels:
        #     #     continue
        #     try:
        #         theta = l[2][0][0]
        #         r = l[3][0][0]
        #         chanlocs[chan_name] = [r*math.sin(np.pi/180*theta),r*math.cos(np.pi/180*theta)]
        #     except:
        #         chanlocs[chan_name] = None
        # parameter.xy_locs = chanlocs
        channel_names = [c[0][0] for c in mat_file[21][0]]

        eeg_data = mat_file[15]
        sr = mat_file[11][0][0]
        epoch_time = (int(mat_file[12]*1000),int(mat_file[13]*1000))

        trial_count = mat_file[15].shape[2]
        events = [i[4][0] for i in mat_file[25][0]]
        if events_dict==dict():
            events_dict = {m:m for m in set(events)}

        channels = dict() # build up a dictionary contained id2channel and channel2id
        for ind,i in enumerate(mat_file[21][0]):
            channels[i[0][0]] = ind
            channels[ind] = i[0][0]
        channel_count = mat_file[15].shape[0]
        channel_names = [i[0][0] for i in mat_file[21][0]]

        # find the indexs of timepoints
        # epoch_start = (epoch_time[0]-raw_epoch_time[0])*sr//1000
        # epoch_end = (epoch_time[1]-raw_epoch_time[0])*sr//1000
        if baseline_time != []:
            baseline_start = (baseline_time[0]-epoch_time[0])*sr//1000
            baseline_end = (baseline_time[1]-epoch_time[1])*sr//1000
            baseline_end_new = -baseline_time[0]*sr//1000
        
        # add eeg data
        count = 0
        epochs_data_each_subj = []
        indexs_each_subj = []
        for T in range(trial_count):
            reject = False
            epochs_data_each_trial = []
            indexs_each_trial = []
            
            for C in range(channel_count):
                eeg = eeg_data[C,:,T]

                if baseline_time != []:
                    eeg -= eeg[baseline_start:baseline_end].mean()

                if extremum_in_all:
                    keep1 = min(eeg)> -extremum_in_all and max(eeg) < extremum_in_all
                else:
                    keep1 = True
                if extremum_in_baseline:
                    keep2 = max(eeg[:baseline_end_new])-min(eeg[:baseline_end_new]) < extremum_in_baseline
                else:
                    keep2 = True

                if keep1 and keep2:
                    epochs_data_each_trial.append(np.array(eeg))
                    indexs_each_trial.append([subjID,events_dict[events[T]],T,channels[C]])
                else:
                    reject = True
                    
            if not reject:
                epochs_data_each_subj += epochs_data_each_trial
                indexs_each_subj += indexs_each_trial
                count+=1

        # convert the data to dataframe
        # set up a timeseries for dataframe
        timespan = np.arange(epoch_time[0], epoch_time[1]+1000//sr, 1000//sr)
        index = pd.MultiIndex.from_tuples(indexs_each_subj, names=['subject','condition','trial','channel'])
        epochs_data_each_subj = pd.DataFrame(epochs_data_each_subj, index=index, columns = timespan)
        epochs_data_each_subj.columns.name = 'time'

        # set up a timeseries index for eeg data. eg. start='-200 ms', end='799 ms', freq='2ms'
        # span = pd.timedelta_range(start='%d ms' %epoch_time[0], end='%d ms' %(epoch_time[1]), freq='%d ms' %(1000/sr)) 
        # epochs_data_each_subj.columns = pd.MultiIndex.from_tuples([('data', ts) for ts in span], names=['','time'])

        epochs_data.append(epochs_data_each_subj)
        print(subjID,': ',round(count/trial_count,2),' in ',trial_count)
        print('Condition count',[(k,v//len(channel_names)) for k,v in Counter(epochs_data_each_subj.index.get_level_values('condition')).items()])

    # combine eeg data of all the subjects
    epochs_data = pd.concat(epochs_data)    
    epochs_data.sort_index(inplace=True)

    return Epochs(epochs_data, montage_path=montage_path, info={'sample_rate':sr})


def load_raw_eeg(files=[], path='', subjIDs=[], montage_path='standard-10-5-cap385', events_dict=dict(), default_reference_channels='Cz', drop_channels=[], filter_range=(0.1, 30.), ref_channels='average', epoch_range=(-0.2, 1), samplerate=500, reject=None):
    filepaths = infer_paths(files, path)

    if isinstance(events_dict, list):
            events_dict = {k:ind+1 for ind,k in enumerate(events_dict)}

    if subjIDs == []:
        subjIDs = [i+1 for i in range(len(filepaths))]

    epochs_data = []
    # load each subject's eeg data
    for filepath in filepaths:
        print(filepath)
        if  filepath[-4:] in ['.edf','.bdf','gdf']:
            read_eeg_file = mne.io.read_raw_edf
        elif filepath[-5:] == '.vhdr':
            read_eeg_file = mne.io.read_raw_brainvision
        elif filepath[-5:] == '.fif':
            read_eeg_file = mne.io.read_raw_fif
        elif filepath[-4:] in ['.egi','.mff']:
            read_eeg_file = mne.io.read_raw_egi
        else:
            raise Exception(f'{filepath} is an unsupported file type')

        EEG = read_eeg_file(filepath,event_id=events_dict, preload=True)

        EEG = mne.add_reference_channels(EEG, default_reference_channels)
        EEG.drop_channels(drop_channels)

        EEG.set_montage(mne.channels.read_montage('standard_1020'))

        EEG.filter(filter_range[0], filter_range[1])

        EEG.set_eeg_reference(ref_channels=ref_channels)  # set EEG average reference
        EEG.apply_proj() # Average reference projection was added, but hasn't been applied yet. Use the .apply_proj() method function to apply projections.
        
        EEG.resample(samplerate)

        epoch = mne.Epochs(EEG, EEG._events, events_dict, epoch_range[0], epoch_range[1], baseline=(epoch_range[0], 0), preload=True, reject=reject)
        
        epochs_data.append(epoch)

    epochs_data = from_mne_epochs(epochs_data,subjIDs=subjIDs,montage_path=montage_path)
    return epochs_data

def load_filetrip_mat():
    pass

def load_AnalyzedData(filepath):
    with open(filepath, 'rb') as f:
        analysis_name = pickle.load(f)
        data = pickle.load(f)
        annotation = pickle.load(f)
        supplement = pickle.load(f)
        default_plot_params = pickle.load(f)

        data.name = pickle.load(f)
        if isinstance(annotation, pd.DataFrame):
            annotation.name = pickle.load(f)

        return Analyzed_data(analysis_name, data, annotation, supplement, default_plot_params)

'copied a lot from MNE 0.14.1'
def load_topolocs(f_path,ch_names):
    def read_montage(kind, ch_names=None, path=None, unit='m', transform=False):
        """Read a generic (built-in) montage.

        Individualized (digitized) electrode positions should be
        read in using :func:`read_dig_montage`.

        In most cases, you should only need the `kind` parameter to load one of
        the built-in montages (see Notes).

        Parameters
        ----------
        kind : str
            The name of the montage file without the file extension (e.g.
            kind='easycap-M10' for 'easycap-M10.txt'). Files with extensions
            '.elc', '.txt', '.csd', '.elp', '.hpts', '.sfp' or '.loc' ('.locs' and
            '.eloc') are supported.
        ch_names : list of str | None
            If not all electrodes defined in the montage are present in the EEG
            data, use this parameter to select subset of electrode positions to
            load. If None (default), all defined electrode positions are returned.

            .. note:: ``ch_names`` are compared to channel names in the montage
                      file after converting them both to upper case. If a match is
                      found, the letter case in the original ``ch_names`` is used
                      in the returned montage.

        path : str | None
            The path of the folder containing the montage file. Defaults to the
            mne/channels/data/montages folder in your mne-python installation.
        unit : 'm' | 'cm' | 'mm'
            Unit of the input file. If not 'm' (default), coordinates will be
            rescaled to 'm'.
        transform : bool
            If True, points will be transformed to Neuromag space.
            The fidicuals, 'nasion', 'lpa', 'rpa' must be specified in
            the montage file. Useful for points captured using Polhemus FastSCAN.
            Default is False.

        Returns
        -------
        pos, ch_names_, kind, selection

        See Also
        --------
        DigMontage
        Montage
        read_dig_montage

        Notes
        -----
        Built-in montages are not scaled or transformed by default.

        Montages can contain fiducial points in addition to electrode
        locations, e.g. ``biosemi64`` contains 67 total channels.

        The valid ``kind`` arguments are:

        ===================   =====================================================
        Kind                  description
        ===================   =====================================================
        standard_1005         Electrodes are named and positioned according to the
                              international 10-05 system.
        standard_1020         Electrodes are named and positioned according to the
                              international 10-20 system.
        standard_alphabetic   Electrodes are named with LETTER-NUMBER combinations
                              (A1, B2, F4, etc.)
        standard_postfixed    Electrodes are named according to the international
                              10-20 system using postfixes for intermediate
                              positions.
        standard_prefixed     Electrodes are named according to the international
                              10-20 system using prefixes for intermediate
                              positions.
        standard_primed       Electrodes are named according to the international
                              10-20 system using prime marks (' and '') for
                              intermediate positions.

        biosemi16             BioSemi cap with 16 electrodes
        biosemi32             BioSemi cap with 32 electrodes
        biosemi64             BioSemi cap with 64 electrodes
        biosemi128            BioSemi cap with 128 electrodes
        biosemi160            BioSemi cap with 160 electrodes
        biosemi256            BioSemi cap with 256 electrodes

        easycap-M10           Brainproducts EasyCap with electrodes named
                              according to the 10-05 system
        easycap-M1            Brainproduct EasyCap with numbered electrodes

        EGI_256               Geodesic Sensor Net with 256 channels

        GSN-HydroCel-32       HydroCel Geodesic Sensor Net with 32 electrodes
        GSN-HydroCel-64_1.0   HydroCel Geodesic Sensor Net with 64 electrodes
        GSN-HydroCel-65_1.0   HydroCel Geodesic Sensor Net with 64 electrodes + Cz
        GSN-HydroCel-128      HydroCel Geodesic Sensor Net with 128 electrodes
        GSN-HydroCel-129      HydroCel Geodesic Sensor Net with 128 electrodes + Cz
        GSN-HydroCel-256      HydroCel Geodesic Sensor Net with 256 electrodes
        GSN-HydroCel-257      HydroCel Geodesic Sensor Net with 256 electrodes + Cz
        ===================   =====================================================

        .. versionadded:: 0.9.0
        """
        if path is None:
            path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'montages')
            supported = ('.elc', '.txt', '.csd', '.sfp', '.elp', '.hpts', '.loc',
                         '.locs', '.eloc')
            montages = [os.path.splitext(f) for f in os.listdir(path)]
            montages = [m for m in montages if m[1] in supported and kind == m[0]]
            if len(montages) != 1:
                raise ValueError('Could not find the montage. Please provide the '
                                 'full path.')
            kind, ext = montages[0]
        else:
            kind, ext = os.path.splitext(kind)
        fname = os.path.join(path, kind + ext)

        if ext == '.sfp':
            # EGI geodesic
            with open(fname, 'r') as f:
                lines = f.read().replace('\t', ' ').splitlines()

            ch_names_, pos = [], []
            for ii, line in enumerate(lines):
                line = line.strip().split()
                if len(line) > 0:  # skip empty lines
                    if len(line) != 4:  # name, x, y, z
                        raise ValueError("Malformed .sfp file in line " + str(ii))
                    this_name, x, y, z = line
                    ch_names_.append(this_name)
                    pos.append([float(cord) for cord in (x, y, z)])
            pos = np.asarray(pos)
        elif ext == '.elc':
            # 10-5 system
            ch_names_ = []
            pos = []
            with open(fname) as fid:
                # Default units are meters
                for line in fid:
                    if 'UnitPosition' in line:
                        units = line.split()[1]
                        scale_factor = dict(m=1., mm=1e-3)[units]
                        break
                else:
                    raise RuntimeError('Could not detect units in file %s' % fname)
                for line in fid:
                    if 'Positions\n' in line:
                        break
                pos = []
                for line in fid:
                    if 'Labels\n' in line:
                        break
                    pos.append(list(map(float, line.split())))
                for line in fid:
                    if not line or not set(line) - set([' ']):
                        break
                    ch_names_.append(line.strip(' ').strip('\n'))
            pos = np.array(pos) * scale_factor
        elif ext == '.txt':
            # easycap
            try:  # newer version
                data = np.genfromtxt(fname, dtype='str', skip_header=1)
            except TypeError:
                data = np.genfromtxt(fname, dtype='str', skiprows=1)
            ch_names_ = list(data[:, 0])
            az = np.deg2rad(data[:, 2].astype(float))
            pol = np.deg2rad(data[:, 1].astype(float))
            pos = _sph_to_cart(np.array([np.ones(len(az)) * 85., az, pol]).T)
        elif ext == '.csd':
            # CSD toolbox
            dtype = [('label', 'S4'), ('theta', 'f8'), ('phi', 'f8'),
                     ('radius', 'f8'), ('x', 'f8'), ('y', 'f8'), ('z', 'f8'),
                     ('off_sph', 'f8')]
            try:  # newer version
                table = np.loadtxt(fname, skip_header=2, dtype=dtype)
            except TypeError:
                table = np.loadtxt(fname, skiprows=2, dtype=dtype)
            ch_names_ = table['label']
            az = np.deg2rad(table['theta'])
            pol = np.deg2rad(90. - table['phi'])
            pos = _sph_to_cart(np.array([np.ones(len(az)), az, pol]).T)
        elif ext == '.elp':
            # standard BESA spherical
            dtype = np.dtype('S8, S8, f8, f8, f8')
            try:
                data = np.loadtxt(fname, dtype=dtype, skip_header=1)
            except TypeError:
                data = np.loadtxt(fname, dtype=dtype, skiprows=1)

            ch_names_ = data['f1'].astype(np.str)
            az = data['f2']
            horiz = data['f3']
            radius = np.abs(az / 180.)
            az = np.deg2rad(np.array([h if a >= 0. else 180 + h
                                      for h, a in zip(horiz, az)]))
            pol = radius * np.pi
            pos = _sph_to_cart(np.array([np.ones(len(az)) * 85., az, pol]).T)
        elif ext == '.hpts':
            # MNE-C specified format for generic digitizer data
            dtype = [('type', 'S8'), ('name', 'S8'),
                     ('x', 'f8'), ('y', 'f8'), ('z', 'f8')]
            data = np.loadtxt(fname, dtype=dtype)
            ch_names_ = data['name'].astype(np.str)
            pos = np.vstack((data['x'], data['y'], data['z'])).T
        elif ext in ('.loc', '.locs', '.eloc'):
            ch_names_ = np.loadtxt(fname, dtype='S4',
                                   usecols=[3]).astype(np.str).tolist()
            dtype = {'names': ('angle', 'radius'), 'formats': ('f4', 'f4')}
            topo = np.loadtxt(fname, dtype=float, usecols=[1, 2])
            sph = _topo_to_sph(topo)
            pos = _sph_to_cart(sph)
            pos[:, [0, 1]] = pos[:, [1, 0]] * [-1, 1]
        else:
            raise ValueError('Currently the "%s" template is not supported.' %
                             kind)
        selection = np.arange(len(pos))

        if unit == 'mm':
            pos /= 1e3
        elif unit == 'cm':
            pos /= 1e2
        elif unit != 'm':
            raise ValueError("'unit' should be either 'm', 'cm', or 'mm'.")
        if transform:
            names_lower = [name.lower() for name in list(ch_names_)]
            if ext == '.hpts':
                fids = ('2', '1', '3')  # Alternate cardinal point names
            else:
                fids = ('nasion', 'lpa', 'rpa')

            missing = [name for name in fids
                       if name not in names_lower]
            if missing:
                raise ValueError("The points %s are missing, but are needed "
                                 "to transform the points to the MNE coordinate "
                                 "system. Either add the points, or read the "
                                 "montage with transform=False. " % missing)
            nasion = pos[names_lower.index(fids[0])]
            lpa = pos[names_lower.index(fids[1])]
            rpa = pos[names_lower.index(fids[2])]

            neuromag_trans = get_ras_to_neuromag_trans(nasion, lpa, rpa)
            pos = apply_trans(neuromag_trans, pos)

        if ch_names is not None:
            # Ensure channels with differing case are found.
            upper_names = [ch_name.upper() for ch_name in ch_names]
            sel, ch_names_ = zip(*[(i, ch_names[upper_names.index(e)]) for i, e in
                                   enumerate([n.upper() for n in ch_names_])
                                   if e in upper_names])
            sel = list(sel)
            pos = pos[sel]
            selection = selection[sel]
        else:
            ch_names_ = list(ch_names_)
        kind = os.path.split(kind)[-1]
        return pos, ch_names_, kind, selection
    
    def _sph_to_cart(sph):
        """Convert spherical coordinates to Cartesion coordinates."""
        assert sph.ndim == 2 and sph.shape[1] == 3
        sph = np.atleast_2d(sph)
        out = np.empty((len(sph), 3))
        out[:, 2] = sph[:, 0] * np.cos(sph[:, 2])
        xy = sph[:, 0] * np.sin(sph[:, 2])
        out[:, 0] = xy * np.cos(sph[:, 1])
        out[:, 1] = xy * np.sin(sph[:, 1])
        return out

    def _cart_to_sph(cart):
        """Convert Cartesian coordinates to spherical coordinates.

        Parameters
        ----------
        cart_pts : ndarray, shape (n_points, 3)
            Array containing points in Cartesian coordinates (x, y, z)

        Returns
        -------
        sph_pts : ndarray, shape (n_points, 3)
            Array containing points in spherical coordinates (rad, azimuth, polar)
        """
        assert cart.ndim == 2 and cart.shape[1] == 3
        cart = np.atleast_2d(cart)
        out = np.empty((len(cart), 3))
        out[:, 0] = np.sqrt(np.sum(cart * cart, axis=1))
        out[:, 1] = np.arctan2(cart[:, 1], cart[:, 0])
        out[:, 2] = np.arccos(cart[:, 2] / out[:, 0])
        out = np.nan_to_num(out)
        return out

    def _pol_to_cart(pol):
        """Transform polar coordinates to cartesian."""
        out = np.empty((len(pol), 2))
        if pol.shape[1] == 2:  # phi, theta
            out[:, 0] = pol[:, 0] * np.cos(pol[:, 1])
            out[:, 1] = pol[:, 0] * np.sin(pol[:, 1])
        else:  # radial distance, theta, phi
            d = pol[:, 0] * np.sin(pol[:, 2])
            out[:, 0] = d * np.cos(pol[:, 1])
            out[:, 1] = d * np.sin(pol[:, 1])
        return out
    
    pos, ch_names, kind, selection = read_montage(f_path,ch_names)
    new_pos = _pol_to_cart(_cart_to_sph(pos)[:, 1:][:, ::-1])
    
    xy_locs = dict([(ch,[x,y]) for ch,(x,y) in zip(ch_names,new_pos)])
    return xy_locs
