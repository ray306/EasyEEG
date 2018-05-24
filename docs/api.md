## Initialization


# `Load data`
load_raw_eeg()
from_mne_epochs()

'load_epochs'
load_dataframe_pickle()
load_dataframe_hdf5(files=[],path='',subjIDs=[],montage_path='standard-10-5-cap385',compress=False)
load_mne_fif()
load_eeglab_mat()
load_filetrip_mat() todo
---

# `Extract the data`
epochs = Epochs(all,average,info)

epochs.save(self, filepath, append=False)

**extracted = epochs.extract(collection_script, average=False)**

extracted.get_batch_names(self, batch_id='all')
extracted.get_dataframe(self, batch_id=0, case_id=0, to_print=False)
extracted.get_array(self, batch_id=0, case_id=0, to_print=False)
extracted.get_index(self, batch_id=0, case_id=0, to_print=False)
extracted.get_info(self, key)
---

# `Analysis`
structure.Extracted_epochs.ERP()
structure.Extracted_epochs.topo_ERPs()
structure.Extracted_epochs.ERPs()
structure.Extracted_epochs.GFP()
structure.Extracted_epochs.Spectrum()
structure.Extracted_epochs.Time_frequency()
structure.Extracted_epochs.topography()
structure.Extracted_epochs.significant_channels_count()
structure.Extracted_epochs.clustering()
structure.Extracted_epochs.TANOVA()
structure.Extracted_epochs.classification()
---

# `Plot`

plot(self, plot_params=None, save=False, return_fig=False)

'figure_group'
float_plot
matrix_plot

'figure_unit'
plot_waveform
plot_spectrum
plot_heatmap
plot_topograph
channel_locs(topo)
