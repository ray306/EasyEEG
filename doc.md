Aim；
Simplifying the analysis
- For beginners: clear logics for the steps (least concepts);
- For data players (who want to check various aspects of data or to try various analysis algorithms): one-line code for each idea;
- For data players (who want to apply DIY analysis methods): plenty of low/high level functions for quickly building your own algorithm.

Advantages:
- Don't need to extract the data in a messy way when you want to do new analysis. You can just use a string or a dictionary to describe the target now.
- Applying EEG analysis method by one-line code. (ERP, topography, specturm, time-frequency, etc.)
- Applying advanced EEG analysis methods (clustering, classification, etc.)
- Beautiful ploting
- Plenty of basic and advanced APIs which can help to build your own analysis algorithm and to visualize the result.


---
# todo

---
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
structure.Extracted_epochs.RMS()
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
float_plot(fig, data, annotation, positions, plot_params)
matrix_plot(fig, data, annotation, x_axis, y_axis, plot_params)

'figure_unit'
plot_waveform(ax, data, plot_params)
plot_spectrum(ax, data, plot_params)
plot_heatmap(ax, data, plot_params)
plot_topograph(ax, data, plot_params)
channel_locs(topo)
