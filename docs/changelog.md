# Future works: 
- correct()
- Better support for parallel cores
- classification and regression

# 0.8.4 (todo)
- (todo) Make the `extract` more flexible
- (todo) Integrate figures with labels into one
- (todo) classification across timepints and across trials
- (todo) roll_on_level on time and time_group

# 0.8.3
- Removed `collection` from the hierarchical Extracted_epochs. Now there are only`case` and `batch`
- New method `subtract(raw, between, align)`
- Fixed an issue of trial number when read more than one files of one subject.
- Changed `two_sample` test hypothesis from 'two-sides' to 'greater' in `classification` 
- Added parameter `samplerate` in `load_raw_eeg`
- Set default values for parameter `plot_params` and `ax` in `plot_waveform`, `plot_spectrum`, `plot_heatmap`, and `plot_topograph`
- Supported to output the scores of different subjects in `classification` and `tanova`
- Changed the `smooth` to `cluster` in `multiple_comparison_correction`

# 0.8.2.5
- Now the AnalyzeData has the __repr__ method
- Fixed a issue of TANOVA plot
- Fixed a issue of saving AnalyzeData
- Renamed the method `RMS` to `GFP`

# 0.8.2.3
- Renamed the parameter `mode` to `strategy` in the method `TANOVA`

# 0.8.2.2 
- Now `Time_frequency` supports the power difference between two conditions
- Fixed a issue when plot the heatmap

# 0.8.2 (2018-03-05)
- Removed `time_group` index which are unnecessary in Analyzed_Result
- Refine X axis in figure
- Improved the compatibility of save() and load() of Analyzed_Result
- New algorithm `frequency_topography()`. Now we can calculate the topography of frequency.
- New method `convert()` in module `Basic`

# 0.8.1 (2018-03-01)
- MASSIVE amount of changes
- Altered the package name to "easyEEG"

# 0.8 (2017-12-25)
- First to public release. Merry Xmax!