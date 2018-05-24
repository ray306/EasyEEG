# (todo)
- correct()
- Better support for parallel cores
- classification and regression
- Make the `extract` more flexible
- Integrate figures with labels into one
- classification across timepints and across trials
- roll_on_level on time and time_group

# 0.8.4.1
- Fixed an issue when the model in `classification` won't ouput weights.

# 0.8.4
- Added a slot `supplement` to `Analyzed_data`
- Supported multiple (>2) return values for `roll_on_levels`
- The output of `classification` will include `weights`, and it will be at attribute `supplement` of `Analyzed_data`
- Add default parameters for the topographical plot in `matrix_plot`
- If the parameter `positions`'s value of `float_plot` is "channels", `positions` will be set as `io.load_topolocs('standard-10-5-cap385', None)`
- Make the default values `annotation=None` and `fig=None` for `float_plot` and `matrix_plot`

# 0.8.3.2
- Changed the Y-axis title to `power` for `Spectrum`
- Changed the printing format of `AnalyzeData`

# 0.8.3.1
- Fixed a issue in `multiple_comparison_correction`
- Fixed a issue of loading `AnalyzeData`

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