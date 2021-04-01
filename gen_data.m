clear all;
close all;
EEGDIR = [uigetdir(pwd, 'Select folder containing EEG data') '/'];
SAVEDIR = [uigetdir(pwd, 'Select where to save generated data') '/'];
ELISTDIR = [uigetdir(pwd, 'Select folder containing eventlists') '/'];
EXPORTDIR = [uigetdir(pwd, 'Select where to save raw epoch data') '/'];
[BDFFILE, BDFPATH] = uigetfile('*.*', 'Select the bin definition file');
SUB = {'1', '2', '3', '4', '5', '6', '7', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '23', '24', '25', '26', '27', '28', '30', '31', '32', '33', '34', '35', '36', '37', '38', '40'};

for idx = 1:length(SUB)
	[ALLEEG EEG CURRENTSET ALLCOM] = eeglab;
	mkdir([SAVEDIR SUB{idx}]);

	EEG = pop_loadset( 'filename', [SUB{idx} '_SHMW-ALL_downsampled250-ICAweights-ICAcompsRemoved.set'], 'filepath', [EEGDIR SUB{idx} '/']);
	[ALLEEG EEG CURRENTSET] = pop_newset(ALLEEG, EEG, 1, 'setname', 'SHMW-ALL_downsampled250-ICAweights-ICAcompsRemoved', 'gui', 'off');

	EEG = pop_importeegeventlist( EEG, [ELISTDIR SUB{idx} '_eventlist_with_words.txt'], 'ReplaceEventList', 'on');
	[ALLEEG EEG CURRENTSET] = pop_newset(ALLEEG, EEG, 2, 'setname', 'SHMW-ALL_downsampled250-ICAweights-ICAcompsRemoved_impel', 'gui', 'off');

	EEG  = pop_overwritevent(EEG, 'code');
	[ALLEEG EEG CURRENTSET] = pop_newset(ALLEEG, EEG, 3, 'setname', 'SHMW-ALL_downsampled250-ICAweights-ICAcompsRemoved_impel_labeled', 'savenew', [SAVEDIR SUB{idx} '/SHMW-ALL_downsampled250-ICAweights-ICAcompsRemoved_impel_labeled.set'], 'gui', 'off');

	EEG  = pop_binlister( EEG , 'BDF', [BDFPATH BDFFILE], 'IndexEL',  1, 'SendEL2', 'EEG&Text', 'UpdateEEG', 'on', 'Voutput', 'EEG' );
	[ALLEEG EEG CURRENTSET] = pop_newset(ALLEEG, EEG, 4, 'setname', 'SHMW-ALL_downsampled250-ICAweights-ICAcompsRemoved_impel_labeled_bins', 'savenew', [SAVEDIR SUB{idx} '/SHMW-ALL_downsampled250-ICAweights-ICAcompsRemoved_impel_labeled_bins.set'], 'gui', 'off');

	% % skip four steps above
	% EEG = pop_loadset( 'filename',  ['SHMW-ALL_downsampled250-ICAweights-ICAcompsRemoved_impel_labeled_bins.set'], 'filepath', [EEGDIR SUB{idx} '/']);
	% [ALLEEG EEG CURRENTSET] = pop_newset(ALLEEG, EEG, 1, 'setname', 'SHMW-ALL_downsampled250-ICAweights-ICAcompsRemoved_impel_labeled_bins', 'gui', 'off');

	% High Pass filter data for Artifact Rejection
	EEG = pop_basicfilter( EEG,  1:32 , 'Cutoff',  0.1, 'Design', 'butter', 'Filter', 'highpass', 'Order',  2 );
	EEG  = pop_basicfilter( EEG,  1:32 , 'Cutoff',  20, 'Design', 'butter', 'Filter', 'lowpass', 'Order',  8 );
	[ALLEEG EEG CURRENTSET] = pop_newset(ALLEEG, EEG, 5, 'setname', 'SHMW-ALL_downsampled250-ICAweights-ICAcompsRemoved_impel_labeled_bins_filt', 'savenew', [SAVEDIR SUB{idx} '/SHMW-ALL_downsampled250-ICAweights-ICAcompsRemoved_impel_labeled_bins_filt.set'], 'gui', 'off');


	%Epoch continuous dataset
	EEG = pop_epochbin( EEG , [-600.0  1000.0],  'none');
	[ALLEEG EEG CURRENTSET] = pop_newset(ALLEEG, EEG, 6, 'setname', 'SHMW-ALL_downsampled250-ICAweights-ICAcompsRemoved_impel_labeled_bins_filt_epoch', 'savenew', [SAVEDIR SUB{idx} '/SHMW-ALL_downsampled250-ICAweights-ICAcompsRemoved_impel_labeled_bins_filt_epoch.set'], 'gui', 'off'); 

	EEG  = pop_artmwppth( EEG , 'Channel',  1:32, 'Flag',  1, 'Threshold',  200, 'Twindow', [ -600 1000], 'Windowsize',  200, 'Windowstep', 100);
	[ALLEEG EEG CURRENTSET] = pop_newset(ALLEEG, EEG, 7, 'setname', 'SHMW-ALL_downsampled250-ICAweights-ICAcompsRemoved_impel_labeled_bins_filt_epoch_mwpp', 'savenew', [SAVEDIR SUB{idx} '/SHMW-ALL_downsampled250-ICAweights-ICAcompsRemoved_impel_labeled_bins_filt_epoch_mwpp.set'], 'gui', 'off');


	% EEG = pop_loadset( 'filename',  ['SHMW-ALL_downsampled250-ICAweights-ICAcompsRemoved_impel_labeled_bins_filt_epoch_mwpp.set'], 'filepath', [EEGDIR SUB{idx} '/']);
	% [ALLEEG EEG CURRENTSET] = pop_newset(ALLEEG, EEG, 1, 'setname', 'SHMW-ALL_downsampled250-ICAweights-ICAcompsRemoved_impel_labeled_bins_filt_epoch_mwpp', 'gui', 'off');

	pop_export(EEG,[EXPORTDIR SUB{idx} '_raw.txt'],'precision',6);

	epoch = struct2table(EEG.EVENTLIST.eventinfo, 'AsArray', true);
	writetable(epoch, [EXPORTDIR 'epochs_' SUB{idx} '.csv'], 'Delimiter', '\t');

	rej = array2table(EEG.reject.rejmanual');
	writetable(rej, [EXPORTDIR 'rejection_' SUB{idx} '.csv'], 'Delimiter', '\t');

	close all

end

% system('python proc.py')