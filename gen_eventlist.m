clear all;
close all;
DIR = uigetdir(pwd, 'Select Folder Containing Subjects'' Data');
SAVEDIR = uigetdir(pwd, 'Select Folder to Save Eventlists');
SUB = {'1', '2', '3', '4', '5', '6', '7', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '23', '24', '25', '26', '27', '28', '30', '31', '32', '33', '34', '35', '36', '37', '38', '40'};

for idx = 1:length(SUB)
	[ALLEEG EEG CURRENTSET ALLCOM] = eeglab;

	EEG = pop_loadset( 'filename', [SUB{idx} '_SHMW-ALL_downsampled250-ICAweights-ICAcompsRemoved.set'], 'filepath', [DIR '/' SUB{idx} '/']);
	[ALLEEG EEG CURRENTSET] = pop_newset(ALLEEG, EEG, 1, 'setname', 'SHMW-ALL_downsampled250-ICAweights-ICAcompsRemoved', 'gui', 'off');

	EEG  = pop_creabasiceventlist( EEG , 'AlphanumericCleaning', 'on', 'BoundaryNumeric', { -99 }, 'BoundaryString', { 'boundary' });
	elist = struct2table(EEG.EVENTLIST.eventinfo, 'AsArray', true);
	writetable(elist, [SAVEDIR '/' 'elist_' SUB{idx} '.csv'], 'Delimiter', '\t');

	close all;
end

