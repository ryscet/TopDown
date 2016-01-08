
eeglab
path = '/Users/ryszardcetnarski/Desktop/Nencki/TD/Data';
output = '/Users/ryszardcetnarski/Desktop/Nencki/TD/Converted_Data/signals/';
second_output = '/Users/ryszardcetnarski/Desktop/Nencki/TD/Converted_Data/events/';
files = dir(path);
%Find all the files with .set type FROM THE FIRST ROUND (before training)
index = find(~cellfun(@isempty,strfind({files.name},'2_TD.set')));
files = {files.name};
%Sort them alphabetically
files = sort(files(index));
all_events = {}
for i = 1:length(files)
    files{i}
    EEG = pop_loadset('filename',files{i},'filepath',path);
    sum(sum(double(EEG.data)));
    all_events{i} = EEG.event;
    %SpectralAnalysis(EEG, 2.0, 0.5)
    %slices = pop_rmdat(EEG, {'S176','S192'},[-2.0 0.5] ,0);
    %EEG = eeg_checkset( EEG );

    %pop_prop(slices , 1, 5, NaN, {'freqrange' [2 70] });

    name = strcat(output,files{i}(1:end-4),'.mat');
    eegToSave = EEG.data;
    save(name,'eegToSave')
    save(strcat(second_output, files{i}(1:end-4),'_EVENTS'),'-struct', 'EEG', 'event' );
    save(strcat(second_output, files{i}(1:end-4),'_UREVENTS'), '-struct', 'EEG', 'urevent');
    save(strcat(second_output, files{i}(1:end-4),'_CONDITION'),'-struct', 'EEG','condition', 'session', 'group', 'ref');
    save(strcat(second_output, files{i}(1:end-4),'_ELECTRODES'),'-struct', 'EEG','chanlocs');
end



