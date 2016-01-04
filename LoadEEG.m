

function x = LoadEEG(input)
    %Run eeglab
    eeglab
    %input.signals
    %input.descriptions
    EEG = pop_loadset('filename', input.descriptions ,'filepath','')
    %save('/Users/ryszardcetnarski/Desktop/test.mat','EEG.xmax') 
    x= EEG.data
    %x= 0
end

