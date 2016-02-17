function [spectr, period]= SpectralAnalysis(loc_EEG, timeBack, timeForth)
    slices = pop_rmdat(loc_EEG, {'S  1'},[timeBack timeForth] ,0); % last argument 0 for invertselection
    tmp = pop_prop(slices, 1, 5, NaN, {'freqrange' [2 70] });
    spectr = 0; 
    period = 0;
    return