function [res_all, res_all_proc] = fft_compute(kostka, fft_window, variant, srate, window_step) % fast fourier transform 

data_length = floor(length(kostka) / fft_window)*fft_window; %????
step = window_step * srate; frames = fft_window /step;
windowarea = sum(hann(fft_window));


for session = 1:size(kostka,3)
	for block = 1:size(kostka,2)
        for w = 1:frames*data_length/fft_window-1;
			start = fft_window*(w-1)/frames+1;
			res_tmp(:,w) = abs(fft(kostka(start:start+fft_window-1,block, session).*hann(fft_window),fft_window))./(windowarea/2);
		end
		res_all(:,block,session) = nanmean(res_tmp,2); % - macierz freq x blocks x sessions
	end
end	

if variant == 1 % compute the input of each frequency as a percent of the entire power
   for session = 1:size(res_all,4)
	for block = 1:size(res_all,3)
      for w = 1:size(res_all,2)  
       suma = sum(res_all(:,block, session),1);
       res_all_proc(:,block,session) = res_all(:,block,session) ./ suma;
    end
    end
   end
else
    res_all_proc = 'was not computed morone';
end

end

