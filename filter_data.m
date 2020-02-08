function [sgf, stgf, stgf_sft] = filter_data(input_signal,input_filter)
%This function builds filtered data matrices based on input filters and
%returns a matrix for the filtered data, fft'd data, and the shifted fft'd
%data
%   input_signal is the signal vector to be filtered
%   input_filter, is the pre-built filter vector (see build_filt)

[jft, ift, kft] = size(input_filter);
[js, is] = size(input_signal);

% if is ~= ift
%     
%     print('Wrong filter size, function will break')
%     
% end

sgf = zeros(jft,ift, kft);
stgf = zeros(jft,ift,kft);
stgf_sft = zeros(jft,ift,kft);

for k = 1:kft
    
    for j = 1:jft
        
        sgf(j,:,k) = input_filter(j,:,k).*input_signal; % apply the filter to the signal
        stgf(j,:,k) = fft(sgf(j,:,k));  % fft the time signal to get freq
        stgf_sft(j,:,k) = abs(fftshift(stgf(j,:,k)));%/max(abs(stggf(i,:,k)));
    end
    
end


end

