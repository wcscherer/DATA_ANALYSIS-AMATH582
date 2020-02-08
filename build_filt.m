function [filter] = build_filt(widths,centers,tspan,f_type)
%This function builds a 1D gabor gaussian or ricker filter
%   Widths is a vector of filter widths (sigma)
%   Centers is a vector of the time values used for the center of the
%   filter
%   tspan is the span of the filter
%   f-type determines if a ricker filter ('MHW')or a gaussian filter ('GG')
%   is desired


% initialize filter vector
filt = zeros(length(centers),length(tspan),length(widths));

if string(f_type) == 'MHW' % Build a Ricker (mexican hat) filter
    
    for k=1:length(widths)
        
        for j = 1:length(centers)
            % build filter using mexihat function 
            filt(j,:,k) = MHW_filter(widths(k),centers(j),tspan);
        
        end
    
    end
    


elseif string(f_type) == 'GG' % Build a Gaussian Gabor filter
    
    for k=1:length(widths) % go through each filter width
        
        for j = 1:length(centers)
            % build filter using gassian gabor function
            filt(j,:,k) = GG_filter(widths(k),centers(j),tspan);
        
        end
    
    end
    
    
elseif string(f_type) == 'SH' % Build a Shannon Gabor filter
    
    for k=1:length(widths) % go through each filter width
        
        for j = 1:length(centers)
            % build filter using gassian gabor function
            filt(j,:,k) = SH_filter(widths(k),centers(j),tspan);
        
        end
    
    end
    


else % only error catch is if the wrong filter type is inputted
    
    print('ERROR - wrong filter type')
    
    
end

filter = filt;

end

