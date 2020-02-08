function gfilt = GG_filter(width,center,tspan)
%Creates a Gabor-Gaussian filter with a specified width and center over
% time span vector tspan
%   Detailed explanation goes here
gfilt = exp(-(1/width)*(tspan-center).^2);

end

