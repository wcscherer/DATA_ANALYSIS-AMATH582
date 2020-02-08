function mhf = MHW_filter(width,center,tspan)
%Creates a Gabor-Ricker filter with a specified width and center over
% time span vector tspan

A = 1; % normalizing constant - set to one for now
mhf = A.*(1-((tspan-center).^2/width.^2)).*exp(-(tspan-center).^2/(2*width.^2)); % the function

end

