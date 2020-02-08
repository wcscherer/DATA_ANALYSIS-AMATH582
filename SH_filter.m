function filter = SH_filter(width,center,tspan)
%Creates a 1D Gabor-shannon filter with a specified width and center over
% time span vector tspan

%width is the width of the filter
%center is the vector of all centers of the filter along tspan
%tspan is the total span over which the filter applies

f = ones(size(tspan));

min = center - width; max = center + width;

z0 = find(min > tspan | tspan > max);

for i=1:length(z0)
    
    f(z0(i)) = 0;
end

filter = f;


end

