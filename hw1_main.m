%AMATH 582 Homework 1: Ultrasound Data Filtering - W Scherer 1/24/2020
% Find the frequency components of the marble in the data
clear all; close all; clc;
load Testdata

L=15; % spatial domain
n=64; % Fourier modes
x2=linspace(-L,L,n+1); x=x2(1:n); y=x; z=x;
k=(2*pi/(2*L))*[0:(n/2-1) -n/2:-1]; ks=fftshift(k);

[X,Y,Z]=meshgrid(x,y,z);
[Kx,Ky,Kz]=meshgrid(k,k,k);
[Ksx,Ksy,Ksz]=meshgrid(ks,ks,ks);

ut_avg = 0;

% FFT the spacial data for each slice into frequency space to find the
% marble frequency
tdat = [];
kdat = [];
plot_sum = zeros(n,n,n);
for j=1:20
    % working with each slice as a 1D vector of 64^3 values to make finding
    % max value easier - unflond linear index later to find 3d indeces
    Un(:)=reshape(Undata(j,:),1,n^3);
    Unt(:) = fftn(Un(:,:,:)); 
    ut_avg = ut_avg + Unt; % accumulate fftn values
    
    tdat{j} = reshape(Undata(j,:),n,n,n);
    kdat{j} = fftn(tdat{j});
     % this is for plotting the marble k
    plot_sum = abs(kdat{j}) + plot_sum;
end

%Plot the first noisy unfiltered spatial data sample
figure(1)
    isosurface(X,Y,Z, abs(tdat{1}),0.5)
    title('Raw Ultrasound Signal Intensity')
    xlabel('X (Units)')
    ylabel('Y (Units)')
    zlabel('Z (Units)')
    grid on;
    axis([-20 20 -20 20 -20 20]);


% determing the average frequency at each point to find the max frequency
uk_avg = abs(ut_avg)/20; % avg over all 20 time measurement slices
[Mk, I] = max(uk_avg);    % find the max value and the associated linear index
[kk, jj, ii] = ind2sub(size(Ky),I); % convert the linear index into 3d

%Plot the identified marble frequency
plot_avg = abs(fftshift(plot_sum))/20; % frequency space with noise filtered
figure(2)
    isosurface(Ksx,Ksy,Ksz, plot_avg/max(plot_avg,[],'all'),0.8)
    title('Normailzed Max Frequency')
    xlabel('Kx (shifted)')
    ylabel('Ky (shifted)')
    zlabel('Kz (shifted)')
    grid on;
    axis([-10 10 -10 10 -10 10]);

%determining the 3 components of the marble frequency using the indeces
kxm = Kx(ii,jj,kk);
kym = Ky(ii,jj,kk);
kzm = Kz(ii,jj,kk);
kmax = [kxm, kym, kzm];

% Now with the marble frequency located, filter each of the 20 frames to 
%find the location of the marble in each frame

% Build 3D gaussian filter around each frequency for filter
tau2 = 0.2; tau5 = 0.5; % filter bandwidth
%gf_3d = exp(-tau.*(Ksx-kxm).^2).*exp(-tau.*(Ksy-kym).^2).*exp(-tau.*(Ksz-kzm).^2);
gf_3dt2 = exp(-tau2.*((Kx-kxm).^2+(Ky-kym).^2+(Kz-kzm).^2));
gf_3dt5 = exp(-tau5.*((Kx-kxm).^2+(Ky-kym).^2+(Kz-kzm).^2));
dat_filt2 = [];
dat_filt5 = [];
dat_space2 = [];
dat_space5 = [];

marble_xyz2 = [];
maxval_xyz2 = [];

marble_xyz5 = [];
maxval_xyz5 = [];
% this loop should be modified to be a function later
for i = 1:length(kdat)
    
    %apply the filter to each instance of the frequency space
    dat_filt2{i} = gf_3dt2.*kdat{i};
    dat_filt5{i} = gf_3dt5.*kdat{i};
    
    %transform from the filtered frequency space to the now filtered time
    %space
    dat_space2{i} = ifftn(dat_filt2{i});
    dat2filt2{i} = dat_space2{i};
    
    dat_space5{i} = ifftn(dat_filt5{i});
    dat2filt5{i} = dat_space5{i};
    %flatten spatial data to find max
    dat_flat2 = reshape(dat2filt2{i},1,n^3);
    [xyz_max2, Imax2] = max(dat_flat2);
    
    dat_flat5 = reshape(dat2filt5{i},1,n^3);
    [xyz_max5, Imax5] = max(dat_flat5);
    
    maxval_xyz2{i} = xyz_max2;
    maxval_xyz5{i} = xyz_max5;
    % find the x y z coordinates the max for each time slice
    [xmax, ymax, zmax] = ind2sub(size(X),Imax2);
    Mx = X(xmax, ymax, zmax); My = Y(xmax, ymax, zmax); 
    Mz = Z(xmax, ymax, zmax);
    marble_xyz2{i} = [Mx, My, Mz];
    
    [xmax, ymax, zmax] = ind2sub(size(X),Imax5);
    Mx = X(xmax, ymax, zmax); My = Y(xmax, ymax, zmax); 
    Mz = Z(xmax, ymax, zmax);
    marble_xyz5{i} = [Mx, My, Mz];
    
end

% Plot the filtered spatial signalusing tau = 0.2 
%showing the path of the marble
for j=1:20
    
    figure(3)
    isosurface(X,Y,Z, abs(dat_space2{j}),0.4)
    title('Filter Bandwidth Tau = 0.2')
    xlabel('X (units)')
    ylabel('Y (units)')
    zlabel('Z (units)')
    grid on;
    axis([-18 18 -18 18 -18 18]);
end

% Plot the filtered spatial signalusing tau = 0.5 
%showing the path of the marble
for j=1:20
    
    figure(4)
    isosurface(X,Y,Z, abs(dat_space5{j}),0.3)
    title('Filter Bandwidth Tau = 0.5')
    xlabel('X (units)')
    ylabel('Y (units)')
    zlabel('Z (units)')
    grid on;
    axis([-18 18 -18 18 -18 18]);
end

% Plot the path of the marble using plot3 with 0.2 bandwidth filter
px = []; py = []; pz = [];
for k = 1:20
    
    % generate the vectors representing the marble location along each
    % dimension
    px(k) = marble_xyz2{k}(1);
    py(k) = marble_xyz2{k}(2);
    pz(k) = marble_xyz2{k}(3);
    
end

    figure(5)
    plot3(px,py,pz)
    grid on;
    xlabel('X (units)')
    ylabel('Y (units)')
    zlabel('Z (units)')
    axis([-15 15 -15 15 -15 15])

