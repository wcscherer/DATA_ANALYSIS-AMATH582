% AMATH 582 Homework 2 - Time-Frequency Analysis, Wyatt Scherer; due 2/7/20
%% Part 1 - Time-frequency Analysis of Handel's Messiah 
clc; clear all; close all;
load handel
s = y'/2; % load Handel Hallelujah Chorus Sample
% y is the sampled data, Fs is the sampling rate

% plot signal sample
figure(1)
plot((1:length(s))/Fs,s);
xlabel('Time [sec]'); ylabel('Amplitude');
title('Unfiltered Signal');

% %playback audio of signal
% p8 = audioplayer(s,Fs);
% playblocking(p8);

% Part 1 Contiuned - Build Spectograms for Handel's Messiah
% create the wave-vector domains and transformations for manipulation
n = length(s); %modern fft alg domains don't require explicit 2^n nodes
t0 = 1/Fs; ts = length(s)/Fs; tspan = linspace(t0,ts,n); t=tspan(1:n);
L = ts-t0;  k = (Fs/n)*[0:n/2 -n/2:-1]; ks = fftshift(k);

st = fft(s(1:n)); stplot = abs(fftshift(st)); smax = abs(max(st));

figure(2)
plot(ks,stplot/smax);
xlabel('Frequency [Hz]'); ylabel('Normalized Amplitude');
%title('Total');

%Build filters to sample through the frequency space
% Starting with Gabor-gaussian filters of varying width

wdm = [0.010,0.050,0.10]; %different filter widths
wdg = [0.010,0.050,0.10];
wds = [0.01,0.05,0.10];
tfilt = linspace(t0,ts,90); %generate a filter center domain changing by 0.1

%build gaussian gabor filters
ggf = build_filt(wdg,tfilt,t,'GG'); %build a gaussian filter for each width
mhf = build_filt(wdm,tfilt,t,'MHW'); %build a Mexihat filter for each width
shf = build_filt(wdm,tfilt,t,'SH'); %build a Mexihat filter for each width

%Applying the different filters to the signal data
[sggf, stggf, stggf_sft] = filter_data(s(1:n),ggf);
[sgmf, stgmf, stgmf_sft] = filter_data(s(1:n),mhf);
[sgsf, stgsf, stgsf_sft] = filter_data(s(1:n),shf);

% figure(3)
% for j =1:length(tfilt)
%     subplot(3,1,1), plot((1:length(s))/Fs,s(1:n),t,ggf(j,:,1))
%     subplot(3,1,2), plot(t,sggf(j,:,1))
%     subplot(3,1,3), plot(ks,stggf_sft(j,:,1))
%     drawnow
%     pause(0.1)
% end

%build spectogram of filtered data for width 1
figure(4)
pcolor(tfilt,ks,stggf_sft(:,:,1).'), shading interp
set(gca,'Fontsize',[14])
colormap(hot)
title('Gaussian Filter width = 0.01')
xlabel('Time (s)')
ylabel('Frequency (Hz)')

%build spectogram of filtered data for width 5
figure(5)
pcolor(tfilt,ks,stggf_sft(:,:,2).'), shading interp
set(gca,'Fontsize',[14])
colormap(hot)
title('Gaussian Filter width = 0.05')
xlabel('Time (s)')
ylabel('Frequency (Hz)')

%build spectogram of filtered data for width 10
figure(6)
pcolor(tfilt,ks,stggf_sft(:,:,3).'), shading interp
set(gca,'Fontsize',[14])
colormap(hot)
title('Gaussian Filter width = 0.10')
xlabel('Time (s)')
ylabel('Frequency (Hz)')

%build spectogram of ricker filtered data for width 5
figure(7)
pcolor(tfilt,ks,stgmf_sft(:,:,2).'), shading interp
set(gca,'Fontsize',[14])
colormap(hot)
title('Ricker Filter width = 0.05')
xlabel('Time (s)')
ylabel('Frequency (Hz)')

%build spectogram of shannon filtered data for width 5
figure(8)
pcolor(tfilt,ks,stgsf_sft(:,:,2).'), shading interp
set(gca,'Fontsize',[14])
colormap(hot)
title('Shannon Filter width = 0.05')
xlabel('Time (s)')
ylabel('Frequency (Hz)')

%% Part 2 - Analysis of Piano Data
clc; clear all; close all;
%load in data
figure(9)
tr_piano = 16;   % record time in seconds
sp = audioread('music1.wav'); Fsp = length(sp)/tr_piano; sp = sp';
plot((1:length(sp))/Fsp,sp);
xlabel('Time [sec]'); ylabel('Amplitude');
title('Mary had a Little Lamb (Piano)');  
drawnow
%p8 = audioplayer(sp,Fsp); playblocking(p8);

% create the wave-vector domains and transformations for manipulation
np = length(sp); %modern fft alg domains don't require explicit 2^n nodes
tp0 = 1/Fsp; tsp = length(sp)/Fsp; tspanp = linspace(tp0,tsp,np); t=tspanp(1:np);
L = tsp-tp0; kp = (Fsp/np)*[0:np/2-1 -np/2:-1]; ksp = fftshift(kp);

stp = fft(sp); stplotp = abs(fftshift(stp)); smaxp = abs(max(stp));

figure(10)
plot(ksp([np/2+1:np/2+50000]),stplotp([np/2+1:np/2+50000])/smaxp);
xlabel('Frequency [Hz]'); ylabel('Normalized Amplitude');
%title('Frequencies of Interest, f(k)');

wdm = [0.10]; %different filter widths
wds = [0.05];
tfilt = [tp0:0.10:tsp]; %generate a filter center domain changing by 0.1

%build gaussian gabor filters
%mhf = build_filt(wdm,tfilt,t,'MHW'); %build a Mexihat filter for each width
sgf = build_filt(wdm,tfilt,t,'SH'); %build a shannon filter for each width

%Applying the different filters to the signal data
[sgmf, stgmf, stgmf_sft] = filter_data(sp,mhf);
[sgsf, stgsf, stgsf_sft] = filter_data(sp,sgf);

figure(11)
pcolor(tfilt,ksp([np/2+1:np/2+10000]),stgsf_sft(:,[np/2+1:np/2+10000],1).'), shading interp
set(gca,'Fontsize',[14])
colormap(hot)
title('Shannon Filter width = 0.05')
xlabel('Time (s)')
ylabel('Frequency [Hz]')


%% Part 2 - Recorder data
clc; close all; clear;
figure(12)
tr_rec=14;  % record time in seconds
sr = audioread('music2.wav'); Fsr=length(sr)/tr_rec; sr = sr';
 plot((1:length(sr))/Fsr,sr);
 xlabel('Time [sec]'); ylabel('Amplitude');
 title('Mary Had a Little Lamb (Recorder)');
 %p8 = audioplayer(y,Fs); playblocking(p8);
 
% create the wave-vector domains and transformations for manipulation
nr = length(sr); %modern fft alg domains don't require explicit 2^n nodes
tr0 = 1/Fsr; tsr = length(sr)/Fsr; tspanr = linspace(tr0,tsr,nr); t=tspanr(1:nr);
L = tsr-tr0; kr = (Fsr/nr)*[0:nr/2-1 -nr/2:-1]; ksr = fftshift(kr);

str = fft(sr); stplotr = abs(fftshift(str)); smaxr = abs(max(str));

figure(13)
plot(ksr([nr/2+1000:nr/2+50000]),stplotr([nr/2+1000:nr/2+50000])/smaxr);
xlabel('Frequency [Hz]'); ylabel('Normalized Amplitude');
%title('Frequencies of Interest, f(k)');

wdm = [0.10]; %different filter widths
wds = [0.05];
tfilt = [tr0:0.10:tsr]; %generate a filter center domain changing by 0.1

%build gaussian gabor filters
%mhf = build_filt(wdm,tfilt,t,'MHW'); %build a Mexihat filter for each width
shf = build_filt(wds,tfilt,t,'SH'); % build a shannon filter

%Applying the different filters to the signal data
%[sgmf, stgmf, stgmf_sft] = filter_data(sr,mhf);
[sgsf, stgsf, stgsf_sft] = filter_data(sr,shf);

%plot the spectrogram
figure(14)
pcolor(tfilt,ksr([nr/2+5000:nr/2+20000]),stgsf_sft(:,[nr/2+5000:nr/2+20000],1).'), shading interp
set(gca,'Fontsize',[14])
colormap(hot)
title('Shannon Filter width = 0.05')
xlabel('Time (s)')
ylabel('Frequency (k)')
