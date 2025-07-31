% simple_audio_demo.m
% Just play audio before and after processing

clear; close all; clc;

% Load audio
audioPath = 'C:\Users\jaker\Research-Project\data\Audio\New Audio\1_DSP_OUT_2025-07-07,17;23;37_(Vocals).wav';
[Y, FS] = audioread(audioPath);
audio = Y(:,2);  % Use channel 2

% Pass to process audio
[lpAudio, lpFs] = processAudio(Y, FS);

playbackLength = 10;

% Play before 
fprintf('Playing before processing...\n');
soundsc(Y(1:min(playbackLength*FS, end), 2), FS);
pause(playbackLength);

% Play after 
fprintf('Playing after processing...\n');
soundsc(lpAudio(1:min(playbackLength*lpFs, end)), lpFs);
pause(playbackLength);


%% Extract MFCCs (modified to return both original and pooled)
% Quick hack - copy the MFCC extraction code but return both versions
lpFs = FS;  % Since we're not downsampling audio anymore
numCoeffs = 13;
windowSize = 0.025;
hopSize = 0.010;
targetFPS = 45;

winLength = round(lpFs * windowSize);
hopLength = round(lpFs * hopSize);
analysisWindow = hamming(winLength, 'periodic');

[mfccOriginal, ~, ~] = mfcc(audio, lpFs, ...
    'NumCoeffs', numCoeffs, ...
    'Window', analysisWindow, ...
    'OverlapLength', winLength - hopLength);

% Now extract using your function to get pooled version
mfccPooled = extractMFCCs(audio, lpFs);

%% Visualize MFCCs
figure('Position', [100 100 1200 600]);

% Original MFCCs (~100 fps)
subplot(2,2,1);
imagesc(mfccOriginal');
colorbar;
title('MFCCs at Original Rate (~100 fps)');
xlabel('Time (frames)');
ylabel('MFCC Coefficient');
colormap('jet');

% Pooled MFCCs (16 fps)  
subplot(2,2,2);
imagesc(mfccPooled(1:14,:));  % Just static features
colorbar;
title('MFCCs after Pooling (45 fps)');
xlabel('Time (frames)');
ylabel('MFCC Coefficient');

%% Extract and visualize spectrograms
% Original spectrogram
winLength = round(lpFs * 0.025);
hopLength = round(lpFs * 0.010);
nfft = 2^nextpow2(winLength);
[S, F, T] = spectrogram(audio, hamming(winLength), ...
                       winLength - hopLength, nfft, lpFs);
specPower = 10*log10(abs(S).^2 + 1e-10);

% Plot original spectrogram
subplot(2,2,3);
imagesc(T, F/1000, specPower);
axis xy;
colorbar;
title('Spectrogram at Original Rate');
xlabel('Time (s)');
ylabel('Frequency (kHz)');
ylim([0 8]);

% Get pooled spectrogram features
spectrogramPooled = extractSpectrograms(audio, lpFs);

% Plot pooled features
subplot(2,2,4);
imagesc(spectrogramPooled(1:14,:));  % Show first 14 features (PCA reduced)
colorbar;
title('Spectrogram Features after Pooling (45 fps, PCA)');
xlabel('Time (frames)');
ylabel('Feature');

sgtitle('Feature Extraction: Before and After Temporal Alignment');



fprintf('Done!\n');