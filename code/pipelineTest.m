

%% Load and clean audio
% Using actor 8, sentence 256
audioFilePath = 'C:\Users\jaker\Research-Project\data\audio\sub8_sen_256_6_svtimriMANUAL.wav';
[Y, FS] = audioread(audioFilePath);
cleanAudio = Y(:,2) - Y(:,1);

%% Load corresponding MR/video data to get frame count
load('C:\Users\jaker\Research-Project\data\mrAndVideoData.mat', 'data');
dataIdx = 9;  % This corresponds to sub8_sen_256
nFrames = size(data(dataIdx).mr_warp2D, 2);  % Should be 40 frames
fprintf('Number of frames: %d\n', nFrames);


%% Create frame-aligned spectrogram
% Calculate how much audio per frame
samplesPerFrame = floor(length(cleanAudio) / nFrames);
fprintf('%d samples per frame\n', samplesPerFrame);

% Parameters for spectrogram
windowLength = 512;  % ~11ms at 44.1kHz
overlap = floor(windowLength/2);  % 50% overlap
nfft = windowLength;

% Compute full spectrogram once
[S, F, T] = spectrogram(cleanAudio, hamming(windowLength), overlap, nfft, FS);
S_logmag = log(abs(S) + 1e-10);

fprintf('Full spectrogram: %d frequencies x %d time windows\n', ...
        size(S_logmag, 1), size(S_logmag, 2));
fprintf('Time resolution: %.3f seconds per window\n', T(2)-T(1));

% Now pool to match video frames
timeWindows = size(S_logmag, 2);
windowsPerFrame = floor(timeWindows / nFrames);

audioFeatures = zeros(size(S_logmag, 1), nFrames);
for frame = 1:nFrames
    startWindow = (frame-1) * windowsPerFrame + 1;
    endWindow = min(frame * windowsPerFrame, timeWindows);
    
    % Average the spectrogram windows for this frame
    audioFeatures(:, frame) = mean(S_logmag(:, startWindow:endWindow), 2);
end

fprintf('Pooled to: %d frequencies x %d frames\n', ...
        size(audioFeatures, 1), size(audioFeatures, 2));


% Save processed audio features to .mat file 
audioData(dataIdx).audioFeatures = audioFeatures;  % Keep 2D 
% Save to the data directory:
save('C:\Users\jaker\Research-Project\data\audioFeaturesData.mat', 'audioData');

% Print statement to show the full save path
filePath = fullfile(pwd, 'audioFeaturesData.mat');
fprintf('File saved to: %s\n', filePath);



%{


%% Visualize both spectrograms in the same figure
figure('Name', 'Spectrogram Comparison');

% --- Plot 1: Original Spectrogram (before pooling) ---
subplot(2, 1, 1);
imagesc(T, F, S_logmag); % Use T and F for correct axis scaling
axis xy;
colorbar;
xlabel('Time (s)');
ylabel('Frequency (Hz)');
title('Original Spectrogram (Before Pooling)');

% --- Plot 2: Frame-aligned Spectrogram (after pooling) ---
subplot(2, 1, 2);
imagesc(audioFeatures);
axis xy;
colorbar;
xlabel('Frame Number');
ylabel('Frequency Bin');
title('Frame-aligned Audio Spectrogram (After Pooling)');

%}