% At the top
load(sprintf('sub%d_sen%d_cleaned.mat', subNum, senNum));


%% Parameters for spectrogram
frameDuration = 0.025;  % 25ms
hopDuration = 0.01;     % 10ms
winLength = round(lpFs * frameDuration);
hopLength = round(lpFs * hopDuration);
nfft = 512;

% Generate high-resolution spectrogram
[S, F, T] = spectrogram(lpCleanAudio, hamming(winLength), winLength - hopLength, nfft, lpFs);
specPower = abs(S).^2;               % Power spectrum
logSpec = 10*log10(specPower + 1e-10);  % Log scale (dB)

% Pooled to 16fps
targetFPS = 16;
audioDuration = length(lpCleanAudio) / lpFs;
numFrames = floor(audioDuration * targetFPS);
frameTimestamps = linspace(0, audioDuration, length(T));
pooledSpec = zeros(size(logSpec,1), numFrames);

for i = 1:numFrames
    t_start = (i-1) / targetFPS;
    t_end = t_start + 1/targetFPS;
    idx = (frameTimestamps >= t_start) & (frameTimestamps < t_end);
    if any(idx)
        pooledSpec(:,i) = mean(logSpec(:,idx), 2);
    else
        if i > 1
            pooledSpec(:,i) = pooledSpec(:,i-1);
        else
            pooledSpec(:,i) = zeros(size(logSpec,1),1);
        end
    end
end

%% Plot spectrograms: before and after pooling
figure('Name', 'Spectrogram Comparison (High-res vs Pooled)', ...
    'Position', [100, 100, 1200, 600]);

subplot(1,2,1);
imagesc(T, F, logSpec); axis xy; colormap jet;
xlabel('Time (s)'); ylabel('Frequency (Hz)');
title('High-Resolution Spectrogram');
colorbar;

subplot(1,2,2);
imagesc((0:numFrames-1)/targetFPS, F, pooledSpec); axis xy; colormap jet;
xlabel('Time (s)'); ylabel('Frequency (Hz)');
title('Pooled Spectrogram (16fps)');
colorbar;
