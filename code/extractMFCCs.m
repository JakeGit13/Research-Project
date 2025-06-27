%% extractMFCCs.m
% Extract Mel-Frequency Cepstral Coefficients (MFCCs) from audio data
% 
% Description:
%   This script extracts MFCCs from cleaned audio recordings at 16 fps
%   to match the temporal resolution of MRI data for multimodal analysis.

clear; close all;

%% Configuration
subNum = 8;      % Subject number
senNum = 252;    % Sentence number
targetFPS = 16;  % Target frame rate to match MRI data

% Audio folder path
audioFolder = '/Users/jaker/Research-Project/data/audio';

%% Load and Clean Audio
% Navigate to audio folder
cd(audioFolder)

% Find the specific audio file
thisFile = dir(['sub' num2str(subNum) '_sen_' num2str(senNum) '*']);

% Error checking
if length(thisFile) > 1
    error('Multiple files found for sub%d_sen_%d', subNum, senNum);
elseif isempty(thisFile)
    error('No file found for sub%d_sen_%d', subNum, senNum);
end

% Load audio (2 channels: noise reference and speech+noise)
[Y, FS] = audioread(thisFile.name);

% Clean audio using channel subtraction
cleanAudio = Y(:,2) - Y(:,1);  % Remove common-mode noise

% Display audio info
audioDuration = length(cleanAudio)/FS;
fprintf('Audio file: %s\n', thisFile.name);
fprintf('Duration: %.2f seconds\n', audioDuration);
fprintf('Sampling rate: %d Hz\n', FS);

%% Extract MFCCs at Target Frame Rate (16 fps)
% Calculate window parameters
windowDuration = 1/targetFPS;                    % 62.5 ms per frame
windowLength = round(windowDuration * FS);       % Samples per window
overlapLength = 0;                               % No overlap for direct alignment

% Extract MFCCs
[coeffs16, ~, ~, ~] = mfcc(cleanAudio, FS, ...
    'Window', hamming(windowLength, 'periodic'), ...
    'OverlapLength', overlapLength);

% Verify extraction results
actualFrames = size(coeffs16, 1);
actualFPS = actualFrames / audioDuration;

fprintf('\n=== MFCC Extraction Results ===\n');
fprintf('Target frame rate: %d fps\n', targetFPS);
fprintf('Actual frame rate: %.2f fps\n', actualFPS);
fprintf('Total frames: %d\n', actualFrames);
fprintf('Coefficients per frame: %d\n', size(coeffs16, 2));

%% Generate comparison plot for report
% First extract MFCCs at default settings for comparison
[coeffsDefault, ~, ~, ~] = mfcc(cleanAudio, FS);
defaultFPS = size(coeffsDefault, 1) / audioDuration;

% Create figure
figure('Name', 'MFCC Frame Rate Comparison', 'Position', [100 100 900 600]);

% Time axes for both plots
timeDefault = (0:size(coeffsDefault,1)-1) / defaultFPS;
time16fps = (0:size(coeffs16,1)-1) / targetFPS;

% Top panel: Default frame rate
subplot(2,1,1);
imagesc(timeDefault, 1:13, coeffsDefault(:,2:14)');  % Skip C0 for clarity
xlabel('Time (seconds)');
ylabel('MFCC Coefficient');
title(sprintf('MFCCs at Default Settings: %.1f fps (%d frames)', ...
    defaultFPS, size(coeffsDefault,1)));
colorbar;
set(gca, 'YDir', 'normal');
caxis([-10 10]);  % Consistent color scale

% Bottom panel: 16 fps
subplot(2,1,2);
imagesc(time16fps, 1:13, coeffs16(:,2:14)');  % Skip C0 for clarity
xlabel('Time (seconds)');
ylabel('MFCC Coefficient');
title(sprintf('MFCCs at Target Frame Rate: %.1f fps (%d frames)', ...
    actualFPS, size(coeffs16,1)));
colorbar;
set(gca, 'YDir', 'normal');
caxis([-10 10]);  % Consistent color scale

% Overall title
sgtitle('MFCC Temporal Resolution Adjustment for MRI Alignment', ...
    'FontSize', 14, 'FontWeight', 'bold');

% Use consistent colormap
colormap('jet');

%% Display Coefficient Statistics
fprintf('\n=== Coefficient Statistics ===\n');
fprintf('Coef | Min     | Max     | Mean    | Std\n');
fprintf('-----|---------|---------|---------|--------\n');
for i = 1:min(5, size(coeffs16, 2))
    fprintf(' %2d  | %7.2f | %7.2f | %7.2f | %6.2f\n', ...
        i, min(coeffs16(:,i)), max(coeffs16(:,i)), ...
        mean(coeffs16(:,i)), std(coeffs16(:,i)));
end

