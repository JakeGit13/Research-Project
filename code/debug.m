% Add description
clear
subNum = 8; % which subject number for the audio
senNum = 252; % which sentence number for the audio
% Where are the audio files stored? [CHANGE THIS]
audioFolder = '/Users/jaker/Research-Project/data/audio';
% Navigate to the audio folder so we can use dir with a wildcard
cd(audioFolder)
thisFile = dir(['sub' num2str(subNum) '_sen_' num2str(senNum) '*']);
if length(thisFile)>1
    error('File name is not unique - more than one file has been found');
elseif length(thisFile)<1
    error('File not found: check the subject and sentence numbers')
end
% Pull in the audio data
[Y,FS] = audioread(thisFile.name);
% Clean Audio through subtraction 
cleanAudio = Y(:,2)-Y(:,1);
%% Basic MFCC extraction
% First, let's just see what MFCCs look like with default settings
% Extract MFCCs using Audio Toolbox
[coeffs, delta, deltaDelta, loc] = mfcc(cleanAudio, FS);
% Check the dimensions
fprintf('MFCC dimensions: %d frames x %d coefficients\n', size(coeffs,1), size(coeffs,2));
fprintf('Audio duration: %.2f seconds\n', length(cleanAudio)/FS);
fprintf('Frame rate with defaults: %.2f fps\n', size(coeffs,1)/(length(cleanAudio)/FS));
% Visualize the MFCCs
figure;
imagesc(coeffs');
xlabel('Frame Number');
ylabel('MFCC Coefficient');
title('MFCCs with Default Settings');
colorbar;
colormap('jet');
%% Step 1b: Understand the MFCC output
% Print what we got from the default settings
fprintf('\n=== MFCC Analysis ===\n');
fprintf('MFCC dimensions: %d frames x %d coefficients\n', size(coeffs,1), size(coeffs,2));
fprintf('Audio duration: %.2f seconds\n', length(cleanAudio)/FS);
fprintf('Frame rate with defaults: %.2f fps\n', size(coeffs,1)/(length(cleanAudio)/FS));
fprintf('Target frame rate: 16 fps\n');
% Look at coefficient statistics
fprintf('\n=== Coefficient Ranges ===\n');
for i = 1:5  % Just first 5 coefficients
    fprintf('Coefficient %d: min=%.2f, max=%.2f, mean=%.2f\n', ...
        i, min(coeffs(:,i)), max(coeffs(:,i)), mean(coeffs(:,i)));
end
% Plot individual coefficients over time to see speech patterns
figure;
subplot(3,1,1);
plot(coeffs(:,1));
title('Coefficient 1 (Overall Energy)');
ylabel('Value');
subplot(3,1,2);
plot(coeffs(:,2));
title('Coefficient 2 (Spectral Tilt)');
ylabel('Value');
subplot(3,1,3);
plot(coeffs(:,5));
title('Coefficient 5 (Mid-level Detail)');
xlabel('Frame Number');
ylabel('Value');
%% Step 2: Adjust parameters for 16 fps
% Calculate window parameters for 16 fps
targetFPS = 16;
windowDuration = 1/targetFPS;  % 62.5 ms per frame
windowLength = round(windowDuration * FS);  % samples per window
overlapLength = 0;  % No overlap for simple alignment
fprintf('\n=== 16 fps Parameters ===\n');
fprintf('Window duration: %.1f ms\n', windowDuration*1000);
fprintf('Window length: %d samples\n', windowLength);
fprintf('Expected frames: %d\n', floor(length(cleanAudio)/windowLength));
% Extract MFCCs with 16 fps parameters
[coeffs16, delta16, deltaDelta16, loc16] = mfcc(cleanAudio, FS, ...
    'Window', hamming(windowLength, 'periodic'), ...
    'OverlapLength', overlapLength);
% Verify the frame rate
actualFrames = size(coeffs16, 1);
actualFPS = actualFrames / (length(cleanAudio)/FS);
fprintf('\n=== Results at 16 fps ===\n');
fprintf('Actual frames: %d\n', actualFrames);
fprintf('Actual frame rate: %.2f fps\n', actualFPS);
% Visualize the difference
figure;
subplot(2,1,1);
imagesc(coeffs');
title(sprintf('Original: %d frames (%.1f fps)', size(coeffs,1), 98.68));
xlabel('Frame Number');
ylabel('MFCC Coefficient');
subplot(2,1,2);
imagesc(coeffs16');
title(sprintf('Adjusted: %d frames (%.1f fps)', actualFrames, actualFPS));
xlabel('Frame Number');
ylabel('MFCC Coefficient');
colormap('jet');