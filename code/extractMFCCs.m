function pooledMFCCs = extractMFCCs(lpCleanAudio, lpFs)

% Example MFCC extraction with standard audio settings 
frameLength = round(0.025 * lpFs);     % 25 ms window
frameOverlap = round(0.015 * lpFs);    % 10 ms hop

[coeffs, ~, ~, ~] = mfcc(lpCleanAudio, lpFs, ...    % 13 Coefficients  
    'Window', hamming(frameLength, 'periodic'), ...
    'OverlapLength', frameOverlap);

actualFPS = lpFs / (frameLength - frameOverlap);    
fprintf('Extracted MFCCs at approx. %.2f fps (%d frames, %d coeffs per frame)\n', ...
    actualFPS, size(coeffs,1), size(coeffs,2)); % 100 Fps 

%% Plot high-res MFCCs before pooling 
% figure;
% imagesc(coeffs'); axis xy; colormap jet;
% xlabel('Frame Index (~100fps)'); ylabel('MFCC Coefficient');
% title('High-Resolution MFCCs (~100 fps)');

%% Pool MFCCs to match MRI 16fps
targetFPS = 16;
mriFrameDuration = 1 / targetFPS;  % 62.5 ms per MRI frame
audioDuration = length(lpCleanAudio) / lpFs;    

% How many pooled frames
numPooledFrames = floor(audioDuration * targetFPS); % How many 16fps frames in the audio

% How many MFCC frames per MRI frame
mfccTimestamps = linspace(0, audioDuration, size(coeffs,1));    % Time stamp for each MFCC frame 
pooledMFCCs = zeros(numPooledFrames, size(coeffs,2));   

for i = 1:numPooledFrames
    % Define start and end time for this MRI frame window
    t_start = (i-1) * mriFrameDuration;
    t_end   = t_start + mriFrameDuration;

    % Find MFCC frames that fall into this window
    inWindow = (mfccTimestamps >= t_start) & (mfccTimestamps < t_end);  % True / False array 
    
    % Mean pool the MFCCs in this window
    if any(inWindow)    % This runs if at least one MFCC frame is in the window
        pooledMFCCs(i,:) = mean(coeffs(inWindow, :), 1);    % Take mean of MFCC frames that happened during time window
    else
        % If no MFCC frames fall in the current MRI window
        if i > 1
            pooledMFCCs(i,:) = pooledMFCCs(i-1,:);  % Repeat previous frame
        else
            pooledMFCCs(i,:) = zeros(1, size(coeffs,2));  % Zero padding for first frame
        end
    end
end

fprintf('Pooled MFCCs to %d frames at %d fps.\n', size(pooledMFCCs,1), targetFPS);

%% Plot pooled MFCCs 
% figure;
% imagesc(pooledMFCCs'); axis xy; colormap jet;
% xlabel('Frame (16fps)'); ylabel('MFCC Coeff #');
% title('Pooled MFCCs aligned to MRI Frame Rate');

% Transpose to match PCA format [coeffs × frames]
pooledMFCCs = pooledMFCCs';  % Now [13 × numFrames] to match PCA format

end

%% Example Call 

% Audio processing
audioFolder = '/Users/jaker/Research-Project/data/audio';
audioFile = fullfile(audioFolder, 'sub8_sen_258_8_svtimriMANUAL.wav');

% Check if file exists
if ~exist(audioFile, 'file')
    error('Audio file not found: %s', audioFile);
end

[Y, FS] = audioread(audioFile);
[lpCleanAudio, lpFs, ~] = audioPreprocessing(Y, FS);
pooledMFCCs = extractMFCCs(lpCleanAudio, lpFs);