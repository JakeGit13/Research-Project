function pooledMFCCs = extractMFCCs(lpCleanAudio, lpFs)
% extractMFCCs - Extract and pool MFCCs to match MRI frame rate

% Example MFCC extraction with standard audio settings (e.g., 25ms/10ms)
frameLength = round(0.025 * lpFs);     % 25 ms window
frameOverlap = round(0.015 * lpFs);    % 10 ms hop

[coeffs, ~, ~, ~] = mfcc(lpCleanAudio, lpFs, ...
    'Window', hamming(frameLength, 'periodic'), ...
    'OverlapLength', frameOverlap);

actualFPS = lpFs / (frameLength - frameOverlap);
fprintf('Extracted MFCCs at approx. %.2f fps (%d frames, %d coeffs per frame)\n', ...
    actualFPS, size(coeffs,1), size(coeffs,2));

% Pool MFCCs to match MRI 16fps
targetFPS = 16;
mriFrameDuration = 1 / targetFPS;  % 62.5 ms per MRI frame
audioDuration = length(lpCleanAudio) / lpFs;

% How many pooled frames?
numPooledFrames = floor(audioDuration * targetFPS);

% How many MFCC frames per MRI frame?
mfccTimestamps = linspace(0, audioDuration, size(coeffs,1));
pooledMFCCs = zeros(numPooledFrames, size(coeffs,2));

for i = 1:numPooledFrames
    % Define start and end time for this MRI frame window
    t_start = (i-1) * mriFrameDuration;
    t_end   = t_start + mriFrameDuration;

    % Find MFCC frames that fall into this window
    inWindow = (mfccTimestamps >= t_start) & (mfccTimestamps < t_end);
    
    % Pool (mean) the MFCCs in this window
    if any(inWindow)
        pooledMFCCs(i,:) = mean(coeffs(inWindow, :), 1);
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

% Transpose to match PCA format [coeffs × frames]
pooledMFCCs = pooledMFCCs';  % Now [13 × numFrames] to match PCA format
end