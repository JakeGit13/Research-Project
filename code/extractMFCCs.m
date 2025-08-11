function mfccFeatures = extractMFCCs(lpCleanAudio, lpFs, targetFPS)

    %% Fixed parameters
    numCoeffs = 13;      % Standard for speech
    numFilters = 26;     % Standard mel filterbank size
    windowSize = 0.025;  % 25ms windows
    hopSize = 0.010;     % 10ms hop
    
    %% Extract MFCCs at original rate (~100 fps)
    % Convert time to samples
    winLength = round(lpFs * windowSize);
    hopLength = round(lpFs * hopSize);
    
    % 1. Create the analysis window vector
    analysisWindow = hamming(winLength, 'periodic');
    
    % 2. Call mfcc() with the new 'Window' parameter
    [mfccWithEnergy, ~, ~] = mfcc(lpCleanAudio, lpFs, ...
        'NumCoeffs', numCoeffs, ...
        'Window', analysisWindow, ...
        'OverlapLength', winLength - hopLength);
        
    % After extracting original MFCCs (around line 25):
    figure('Name', 'MFCC Features');
    
    % Plot original MFCCs
    subplot(2,1,1);
    imagesc(mfccWithEnergy');
    axis xy;
    colorbar;
    xlabel('Frame Index (~100 fps)');
    ylabel('MFCC Coefficient');
    title('MFCCs');
    
    %% Pool to 16 fps
    numFramesOriginal = size(mfccWithEnergy, 1); % <-- ADD THIS LINE

    audioDuration = length(lpCleanAudio) / lpFs;
    numFramesTarget = floor(audioDuration * targetFPS);
    
    % Time stamps for original and target frame rates
    timeOriginal = (0:numFramesOriginal-1) * hopSize;
    
    % Pool by averaging frames within each target window
    pooledMFCCs = zeros(numFramesTarget, 14);

    % Plot pooled MFCCs
    subplot(2,1,2);
    imagesc(pooledMFCCs');
    axis xy;
    colorbar;
    xlabel(['Frame Index (' num2str(targetFPS) ' fps)']);
    ylabel('MFCC Coefficient');
    title('Pooled MFCCs');
    
    for i = 1:numFramesTarget
        % Time window for this target frame
        tStart = (i-1) / targetFPS;
        tEnd = i / targetFPS;
        
        % Find original frames in this window
        idx = find(timeOriginal >= tStart & timeOriginal < tEnd);
        
        if ~isempty(idx)
            pooledMFCCs(i, :) = mean(mfccWithEnergy(idx, :), 1);
        elseif i > 1
            % If no frames in window, copy previous
            pooledMFCCs(i, :) = pooledMFCCs(i-1, :);
        end
    end
    
    %% Calculate velocity (delta) features
    velocity = zeros(size(pooledMFCCs));
    for t = 2:numFramesTarget-1
        velocity(t, :) = (pooledMFCCs(t+1, :) - pooledMFCCs(t-1, :)) / 2;
    end
    % Handle edges
    velocity(1, :) = velocity(2, :);
    velocity(end, :) = velocity(end-1, :);
    
    %% Calculate acceleration (delta-delta) features
    acceleration = zeros(size(pooledMFCCs));
    for t = 2:numFramesTarget-1
        acceleration(t, :) = (velocity(t+1, :) - velocity(t-1, :)) / 2;
    end
    % Handle edges
    acceleration(1, :) = acceleration(2, :);
    acceleration(end, :) = acceleration(end-1, :);
    
    %% Combine all features and transpose
    % Concatenate static + velocity + acceleration
    % Transpose to match expected format (features × frames)
    mfccFeatures = [pooledMFCCs'; velocity'; acceleration'];
    
    %% Print summary
    fprintf('MFCC extraction complete:\n');
    fprintf('  Audio duration: %.1f seconds\n', audioDuration);
    fprintf('  Static features: %d (13 MFCCs + energy)\n', 14);
    fprintf('  Total features: %d (with velocity/acceleration)\n', size(mfccFeatures, 1));
    fprintf('  Output frames: %d at 16 fps\n', numFramesTarget);
end