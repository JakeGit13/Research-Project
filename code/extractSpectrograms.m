function spectrogramFeatures = extractSpectrograms(lpCleanAudio, lpFs, targetFPS)

    %% Fixed optimal parameters
    windowSize = 0.025;  % 25ms (in seconds)
    hopSize = 0.010;     % 10ms (in seconds)
    maxFreq = 8000;      % Hz
    
    %% Compute spectrogram
    winLength = round(lpFs * windowSize);
    hopLength = round(lpFs * hopSize);
    nfft = 2^nextpow2(winLength);
    
    % Generate spectrogram
    [S, F, T] = spectrogram(lpCleanAudio, hamming(winLength), ...
                           winLength - hopLength, nfft, lpFs);
    
    % Convert to log scale (what worked best)
    specPower = abs(S).^2;
    logSpec = 10*log10(specPower + 1e-10);


    
    %% Keep only 0-8000 Hz
    freqIdx = F <= maxFreq;
    logSpec = logSpec(freqIdx, :);


    % After computing spectrogram
    figure('Name', 'Spectrogram Features');
    
    % Plot original spectrogram
    subplot(2,1,1);
    imagesc(T, F(freqIdx)/1000, logSpec);
    axis xy;
    colorbar;
    xlabel('Time (s)');
    ylabel('Frequency (kHz)');
    title('Spectrogram');
    colormap('jet');
    
    %% Pool to target frame rate fps
    audioDuration = length(lpCleanAudio) / lpFs;
    numFrames = floor(audioDuration * targetFPS);
    frameTimestamps = T;
    pooledSpec = zeros(size(logSpec,1), numFrames);

    % Plot pooled spectrogram
    subplot(2,1,2);
    timeVector = (0:numFrames-1) / targetFPS;
    imagesc(timeVector, F(freqIdx)/1000, pooledSpec);
    axis xy;
    colorbar;
    xlabel('Time (s)');
    ylabel('Frequency (kHz)');
    title(['Pooled Spectrogram (' num2str(targetFPS) ' fps)']);
    colormap('jet');
    
    for i = 1:numFrames
        t_start = (i-1) / targetFPS;
        t_end = t_start + 1/targetFPS;
        idx = (frameTimestamps >= t_start) & (frameTimestamps < t_end);
        
        if any(idx)
            % Mean pooling to be consistent with MFCC 
            pooledSpec(:,i) = mean(logSpec(:,idx), 2);
            
        else
            if i > 1
                pooledSpec(:,i) = pooledSpec(:,i-1);
            end
        end
    end

    %pooledSpec = pca(pooledSpec', 'NumComponents', 14)';  % Reduce to 14 features

    
    %% Add velocity and acceleration
    % Velocity (how features change)
    velocity = zeros(size(pooledSpec));
    for t = 2:size(pooledSpec,2)-1
        velocity(:,t) = (pooledSpec(:,t+1) - pooledSpec(:,t-1)) / 2;
    end
    velocity(:,1) = velocity(:,2);
    velocity(:,end) = velocity(:,end-1);
    
    % Acceleration (how fast they change)
    acceleration = zeros(size(pooledSpec));
    for t = 2:size(velocity,2)-1
        acceleration(:,t) = (velocity(:,t+1) - velocity(:,t-1)) / 2;
    end
    acceleration(:,1) = acceleration(:,2);
    acceleration(:,end) = acceleration(:,end-1);
    
    % Combine all features
    spectrogramFeatures = [pooledSpec; velocity; acceleration];
    
    %% Print summary
    fprintf('Spectrogram extraction complete:\n');
    fprintf('  Audio: %.1f seconds at %d Hz\n', audioDuration, lpFs);
    fprintf('  Static features: %d frequencies\n', size(pooledSpec,1));
    fprintf('  Total features: %d (with velocity/acceleration)\n', size(spectrogramFeatures,1));
    fprintf('  Output frames: %d at 16 fps\n', numFrames);
end