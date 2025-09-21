function processedAudio = processAudio(wavPath, nFrames, opts)   
    
    % Default arguments 
    arguments
        wavPath (1,:) char
        nFrames (1,1) double 
        opts.VERBOSE (1,1) logical = false  % false as default
        opts.PLOTTING (1,1) logical = false  % false as default
    end
    
    VERBOSE = opts.VERBOSE;   
    PLOTTING = opts.PLOTTING;
 

    %% Load audio & remove background noise (SHOULD PLOT SOMETHING HERE)
    [originalAudio, sampleRate] = audioread(wavPath);
    % Simple scanner-noise cancellation by channel subtraction 
    cleanAudio = originalAudio(:,2) - originalAudio(:,1);
    cleanAudio = cleanAudio(:);
    cleanAudio = cleanAudio - mean(cleanAudio);   % remove DC offset 


    

    %% Preprocessing ===================================
    
    % Basic info
    nSamples         = numel(cleanAudio);
    audioDurationSec = nSamples / sampleRate;
    
    if VERBOSE
        fprintf('Audio duration     : %.3f s\n', audioDurationSec);
        fprintf('nFrames (MR)               : %d\n', nFrames);
    end
    
    %% Fixed 50 ms frame-centred snippets (even split across audio)
    
    % Even split in SAMPLES and centre locations (half-step offset)
    samplesPerFrame = nSamples / nFrames;                          % samples per MR frame
    centerSampleIdx = (0:nFrames-1) * samplesPerFrame + samplesPerFrame/2;
    centerSampleIdx = round(centerSampleIdx) + 1;                  % 1-based indexing
    centerSampleIdx = max(1, min(centerSampleIdx, nSamples));      % clamp to [1, nSamples]
    
    % Fixed 50 ms Hann window (force odd length so we have a true centre sample)
    windowLen  = round(0.050 * sampleRate);
    if mod(windowLen,2)==0, windowLen = windowLen + 1; end
    halfWindow = (windowLen - 1)/2;
    hannWindow = hann(windowLen, 'symmetric');
    
    if VERBOSE
        fprintf('Window length              : %d samples (%.1f ms)\n', ...
            windowLen, 1000*windowLen/sampleRate);
    end
    
    % Zero-pad to handle edge frames 
    paddedAudio     = [zeros(halfWindow,1); cleanAudio; zeros(halfWindow,1)];
    centerIdxPadded = centerSampleIdx + halfWindow;
    
    % Extract one windowed snippet per MR frame (columns = frames)
    frameSnippets = zeros(windowLen, nFrames, 'like', cleanAudio);
    for k = 1:nFrames
        iLo = centerIdxPadded(k) - halfWindow;
        iHi = centerIdxPadded(k) + halfWindow;
        frameSnippets(:,k) = paddedAudio(iLo:iHi) .* hannWindow;
    end
    
    fprintf('Extracted %d snippets of %d samples each (~%.1f ms)\n', ...
        nFrames, windowLen, 1000*windowLen/sampleRate);



    % ---- Return struct for feature extraction ---- 
    processedAudio.sampleRate          = sampleRate;
    processedAudio.windowLengthSamples = windowLen;
    processedAudio.centerSampleIdx     = centerSampleIdx(:);   % nFrames×1
    processedAudio.frameSnippets       = frameSnippets;        % winLen×nFrames
    processedAudio.nFrames             = nFrames;
    processedAudio.durationSec         = audioDurationSec;
    processedAudio.windowMs      = 1000 * processedAudio.windowLengthSamples / processedAudio.sampleRate;
    processedAudio.centerTimeSec = (processedAudio.centerSampleIdx - 1) / processedAudio.sampleRate;



end






