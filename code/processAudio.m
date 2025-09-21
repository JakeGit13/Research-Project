function processedAudio = processAudioFile(wavPath, nFrames, opts)   
    
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

    plotPreprocWaveform(processedAudio, cleanedAudio)  % or originalAudio

    


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





% Corrected map between MR/Video and Audio files [dataIdx, filename]
manifest = {
     9,  'sub8_sen_256_6_svtimriMANUAL.wav';
     1,  'sub8_sen_252_18_svtimriMANUAL.wav';   % Swapped dataIdx with ...
     5,  'sub1_sen_252_1_svtimriMANUAL.wav';    % ... this dataIdx
     6,  'sub8_sen_253_18_svtimriMANUAL.wav';
     7,  'sub8_sen_254_15_svtimriMANUAL.wav';
     8,  'sub8_sen_255_17_svtimriMANUAL.wav';
    10,  'sub8_sen_257_15_svtimriMANUAL.wav';
    11,  'sub8_sen_258_8_svtimriMANUAL.wav';
    12,  'sub8_sen_259_18_svtimriMANUAL.wav';
    13,  'sub8_sen_260_1_svtimriMANUAL.wav';
    14,  'sub8_sen_261_15_svtimriMANUAL.wav';
    18,  'sub14_sen_252_14_svtimriMANUAL.wav'
};






%% Paths =====
projectRoot = '/Users/jaker/Research-Project/data';     % This should be the only thing that the user needs to set up? e.g. path to research project?  
addpath(projectRoot);

% Load in MR and video data struct
mrAndVid = load(fullfile(projectRoot, 'mrAndVideoData.mat'), 'data');   % MR / video data struct 
mrAndVideoData = mrAndVid.data;

% Path to raw audio folder
audioFolderPath = fullfile(projectRoot, 'Audio', 'Raw Audio');     % Where raw audio files are located



manifestLength = size(manifest,1) / 12;

for i = 1:manifestLength

    % Get necessary attributes for sentence i of the manifest 
    dataIdx = manifest{i,1};
    wavName = manifest{i,2};

    % Get direct path to .wav file of sentence i by concatenating 
    wavPath = fullfile(audioFolderPath, wavName);

    nFrames = size(mrAndVideoData(dataIdx).mr_warp2D, 2); % Number of MR / video frames 

    processAudioFile(wavPath, nFrames, VERBOSE = true, PLOTTING = true);

end


