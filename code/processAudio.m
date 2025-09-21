function audioFeatures = processAudio(wavPath, nFrames, opts)   
    
    % Default arguments 
    arguments
        wavPath (1,:) char
        nFrames (1,1) double 
        opts.VERBOSE (1,1) logical = false  % false as default
        opts.PLOTTING (1,1) logical = false  % false as default
    end
    
    VERBOSE = opts.VERBOSE;   

    clc;   


    %% Load audio & remove background noise (SHOULD PLOT SOMETHING HERE)
    [originalAudio, sampleRate] = audioread(wavPath);
    % Simple scanner-noise cancellation by channel subtraction 
    cleanAudio = originalAudio(:,2) - originalAudio(:,1);
    cleanAudio = cleanAudio(:);
    cleanAudio = cleanAudio - mean(cleanAudio);   % remove DC offset DOUBLE CHECK THAT'S OKAY 


    %% Preprocessing ===================================

    % Basic info
    nSamples          = numel(cleanAudio);
    audioDurationSec  = nSamples / sampleRate;

    if VERBOSE;
        fprintf('Audio duration (from file) : %.3f s\n', audioDurationSec);
        fprintf('nFrames (MR)               : %d\n', nFrames);
        fprintf('Interval per frame         : %.3f s\n\n', audioDurationSec / nFrames);
    end

    
   % Step size in SAMPLES for an even split across the audio
   samplesPerFrame = nSamples / nFrames;

    if VERBOSE
        fprintf('samplesPerFrame           : %.2f samples/frame\n', samplesPerFrame);
        fprintf('interval per frame        : %.3f s\n\n', audioDurationSec / nFrames);
    end

    
    % Centers in SAMPLES (half-step offset), then clamp to 1..nSamples
    centerSample = (0:nFrames-1) * samplesPerFrame + samplesPerFrame/2;
    centerSample = round(centerSample) + 1;                % 1-based indexing
    centerSample = max(1, min(centerSample, nSamples));     % clamp to bounds
    
    
    intervalMs  = 1000 * (audioDurationSec / nFrames);
    windowMs    = min(60, max(40, intervalMs));
    windowLen   = round(windowMs/1000 * sampleRate);
    if mod(windowLen,2)==0, windowLen = windowLen + 1; end   % make odd length
    halfWindow  = (windowLen - 1)/2;
    if VERBOSE
        fprintf('Chosen window             : %d samples (%.1f ms). Interval=%.1f ms\n', ...
            windowLen, windowMs, intervalMs);
    end

    % Hann taper and zero-padding so edge frames are handled cleanly
    hannWindow      = hann(windowLen, 'symmetric');
    paddedAudio     = [zeros(halfWindow,1); audioForProcessing; zeros(halfWindow,1)];
    centerIdxPadded = centerSampleIdx + halfWindow;












    

    
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


% Pick one
% Find the num frames of that MR and then pass both in 


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

    processAudio(wavPath, nFrames, VERBOSE = true);

end

