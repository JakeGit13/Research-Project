function audioFeatures = extractAudioFeaturesTEST(processedAudio, opts)
% EXTRACTAUDIOFEATURESTEST  F0-only extraction (one value per MR frame)
% Requires processedAudio from processAudioFile.m

    arguments
        processedAudio struct
        opts.VERBOSE (1,1) logical = false
        opts.Method  (1,:) char    = 'NCF'     % 'NCF' | 'ACF' | 'CEP' | 'SRH'
        opts.Range   (1,2) double   = [50 300] % Hz
        opts.UnvoicedValue (1,1) double = 0    % use 0 (or NaN) for unvoiced
    end

     fprintf("Running...\n");


    % --- pull from preprocessing struct ---

    frameSnippets      = processedAudio.frameSnippets;              % [winLen x nFrames]
    sampleRate     = processedAudio.sampleRate;
    winLen = processedAudio.windowLengthSamples;
    nFrames     = processedAudio.nFrames;

    if opts.VERBOSE
        fprintf('[F0] sampleRate=%g Hz | winLen=%d samples (~%.1f ms) | frames=%d\n', ...
            sampleRate, winLen, 1000*winLen/sampleRate, nFrames);
    end

    % --- one F0 per frame ---
    f0   = zeros(nFrames,1);
    mask = false(nFrames,1);

    for k = 1:nFrames
        seg = frameSnippets(:,k);

        % One estimate per frame: window = full snippet; no overlap; no median smoothing
        f0k = pitch(seg, sampleRate, ...
            'Method', opts.Method, ...
            'Range',  opts.Range, ...
            'WindowLength', winLen, ...
            'OverlapLength', 0, ...
            'MedianFilterLength', 1);

        if ~isempty(f0k) && f0k(1) > 0
            f0(k)   = f0k(1);
            mask(k) = true;
        else
            f0(k)   = opts.UnvoicedValue;  % 0 (default) or NaN
            mask(k) = false;
        end
    end

    if opts.VERBOSE
        fprintf('[F0] voiced frames: %d/%d (%.1f%%)\n', sum(mask), nFrames, 100*sum(mask)/nFrames);
    end

    

    % --- return struct ---
    audioFeatures.type                 = 'F0_only';
    audioFeatures.f0Hz                = f0;                          % nFrames×1
    audioFeatures.voicedMask          = mask;                        % nFrames×1 (logical)
    audioFeatures.centerTimeSec       = processedAudio.centerTimeSec;
    audioFeatures.sampleRate          = sampleRate;
    audioFeatures.windowLengthSamples = winLen;
    audioFeatures.nFrames             = nFrames;


    


    fprintf("DONE");
end


function plotF0Results(processedAudio, audioF0, audioVector)
    sr     = processedAudio.sampleRate;
    t      = (0:numel(audioVector)-1)/sr;
    f0Hz   = audioF0.f0Hz;
    ctrT   = processedAudio.centerTimeSec;

    figure('Color','w');
    subplot(2,1,1);
    plot(t, audioVector, 'k'); hold on; grid on;
    xline(ctrT, ':', 'Color', [0.6 0.6 0.6]);
    title('Waveform with MR frame centers');
    xlabel('Time (s)'); ylabel('Amplitude');

    subplot(2,1,2);
    plot(ctrT, f0Hz, 'o-b','MarkerFaceColor','b');
    grid on; ylim([0 400]); % typical speech pitch range
    title('Extracted F0 per MR frame');
    xlabel('Time (s)'); ylabel('F0 (Hz)');
end




function printF0Diagnostics(processedAudio, audioF0)
    sr   = processedAudio.sampleRate;
    cT   = processedAudio.centerTimeSec(:);
    dt   = diff(cT);
    dur  = processedAudio.durationSec;
    nF   = processedAudio.nFrames;
    half = (processedAudio.windowLengthSamples-1)/(2*sr);
    f0   = audioF0.f0Hz(:);
    voiced = audioF0.voicedMask(:);

    fprintf('--- Alignment diagnostics ---\n');
    fprintf('Duration (s)            : %.3f\n', dur);
    fprintf('Frames (n)              : %d\n', nF);
    fprintf('Mean inter-center (ms)  : %.2f ± %.2f  [min=%.2f, max=%.2f]\n', ...
        1000*mean(dt), 1000*std(dt), 1000*min(dt), 1000*max(dt));
    fprintf('Expected interval (ms)  : %.2f\n', 1000*(dur/nF));
    fprintf('First center margin (ms): start=%.2f before/after 0\n', 1000*(cT(1)-half));
    fprintf('Last center margin (ms) : end=%.2f before/after dur\n', 1000*(dur-(cT(end)+half)));

    fprintf('\n--- F0 diagnostics ---\n');
    fprintf('Voiced frames           : %d/%d (%.1f%%)\n', sum(voiced), nF, 100*mean(voiced));
    if any(voiced)
        f0v = f0(voiced);
        pct = prctile(f0v,[5 25 50 75 95]);
        fprintf('F0 (Hz) voiced          : min=%.1f, max=%.1f, mean=%.1f, median=%.1f\n', ...
            min(f0v), max(f0v), mean(f0v), median(f0v));
        fprintf('F0 percentiles (Hz)     : p5=%.1f, p25=%.1f, p50=%.1f, p75=%.1f, p95=%.1f\n', pct);
    end
end








%% PIPELINE SIM 

clc;

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

    preProcessedAudioStruct = processAudio(wavPath, nFrames, VERBOSE = true);

    audioFeatureStruct = extractAudioFeaturesTEST(preProcessedAudioStruct,VERBOSE = true);

    % plotF0Results(processedAudioStruct, audioFeatureStruct, cleanAudio);

    % After you get processedAudio and audioF0:
    % printF0Diagnostics(preProcessedAudioStruct, audioFeatureStruct);




end
