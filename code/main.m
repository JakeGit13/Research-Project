clear all; clc;


% Paths (match H1/H2 expectation that data file is on MATLAB path)
projectRoot = '/Users/jaker/Research-Project/data';      % use your real root
addpath(projectRoot);

audioFolderPath      = fullfile(projectRoot, 'Audio', 'Raw Audio');


% Load MR/Video once to obtain per-item frame counts
S = load(fullfile(projectRoot, 'mrAndVideoData.mat'), 'data');   % provides 'data'
data = S.data;                                                    % fields used by H1/H2 too. :contentReference[oaicite:4]{index=4}

% Correct map between MR/Video and Audio files [dataIdx, filename]
manifest = {
     9,  'sub8_sen_256_6_svtimriMANUAL.wav';
     1,  'sub8_sen_252_18_svtimriMANUAL.wav';
     5,  'sub1_sen_252_1_svtimriMANUAL.wav';
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


for i = 1:size(manifest,1)

    fprintf("Run %d/%d\n",i,size(manifest,1));
    dataIdx = manifest{i,1};
    wavName = manifest{i,2};
    wavPath = fullfile(audioFolderPath, wavName); % concatenate both parts to make correct wav path
    nFrames = numel(data(dataIdx).video_frames);  % frame count per item

    audioFeatures = extractAudioFeatures(wavPath, nFrames);  % [features × frames] 


    
    
    r1 = trimodalH1(data, audioFeatures, dataIdx,"VERBOSE",false);          % returns results for H1

    % Short H1 summary print
    fprintf('H1 dataIdx=%d | VAF=%.1f%% | 95%% CI=[%.1f%%, %.1f%%] | p=%.3g | vecR=%.3f (p=%.3g)\n\n', ...
    dataIdx, ...
    100*r1.h1_VAF_real, ...
    100*r1.h1_eval_VAF_ci(1), 100*r1.h1_eval_VAF_ci(2), ...
    r1.h1_eval_VAF_p, ...
    r1.h1_vecR_real, r1.h1_vecR_p);

    %{
    

    r2 = trimodalH2(data, audioFeatures, dataIdx, "VERBOSE",false);          % returns results for H2
    % Short H2 summary print
    
    fprintf('H2 target=%d | R=%.3f (p=%.3g) | slope=%.3f (p=%.3g) | SSE=%.3e (p=%.3g) | ΔR=%.4f, ΔSSE=%.3e | T=%d\n', ...
    r2.h2_reconstructId, ...
    r2.h2_tri.R, r2.h2_p.R, ...
    r2.h2_tri.slope, r2.h2_p.slope, ...
    r2.h2_tri.SSE, r2.h2_p.SSE, ...
    r2.h2_delta.dR, r2.h2_delta.dSSE, ...
    r2.h2_T);
    %}
end



