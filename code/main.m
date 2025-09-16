clear all; clc;
% Paths =====
projectRoot = '/Users/jaker/Research-Project/data';      
addpath(projectRoot);

audioFolderPath      = fullfile(projectRoot, 'Audio', 'Raw Audio');
S = load(fullfile(projectRoot, 'mrAndVideoData.mat'), 'data');   % provides MR / video data struct
data = S.data;

resultsRoot = fullfile(projectRoot, 'results'); % Where H1 / H2 results save to 

% === Output folders (H1) ===
h1Root      = fullfile(resultsRoot, 'H1');           
if ~exist(h1Root, 'dir'); mkdir(h1Root); end

% === Output folders (H2) ===
h2Root = fullfile(resultsRoot, 'H2');
if ~exist(h2Root, 'dir'); mkdir(h2Root); end


% Controls ====
nBoots = 10;
doSaveH1 = false; 
doH1 = false;
doH2 = false;
doSaveH2 = false; 

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


manifestLength = size(manifest,1);

for i = 1:manifestLength        % Loop through all 12 sentences using manifest

    fprintf("Run %d/%d\n",i,manifestLength);
    dataIdx = manifest{i,1};
    wavName = manifest{i,2};
    wavPath = fullfile(audioFolderPath, wavName);
    nFrames = size(data(dataIdx).mr_warp2D, 2);

    actorID    = data(dataIdx).actor;   % RIGHT FIX THIS
    sentenceID = data(dataIdx).sentence;

    % === H1 output subfolder  ===
    itemFolderH1 = fullfile(h1Root, sprintf('s%03d_a%02d', sentenceID, actorID));
    if ~exist(itemFolderH1,'dir'); mkdir(itemFolderH1); end

    % === H2 output subfolder  ===
    itemFolderH2 = fullfile(h2Root, sprintf('s%03d_a%02d', sentenceID, actorID));
    if ~exist(itemFolderH2,'dir'); mkdir(itemFolderH2); end


    audioFeatures = extractAudioFeatures(wavPath,nFrames);


if doH1

    fprintf("Starting H1\n");

    %% Run H1 (OBSERVER MR+VID) ==============              
    r1 = trimodalH1(data, audioFeatures, ...
                    dataIdx,reconstructId=3, ...
                    shuffleTarget=1, ...
                    observedMode="MR+VID", ...
                    nBoots=nBoots);
    
    
    % META DATA
    meta = struct( ...
      'actorID',        actorID, ...
      'sentenceID',     sentenceID, ...
      'dataIdx',        dataIdx, ...
      'observedMode',   'MR+VID', ...   % or 'MR' / 'VID'
      'reconstructId',  3,        ...   % 3 = Audio
      'shuffleTarget',  1,        ...   % 1 = MR (refit-null)
      'nBoots',         nBoots);
    
    outPath = fullfile(itemFolderH1, 'H1_MR+VID.mat');
    
    if doSaveH1
        save(outPath, 'r1', 'meta', '-v7.3');
    end
    
    
    
    %% Run H1 (OBSERVE MR) ==============
    r1 = trimodalH1(data, audioFeatures, ...
                    dataIdx,reconstructId=3, ...
                    shuffleTarget=1, ...
                    observedMode="MR", ...
                    nBoots=nBoots);
    
    % META DATA
    meta = struct( ...
      'actorID',        actorID, ...
      'sentenceID',     sentenceID, ...
      'dataIdx',        dataIdx, ...
      'observedMode',   'MR', ...   % or 'MR' / 'VID'
      'reconstructId',  3,        ...   % 3 = Audio
      'shuffleTarget',  1,        ...   % 1 = MR (refit-null)
      'nBoots',         nBoots);
    
    outPath = fullfile(itemFolderH1, 'H1_MR.mat');
    
    if doSaveH1
        save(outPath, 'r1', 'meta', '-v7.3');
    end
    
    
    %% Run H1 (OBSERVE VID) ==============
    r1 = trimodalH1(data, audioFeatures, ...
                    dataIdx,reconstructId=3, ...
                    shuffleTarget=2, ...
                    observedMode="VID", ...
                    nBoots=nBoots);
    
    % META DATA
    meta = struct( ...
      'actorID',        actorID, ...
      'sentenceID',     sentenceID, ...
      'dataIdx',        dataIdx, ...
      'observedMode',   'VID', ...   % or 'MR' / 'VID'
      'reconstructId',  3,        ...   % 3 = Audio
      'shuffleTarget',  2,        ...   % 2 = VID
      'nBoots',         nBoots);
    
    outPath = fullfile(itemFolderH1, 'H1_VID.mat');
    
    if doSaveH1
        save(outPath, 'r1', 'meta', '-v7.3');

        fprintf("H1 Complete!\n");
    end

end % if do H1


if doH2
    fprintf("Starting H2\n");

    %% H2: Does adding Audio help reconstruct MR/Video?

    
    % ---- Target = MR (reconstruct MR from Video+Audio) ----
    r2_mr = trimodalH2(data, audioFeatures, dataIdx, ...
                       reconstructId=1, ...   % target MR
                       shuffleTarget=3, ...    % shuffle Audio for null
                       nBoots=nBoots);
    
    metaH2_mr = struct( ...
      'actorID',       actorID, ...
      'sentenceID',    sentenceID, ...
      'dataIdx',       dataIdx, ...
      'reconstructId', 1, ...      % MR
      'shuffleTarget', 3, ...      % Audio
      'wavName',       wavName, ...
      'nBoots',        nBoots);
    
    if doSaveH2
        save(fullfile(itemFolderH2, 'H2_targetMR_shufAUD.mat'), 'r2_mr', 'metaH2_mr', '-v7.3');
    end
    
    % ---- Target = Video (reconstruct Video from MR+Audio) ----
    r2_vid = trimodalH2(data, audioFeatures, dataIdx, ...
                        reconstructId=2, ...   % target Video
                        shuffleTarget=3, ...    % shuffle Audio for null
                        nBoots=nBoots);
    
    metaH2_vid = struct( ...
      'actorID',       actorID, ...
      'sentenceID',    sentenceID, ...
      'dataIdx',       dataIdx, ...
      'reconstructId', 2, ...      % Video
      'shuffleTarget', 3, ...      % Audio
      'wavName',       wavName, ...
      'nBoots',        nBoots);

    fprintf('H2 MR: Tri R=%.3f vs Bi R=%.3f | ΔR=%.3f (p=%.3g)\n', ...
    r2_mr.h2_tri.R, r2_mr.h2_bi.R, r2_mr.h2_delta.dR, r2_mr.h2_delta_p.dR);
    fprintf('H2 VID: Tri R=%.3f vs Bi R=%.3f | ΔR=%.3f (p=%.3g)\n', ...
        r2_vid.h2_tri.R, r2_vid.h2_bi.R, r2_vid.h2_delta.dR, r2_vid.h2_delta_p.dR);

    
    if doSaveH2
        save(fullfile(itemFolderH2, 'H2_targetVID_shufAUD.mat'), 'r2_vid', 'metaH2_vid', '-v7.3');
    end


end



end






