clear all; clc;
% Paths =====
projectRoot = '/Users/jaker/Research-Project/data';      
addpath(projectRoot);

audioFolderPath      = fullfile(projectRoot, 'Audio', 'Raw Audio');
S = load(fullfile(projectRoot, 'mrAndVideoData.mat'), 'data');   % provides 'data'
data = S.data;

% === Output folders (H1) ===
resultsRoot = fullfile(projectRoot, 'results');
h1Root      = fullfile(resultsRoot, 'H1');           
if ~exist(h1Root, 'dir'); mkdir(h1Root); end




% Controls ====
nBoots = 100;

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
    wavPath = fullfile(audioFolderPath, wavName);
    nFrames = size(data(dataIdx).mr_warp2D, 2);

    actorID    = data(dataIdx).actor;
    sentenceID = data(dataIdx).sentence;

    itemFolder = fullfile(h1Root, sprintf('s%03d_a%02d', sentenceID, actorID));
if ~exist(itemFolder,'dir'); mkdir(itemFolder); end

    audioFeatures = extractAudioFeatures(wavPath,nFrames);

% Run H1 (minimum test)
r1 = trimodalH1(data, audioFeatures, dataIdx, ...
                'reconstructId', 3, ...
                'shuffleTarget', 1, ...
                'nBoots', nBoots);

% Slim meta: only keep essentials
meta = struct( ...
  'actorID',        actorID, ...
  'sentenceID',     sentenceID, ...
  'dataIdx',        dataIdx, ...
  'observedMode',   'MR+VID', ...   % or 'MR' / 'VID'
  'reconstructId',  3,        ...   % H1: 3 = Audio
  'shuffleTarget',  1,        ...   % 1 = MR (refit-null)
  'nBoots',         nBoots,   ...
  'rngSeed',        rngSeed,  ...
  'timestamp',      datetime('now','Format','yyyy-MM-dd_HH:mm:ss'));


outPath = fullfile(itemFolder, 'H1_MR+VID.mat');
save(outPath, 'r1', 'meta', '-v7.3');

end




