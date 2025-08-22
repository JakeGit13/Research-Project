%% === Audio feature extraction: speech-band, dense pooling ===
% Same cleaning as before; same dataIdx; saves to new filename.
% Produces ~ (numBands*(6 + K) + ~6) features per frame.

clc;

%% Load and clean audio (unchanged)
audioFilePath = 'C:\Users\jaker\Research-Project\data\audio\sub8_sen_256_6_svtimriMANUAL.wav';
[Y, FS] = audioread(audioFilePath);
cleanAudio = Y(:,2) - Y(:,1);

%% Load MR/video to get frame count (unchanged)
load('C:\Users\jaker\Research-Project\data\mrAndVideoData.mat', 'data');
dataIdx = 9;  % sub8_sen_256
nFrames = size(data(dataIdx).mr_warp2D, 2);
fprintf('[INFO] Frames (MRI/video): %d\n', nFrames);

%% High-resolution Mel spectrogram (speech band)
% Use Window (vector) and ensure FFTLength >= window length
winDur = 0.025;                  % 25 ms
hopDur = 0.010;                  % 10 ms
winSamp = round(winDur*FS);
hopSamp = round(hopDur*FS);
winVec  = hamming(winSamp, 'periodic');

% FFT length: power-of-two >= window length
nfft = 2^nextpow2(winSamp);
numBands = 80;                   % 80 as no real data past 8khz? 
freqRange = [50 8000];           % Focus on speech band

[S, Fmel, Tmel] = melSpectrogram(cleanAudio, FS, ...
    'Window', winVec, ...
    'OverlapLength', winSamp - hopSamp, ...
    'FFTLength', nfft, ...
    'NumBands', numBands, ...
    'FrequencyRange', freqRange);

X = log(S + 1e-10);  % log-mel
fprintf('[INFO] Mel spectrogram: %d bands x %d time windows, [%.0f–%.0f] Hz\n', ...
        size(X,1), size(X,2), freqRange(1), freqRange(2));

%% Frame segmentation to match MRI/video
totalDuration = length(cleanAudio) / FS;
frameEdges = linspace(0, totalDuration, nFrames+1);
K = 20;   % # low-frequency temporal DCT coeffs kept per band 

% Total per-frame ≈ numBands*(6+K) + ~6 prosody scalars
D_est = numBands*(6 + K) + 6;
audioFeaturesRich = zeros(nFrames, D_est);
featLengths = zeros(nFrames,1); % track actual length in case K > columns

% Precompute prosody on short hop grid to avoid per-frame re-estimation
% (robust against very short frames)
[f0_all, ~] = pitch(cleanAudio, FS, 'Method','NCF', ...
    'WindowLength', winSamp, 'OverlapLength', winSamp - hopSamp, ...
    'Range',[50 400]);  % typical speech f0 range
logEnergy_all = log(movsum(cleanAudio.^2, winSamp) + 1e-10);
% For spectral centroid/rolloff, we compute per frame on the raw segment.

for i = 1:nFrames
    t0 = frameEdges(i);
    t1 = frameEdges(i+1);

    % Indices in mel time grid
    idx = find(Tmel >= t0 & Tmel < t1);
    Xi = X(:, idx);  % [numBands x Ti]
    if isempty(idx)
        Xi = zeros(numBands,1);
    end

    % --- Stats per band across time
    mu  = mean(Xi, 2);
    sd  = std(Xi, 0, 2);
    mn  = min(Xi, [], 2);
    mx  = max(Xi, [], 2);
    sk  = skewness(Xi, 0, 2);
    ku  = kurtosis(Xi, 0, 2);

    % --- Temporal DCT per band along time (captures within-frame dynamics)
    Ti = size(Xi,2);
    if Ti > 1
        C = dct(Xi, [], 2);                  % [numBands x Ti]
        Kuse = min(K, Ti);
        Ck = C(:, 1:Kuse);                    % [numBands x Kuse]
    else
        Kuse = min(K, 1);
        Ck = zeros(numBands, Kuse);
    end

    % --- Prosody inside this MRI frame (simple, stable set)
    segStart = max(1, floor(t0*FS)+1);
    segEnd   = min(length(cleanAudio), floor(t1*FS));
    frameAudio = cleanAudio(segStart:segEnd);

    % f0 mean/std from global track subset overlapping the frame
    % approximate mapping: indices in f0_all are hop-based; compute times:
    hopTimes = (0:length(f0_all)-1) * (hopSamp/FS);
    idp = find(hopTimes >= t0 & hopTimes < t1);
    if isempty(idp)
        f0mean = 0; f0std = 0;
    else
        f0seg = f0_all(idp);
        f0seg = f0seg(f0seg>0);  % voiced only
        if isempty(f0seg), f0mean = 0; f0std = 0; else
            f0mean = mean(f0seg); f0std = std(f0seg);
        end
    end

    % Log energy in frame
    E = log(sum(frameAudio.^2) + 1e-10);

    % Spectral centroid & rolloff in frame (single-window estimate)
    sc = spectralCentroid(frameAudio, FS, 'Window', winVec, 'OverlapLength', 0);
    sr = spectralRolloffPoint(frameAudio, FS, 'Window', winVec, 'OverlapLength', 0);

    % --- Concatenate
    feat = [mu; sd; mn; mx; sk; ku; Ck(:); f0mean; f0std; E; sc; sr];

    % Store
    D = numel(feat);
    audioFeaturesRich(i,1:D) = feat;
    featLengths(i) = D;
end

% Trim any unused trailing columns (if Kuse < K in early frames)
Dmax = max(featLengths);
audioFeaturesRich = audioFeaturesRich(:,1:Dmax);

fprintf('[INFO] Final per-frame dimensionality: %d features\n', Dmax);
fprintf('[INFO] Final feature matrix: %d frames x %d features\n', size(audioFeaturesRich,1), size(audioFeaturesRich,2));
fprintf('[INFO] Settings -> Bands: %d, DCT per band: %d, Stats per band: 6, FreqRange: [%d %d] Hz\n', ...
        numBands, K, freqRange(1), freqRange(2));

%% Save alongside existing outputs (new filename)
audioData(dataIdx).audioFeatures_richSpeech8k = audioFeaturesRich;
save('C:\Users\jaker\Research-Project\data\audioFeaturesData_richSpeech8k.mat', 'audioData');

filePath = fullfile(pwd, 'audioFeaturesData_richSpeech8k.mat');
fprintf('[INFO] File saved to: %s\n', filePath);
