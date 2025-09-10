%% === Trimodal PCA: Compact, Frame-Synchronous Audio Feature Extraction (16 fps) ===
% Author: ChatGPT (generated)
% Purpose: Extract compact, articulator-linked audio features per MR/video frame (T ≈ 40 at 16 fps, ~62.5 ms)
% Notes:
%   • STRICT intra-frame extraction: uses short sub-windows *inside* each MR frame; no cross-frame deltas.
%   • Keep the supervisors' audio cleaning step (R–L channel subtraction).
%   • Block balancing and PCA are done elsewhere; this script only extracts raw features (no per-feature z-scoring).
%   • Includes a VERBOSE toggle and prints progress/details.
%
% Recommended footprint (~110–150 dims/frame by default):
%   - Log-mel energies (40 bands)               : 40
%   - MFCC (13) + within-frame Δ (13 slopes)    : 26
%   - Prosody (voicing fraction, F0, log-energy): 3
%   - Spectral shape (centroid, spread, rolloff, flatness, tilt): 5
%   - Formants (F1,F2,F3) + F2/F1               : 4
%   - (Optional) tiny raw spectrum (32 bins)     : +32
%
% Dependencies: MATLAB Audio Toolbox & Signal Processing Toolbox

clc; clear;

%% ====================== USER OPTIONS / PATHS ======================
opts.verbose            = true;     % Toggle verbose printing
opts.includeRawSpectrum = false;    % Keep OFF for H1
opts.saveVarName        = 'audioData';
opts.saveFieldName      = 'audioFeatures';

% === Hyperparameters to toggle (single-point of control) ===
opts.numMelBands        = 40;       % Toggle: 40  -> 64
opts.numMFCC            = 13;       % Toggle: 13  -> 20
opts.includeFormants    = true;     % Toggle: true -> false (keeps fixed length: zeros if false)

% --- IO paths (edit to match your project; kept consistent with your boilerplate) ---
audioFilePath   = 'C:\\Users\\jaker\\Research-Project\\data\\audio\\sub8_sen_256_6_svtimriMANUAL.wav';
mrVideoDataPath = 'C:\\Users\\jaker\\Research-Project\\data\\mrAndVideoData.mat';
saveMatPath     = 'C:\\Users\\jaker\\Research-Project\\data\\audioFeaturesData.mat';
dataIdx         = 9;  % index of sentence entry in mrAndVideoData.mat (adjust per file)

%% ====================== LOAD DATA & CLEAN AUDIO ======================
[Y, FS] = audioread(audioFilePath);
if size(Y,2) < 2
    error('Expected stereo audio (no mono files per supervisor); got %d channel(s).', size(Y,2));
end
% Supervisors' simple denoising: channel difference
cleanAudio = Y(:,2) - Y(:,1);

if opts.verbose
    fprintf('[INFO] Audio file: %s\n', audioFilePath);
    fprintf('[INFO] Sampling rate (raw): %d Hz | Duration: %.3f s | Channels: %d\n', FS, numel(cleanAudio)/FS, size(Y,2));
end
% --- Resample once for robust features/formants ---
targetFS = 16000;
if FS ~= targetFS
    cleanAudio = resample(cleanAudio, targetFS, FS);
    if opts.verbose
        fprintf('[INFO] Resampled audio to %d Hz (new duration: %.3f s)\n', targetFS, numel(cleanAudio)/targetFS);
    end
    FS = targetFS;
end


% Load MR/video data to get number of frames (T); do NOT hard-code
S = load(mrVideoDataPath, 'data');
if ~isfield(S, 'data'), error('File does not contain variable ''data'': %s', mrVideoDataPath); end
nFrames = size(S.data(dataIdx).mr_warp2D, 2);
if opts.verbose
    fprintf('[INFO] MR/video frames (T): %d (target fps ~16 -> frame ~62.5 ms)\n', nFrames);
end

% Define frame edges from audio duration to align T frames strictly within sentence
totalDuration = numel(cleanAudio) / FS;
frameEdges = linspace(0, totalDuration, nFrames + 1);

%% ====================== FEATURE PARAMS (COMPACT DEFAULT) ======================
% Sub-windows INSIDE each MR frame (no cross-frame ops)
subWin_s  = 0.025;   % 25 ms sub-window
subHop_s  = 0.0125;  % 12.5 ms hop  (4 windows per 62.5 ms frame if evenly divisible)
subWin    = max(1, round(subWin_s * FS));
subHop    = max(1, round(subHop_s * FS));
hamWin    = hamming(subWin, 'periodic');

% Mel & MFCC sizes
numMelBands = opts.numMelBands;
numMFCC     = opts.numMFCC;


% Pitch range (Hz)
pitchRange = [50 300];

% Formant LPC params (@16 kHz)
lpcOrder   = 14;          % 12–16 reasonable; we use 14
formWin    = round(0.030 * FS);  % 30 ms window
formHop    = round(0.010 * FS);  % 10 ms hop
formHam    = hamming(formWin, 'periodic');
maxBW_Hz   = 400;         % bandwidth guardrail

% Optional raw spectrum (very coarse)
rawSpecBins = 32;         % keep tiny by default

if opts.verbose
   fprintf('[CFG] Mel=%d | MFCC=%d | Formants=%s | subWin=%d | subHop=%d\n', ...
    numMelBands, numMFCC, string(opts.includeFormants), subWin, subHop);
end

%% ====================== HELPERS ======================
% Safe log10
safelog10 = @(x) log10(max(x, realmin('double')));

% Compute mel-spectrogram and average within-frame
computeLogMelMean = @(x) ( ...
    mean( safelog10( melSpectrogram(x, FS, 'Window', hamWin, 'OverlapLength', subWin - subHop, ...
           'NumBands', numMelBands, 'FrequencyRange', [50 8000]) ), 2, 'omitnan')' );

% Compute MFCC static + within-frame Δ slopes
function [mfccStatic, mfccDelta] = mfccWithinFrame(x, fs, hamWin, subWin, subHop, numMFCC)
    coeffs = mfcc(x, fs, 'Window', hamWin, 'OverlapLength', subWin - subHop, ...
                  'NumCoeffs', numMFCC, 'LogEnergy', 'Ignore'); % frames x numMFCC
    if isempty(coeffs)
        mfccStatic = zeros(1, numMFCC);
        mfccDelta  = zeros(1, numMFCC);
        return;
    end
    mfccStatic = mean(coeffs, 1, 'omitnan');
    % Within-frame Δ: slope per coefficient
    nf = size(coeffs,1);
    t  = (0:nf-1).';
    mfccDelta = zeros(1, numMFCC);
    if nf >= 2
        % vectorized least-squares slope for each column:
        % slope = cov(t,y)/var(t)  with y demeaned across time
        t0 = t - mean(t);
        denom = sum(t0.^2);
        Y = coeffs - mean(coeffs,1);
        mfccDelta = (t0.' * Y) / max(denom, eps);
    end
end

% Spectral shape features averaged within-frame
function shp = spectralShapeWithinFrame(x, fs, hamWin, subWin, subHop, melBands)
    % Centroid, spread, rolloff(85%), flatness, tilt (low vs high)
    c  = spectralCentroid(x, fs, 'Window', hamWin, 'OverlapLength', subWin - subHop);
    s  = spectralSpread (x, fs, 'Window', hamWin, 'OverlapLength', subWin - subHop);
    r  = spectralRolloffPoint(x, fs, 'Window', hamWin, 'OverlapLength', subWin - subHop, 'Threshold', 0.85);
    f  = spectralFlatness(x, fs, 'Window', hamWin, 'OverlapLength', subWin - subHop);
    % Tilt from mel energies (low 0–1 kHz vs high 4–8 kHz)
    [melSpec, melFreqs] = melSpectrogram(x, fs, 'Window', hamWin, 'OverlapLength', subWin - subHop, ...
                                         'NumBands', melBands, 'FrequencyRange', [50 8000]);
    melPow = max(melSpec, 0);
    lowIdx  = melFreqs <= 1000;
    highIdx = melFreqs >= 4000;
    if any(lowIdx) && any(highIdx)
        lowE  = mean(sum(melPow(lowIdx, :), 1));
        highE = mean(sum(melPow(highIdx, :), 1));
        tilt  = 10*log10((lowE + eps) / (highE + eps));
    else
        tilt = 0;
    end
    shp = [mean(c,'omitnan') mean(s,'omitnan') mean(r,'omitnan') mean(f,'omitnan') tilt];
end

% Formant extraction (F1..F3 medians over voiced sub-windows) + ratio F2/F1
function [F1,F2,F3,F2F1] = formantsWithinFrame(x, fs, ham, win, hop, p, maxBW)
    % Pre-emphasis and windowing inside; voiced-only handled by caller
    n  = numel(x);
    idx = 1:hop:(n-win+1);
    F = nan(numel(idx),3);

    for k = 1:numel(idx)
        seg = x(idx(k):min(idx(k)+win-1, n));
        if numel(seg) < win, seg(end+1:win) = 0; end
        seg = filter([1 -0.97],1, seg(:).*ham);

        a = lpc(seg, p);
        r = roots(a); r = r(imag(r)>0);
        ang = atan2(imag(r), real(r));
        fr  = ang * (fs/(2*pi));                  % Hz
        bw  = -log(abs(r)) * fs/pi;

        keep = fr>90 & fr<fs/2 & bw<maxBW;
        fr  = sort(fr(keep));

        if numel(fr)>=3, F(k,:) = fr(1:3).'; 
        elseif numel(fr)==2, F(k,:) = [fr(1:2).' NaN];
        elseif numel(fr)==1, F(k,:) = [fr(1) NaN NaN];
        end
    end

    % Plausibility clip per formant (adult speech; generous)
    lo = [150  600 1200];  hi = [1200 3500 4500];
    for j=1:3
        bad = F(:,j)<lo(j) | F(:,j)>hi(j);
        F(bad,j) = NaN;
    end

    F1 = median(F(:,1),'omitnan'); 
    F2 = median(F(:,2),'omitnan'); 
    F3 = median(F(:,3),'omitnan'); 
    if ~isfinite(F1) || ~isfinite(F2) || F1<=0 || F2<=0
        F2F1 = 0;
    else
        F2F1 = F2/F1;
    end
    if ~isfinite(F1), F1=0; end
    if ~isfinite(F2), F2=0; end
    if ~isfinite(F3), F3=0; end
end


% Optional tiny raw spectrum (32 bins) from full 62.5 ms frame (Hann)
function rs = rawSpectrumTiny(x, fs, bins)
    N = 2^nextpow2(max(numel(x), 256));
    w = hann(numel(x));
    X = abs(fft(x(:).*w, N));
    X = X(1:floor(N/2)+1);
    % downsample to fixed bins
    edges = round(linspace(1, numel(X), bins+1));
    rs = zeros(1, bins);
    for b = 1:bins
        rs(b) = mean(X(edges(b):edges(b+1)-1));
    end
    rs = log10(max(rs, realmin('double')));
end

%% ====================== MAIN LOOP ======================
allFrameFeatures = [];
featLen = [];  % will be set after first frame
tic;
for frameIdx = 1:nFrames
    % Segment audio for this MR/video frame
    t0 = frameEdges(frameIdx);   t1 = frameEdges(frameIdx+1);
    s0 = max(1, floor(t0 * FS) + 1);
    s1 = min(numel(cleanAudio), floor(t1 * FS));
    frameAudio = cleanAudio(s0:s1);

    % Ensure minimum length for windowing
    if numel(frameAudio) < subWin
        frameAudio(end+1:subWin) = 0;
    end

    % --- Pre-emphasis before spectral features ---
    frameAudioPE = filter([1 -0.97], 1, frameAudio);

    % 1) Log-mel energies (mean over sub-windows) --> [1 x numMelBands]
    melMean = computeLogMelMean(frameAudioPE);  % row vector length = numMelBands

    % 2) MFCC static + within-frame Δ slopes (no cross-frame leakage)
    [mfccStatic, mfccDelta] = mfccWithinFrame(frameAudioPE, FS, hamWin, subWin, subHop, numMFCC);

    % 3) Prosody: voicing fraction, F0 (Hz), log-energy  [robust, no unsupported args]
    f0Track = pitch(frameAudio, FS, ...
                    'Method','CEP', ...                 
                    'Range', pitchRange, ...
                    'WindowLength', subWin, ...
                    'OverlapLength', subWin - subHop, ...
                    'MedianFilterLength', 3);
    
    % Energy per sub-window (aligned to pitch frames)
    nf = max(0, floor((numel(frameAudio)-subWin)/(subHop)) + 1);
    energyTrack = zeros(nf,1);
    for ii = 1:nf
        s = (ii-1)*subHop + 1;
        e = s + subWin - 1;
        seg = frameAudio(s:e);
        energyTrack(ii) = mean(seg.^2);
    end
    energyDB = 10*log10(energyTrack + eps);
    
    % Percentile-based energy gate (less brittle than fixed dB offset)
    thr_dB = prctile(energyDB, 30);              % start at 30th percentile
    voicedMask = (f0Track > 0) & (energyDB > thr_dB);
    
    % Adaptive relax if too few voiced frames
    if mean(voicedMask) < 0.15
        thr_dB = prctile(energyDB, 20);
        voicedMask = (f0Track > 0) & (energyDB > thr_dB);
    end
    
    voicedFrac = mean(voicedMask);
    F0 = 0;
    if voicedFrac > 0
        F0 = median(f0Track(voicedMask));
    end
    logE = safelog10(sum(frameAudio.^2));        % frame-level energy

    



    % 4) Spectral shape (centroid, spread, rolloff85, flatness, tilt)
    shp = spectralShapeWithinFrame(frameAudioPE, FS, hamWin, subWin, subHop, numMelBands); % [1x5]

    % 5) Formants F1..F3 + F2/F1 (within-frame medians) — only if sufficiently voiced
    if voicedFrac < 0.20
        F1 = 0; F2 = 0; F3 = 0; F2F1 = 0;
    else
        [F1, F2, F3, F2F1] = formantsWithinFrame(frameAudio, FS, formHam, formWin, formHop, lpcOrder, maxBW_Hz);
    end

    if ~opts.includeFormants, F1 = 0; F2 = 0; F3 = 0; F2F1 = 0; end


    % 6) Optional raw spectrum (very coarse)
    if opts.includeRawSpectrum
        rs = rawSpectrumTiny(frameAudioPE, FS, rawSpecBins);
    else
        rs = [];
    end

    % Concatenate features for this frame (fixed-length row)
    frameFeat = [melMean, mfccStatic, mfccDelta, voicedFrac, F0, logE, shp, F1, F2, F3, F2F1, rs];

    % Initialize storage after first frame when feat length known
    if isempty(featLen)
        featLen = numel(frameFeat);
        allFrameFeatures = zeros(nFrames, featLen);
        if opts.verbose
            extraStr = '';
            if opts.includeRawSpectrum
                extraStr = ', +rawSpec32';
            end
            fprintf('[INIT] Feature length per frame: %d (MFCC=%d, mel=%d%s)\n', ...
                featLen, numMFCC, numMelBands, extraStr);
        end
    else
        if numel(frameFeat) ~= featLen
            error('Feature length changed within run (%d -> %d) at frame %d', featLen, numel(frameFeat), frameIdx);
        end
    end

    allFrameFeatures(frameIdx, :) = frameFeat;

    % Progress
    if opts.verbose && (mod(frameIdx, max(1, floor(nFrames/10))) == 0 || frameIdx == nFrames)
        fprintf('[PROGRESS] %d / %d frames (%.1f%%)\n', frameIdx, nFrames, 100 * frameIdx / nFrames);
    end
end
elapsed = toc;

%% ====================== SUMMARY & SAVE ======================
fprintf('\n=== EXTRACTION COMPLETE ===\n');
fprintf('Frames: %d | Features/frame: %d | Elapsed: %.2f s (%.2f ms/frame)\n', nFrames, featLen, elapsed, 1000*elapsed/max(1,nFrames));

% Feature block breakdown (approximate, depends on opts)
fprintf('\n[BLOCKS]\n');
fprintf('  Log-mel (%d)\n', numMelBands);
fprintf('  MFCC (%d) + Δ (%d)\n', numMFCC, numMFCC);
fprintf('  Prosody (voicedFrac, F0, logE) = 3\n');
fprintf('  Spectral shape (centroid, spread, rolloff, flatness, tilt) = 5\n');
fprintf('  Formants (F1,F2,F3,F2/F1) = %d\n', 4 * opts.includeFormants + 0*(~opts.includeFormants));
if opts.includeRawSpectrum
    fprintf('  Raw spectrum (bins=%d)\n', rawSpecBins);
end

% Insert into struct and save alongside existing pipeline expectations
try
    if exist(saveMatPath, 'file')
        L = load(saveMatPath, opts.saveVarName);
        if isfield(L, opts.saveVarName)
            eval([opts.saveVarName ' = L.(opts.saveVarName);']);
        end
    end
catch
    % ignore if file doesn't exist or struct missing; we'll create fresh
end

if ~exist('audioData','var')
    audioData = struct();
end
audioData(dataIdx).(opts.saveFieldName) = allFrameFeatures; %#ok<STRNU>

save(saveMatPath, 'audioData', '-v7.3');
fprintf('[SAVE] %s -> variable ''audioData(%d).%s''\n', saveMatPath, dataIdx, opts.saveFieldName);

%% ====================== END ======================
