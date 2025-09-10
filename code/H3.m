
% Purpose: Using the *same* PCA regime as H1/H2 (block balancing; standard PCA;
%          zero-out & project), quantify how well *each audio family* is
%          reconstructed when inferring scores from MR+Video only.
%
% Key points:
%   • No re-extraction; no architecture changes.
%   • Works sentence-by-sentence (T frames).
%   • Computes per-family VAF in the *weighted, centred audio space*.
%   • Optional drop-one ΔVAF: set reconstructed rows of one family to zero,
%     then recompute overall Audio VAF to measure its contribution.
%
% Inputs expected:
%   - MR/Video: C:\Users\jaker\Research-Project\data\mrAndVideoData.mat  (variable: data)
%               with fields .mr_warp2D and .vid_warp2D (features x T)
%   - Audio:    C:\Users\jaker\Research-Project\data\audioFeaturesData.mat (variable: audioData)
%               with field .audioFeatures (T x D) per sentence
%   - (Optional) audioIndex: struct with row-index map for audio families (in dims 1..D)
%               .mel, .mfcc, .mfccDelta, .prosody, .shape, .formants, (.rawspec optional)
%               If absent and D==78, a default compact mapping is auto-created.
%
% Output:
%   - Prints per-family VAF and (optional) drop-one ΔVAF.
%   - Saves a struct H3 to: C:\Users\jaker\Research-Project\data\H3_results.mat
%
% Author: ChatGPT (generated)

clc; clear;

%% ====================== USER OPTIONS / PATHS ======================
opts.verbose       = true;
opts.dataIdx       = 9;    % sentence index
opts.doDropOne     = true; % compute ΔVAF by zeroing one family at reconstruction
opts.mrOnlyAlso    = false; % optional: MR-only → Audio profiling
opts.videoOnlyAlso = false; % optional: Video-only → Audio profiling

mrVideoDataPath = 'C:\Users\jaker\Research-Project\data\mrAndVideoData.mat';
audioFeatPath   = 'C:\Users\jaker\Research-Project\data\audioFeaturesData.mat';
saveOutPath     = 'C:\Users\jaker\Research-Project\data\H3_results.mat';

%% ====================== LOAD DATA ======================
S = load(mrVideoDataPath, 'data');
if ~isfield(S,'data'), error('File missing variable ''data'': %s', mrVideoDataPath); end
D = S.data;

ii = opts.dataIdx;
if opts.verbose
    fprintf('[INFO] H3 on sentence index: %d\n', ii);
end

% Try both common video field names
if isfield(D(ii),'vid_warp2D')
    thisVidWarp = double(D(ii).vid_warp2D);
elseif isfield(D(ii),'video_warp2D')
    thisVidWarp = double(D(ii).video_warp2D);
else
    error('Could not find video field (.vid_warp2D or .video_warp2D) in data(%d).', ii);
end
thisMRWarp  = double(D(ii).mr_warp2D);

A = load(audioFeatPath);
if ~isfield(A,'audioData'), error('File missing variable ''audioData'': %s', audioFeatPath); end
if ~isfield(A.audioData(ii), 'audioFeatures')
    error('audioData(%d).audioFeatures not found in %s', ii, audioFeatPath);
end
% Audio features saved as T x D --> transpose to (dims x T)
thisAudio = double(A.audioData(ii).audioFeatures)'; 

% Sanity check T across blocks
T_mr  = size(thisMRWarp,2);
T_vid = size(thisVidWarp,2);
T_aud = size(thisAudio, 2);
assert(T_mr==T_vid && T_vid==T_aud, 'Frame mismatch MR=%d, Video=%d, Audio=%d', T_mr, T_vid, T_aud);
T = T_mr;

if opts.verbose
    fprintf('[INFO] Shapes  MR: %d x %d  |  Video: %d x %d  |  Audio: %d x %d\n', ...
        size(thisMRWarp,1), T, size(thisVidWarp,1), T, size(thisAudio,1), T);
end

%% ====================== AUDIO FAMILY INDEX MAP ======================
audioIndex = [];
if isfield(A, 'audioIndex')
    audioIndex = A.audioIndex;
end

Da = size(thisAudio,1);
if isempty(audioIndex)
    if Da == 78
        % Default compact mapping
        audioIndex.mel        = 1:40;
        audioIndex.mfcc       = 41:53;
        audioIndex.mfccDelta  = 54:66;
        audioIndex.prosody    = 67:69;  % [voicedFrac, F0, logE]
        audioIndex.shape      = 70:74;  % [centroid, spread, rolloff, flatness, tilt]
        audioIndex.formants   = 75:78;  % [F1,F2,F3,F2/F1]
        if opts.verbose
            fprintf('[INFO] Using default compact audioIndex (D=78).\n');
        end
    else
        error(['No audioIndex found in %s, and D=%d is not compact.\n' ...
               'Please add a struct ''audioIndex'' to the MAT file with fields:\n' ...
               '  .mel, .mfcc, .mfccDelta, .prosody, .shape, .formants, (.rawspec optional)\n' ...
               'Each a vector of row indices within 1..D_aud.'], audioFeatPath, Da);
    end
end

% Build a list of families present
famNames = fieldnames(audioIndex);
% sanity: ensure indices are valid and non-empty
validMask = true(numel(famNames),1);
for i = 1:numel(famNames)
    idx = audioIndex.(famNames{i});
    if isempty(idx) || any(idx<1 | idx>Da)
        validMask(i) = false;
        warning('[WARN] Dropping family "%s" due to empty or invalid indices.', famNames{i});
    end
end
famNames = famNames(validMask);

if opts.verbose
    fprintf('[INFO] Audio families used: %s\n', strjoin(famNames', ', '));
end

%% ====================== BLOCK BALANCING (weights) ======================
% Row-center each block
zc = @(X) bsxfun(@minus, X, mean(X,2));

Xmr0  = zc(thisMRWarp);
Xvid0 = zc(thisVidWarp);
Xaud0 = zc(thisAudio);

% lambda1 of X'X (T x T)  --> sigma1 = sqrt(lambda1)
topSigma = @(X) sqrt(max(eig( (X.')*X )));
sig_mr  = topSigma(Xmr0);   if sig_mr==0,  sig_mr=1;  end
sig_vid = topSigma(Xvid0);  if sig_vid==0, sig_vid=1; end
sig_aud = topSigma(Xaud0);  if sig_aud==0, sig_aud=1; end

w_mr  = 1 / sig_mr;
w_vid = 1 / sig_vid;
w_aud = 1 / sig_aud;

% Weighted, centred blocks
Xmr  = w_mr  * Xmr0;
Xvid = w_vid * Xvid0;
Xaud = w_aud * Xaud0;

% Concatenate (features x T)
Xtri = [Xmr; Xvid; Xaud];

% Row offsets for audio inside concatenation
n_mr  = size(Xmr,1);
n_vid = size(Xvid,1);
n_aud = size(Xaud,1);
audStart = n_mr + n_vid + 1;
audEnd   = n_mr + n_vid + n_aud;
audRows  = audStart:audEnd;

%% ====================== PCA (standard) ======================
% Standard PCA via economy SVD on features x T
[U,S,~] = svd(Xtri, 'econ');
k = min(T-1, size(U,2));  % rank ≤ T-1
Uk = U(:,1:k);

%% ====================== Score inference: MR+Video -> Audio ======================
partial = Xtri;
partial(audRows,:) = 0;        % hide audio (zero-out), only MR+Video observed
L = Uk' * partial;             % scores (k x T)
Xhat = Uk * L;                 % reconstruction (features x T)

Xaud_true = Xaud;                  % weighted, centred audio (ground truth)
Xaud_hat  = Xhat(audRows,:);       % reconstructed audio rows (weighted, centred)

%% ====================== Per-family VAF ======================
famVAF = struct();
for i = 1:numel(famNames)
    nm  = famNames{i};
    idx = audioIndex.(nm);
    X   = Xaud_true(idx,:);
    Y   = Xaud_hat(idx,:);
    num = sum((X(:) - Y(:)).^2);
    den = sum(X(:).^2);
    if den <= eps
        vaf = NaN;
    else
        vaf = 100 * (1 - num/den);
    end
    famVAF.(nm) = vaf;
end

% Overall audio VAF (for reference)
num_all = sum((Xaud_true(:) - Xaud_hat(:)).^2);
den_all = sum(Xaud_true(:).^2);
VAF_all = 100 * (1 - num_all/den_all);

%% ====================== Optional: drop-one ΔVAF ======================
dropOne = struct();
if opts.doDropOne
    dropOne.fullVAF = VAF_all;
    for i = 1:numel(famNames)
        nm  = famNames{i};
        idx = audioIndex.(nm);
        Y   = Xaud_hat;        % copy
        Y(idx,:) = 0;          % drop this family at readout
        num = sum((Xaud_true(:) - Y(:)).^2);
        den = sum(Xaud_true(:).^2);
        vafDrop = 100 * (1 - num/den);
        dropOne.(['drop_',nm]) = dropOne.fullVAF - vafDrop; % contribution in pp VAF
    end
end

%% ====================== Optional MR-only / Video-only profiling ======================
prof = struct();
if opts.mrOnlyAlso
    partial2 = zeros(size(Xtri));          % MR only
    partial2(1:n_mr,:) = Xmr;
    L2 = Uk' * partial2;
    Xhat2 = Uk * L2;
    Xaud_hat2 = Xhat2(audRows,:);
    vaf2 = 100 * (1 - sum((Xaud_true(:)-Xaud_hat2(:)).^2)/sum(Xaud_true(:).^2));
    prof.MRonly_allVAF = vaf2;
    for i = 1:numel(famNames)
        nm = famNames{i}; idx = audioIndex.(nm);
        X = Xaud_true(idx,:); Y = Xaud_hat2(idx,:);
        prof.(['MRonly_',nm]) = 100*(1 - sum((X(:)-Y(:)).^2)/sum(X(:).^2));
    end
end
if opts.videoOnlyAlso
    partial3 = zeros(size(Xtri));          % Video only
    partial3(n_mr+1:n_mr+n_vid,:) = Xvid;
    L3 = Uk' * partial3;
    Xhat3 = Uk * L3;
    Xaud_hat3 = Xhat3(audRows,:);
    vaf3 = 100 * (1 - sum((Xaud_true(:)-Xaud_hat3(:)).^2)/sum(Xaud_true(:).^2));
    prof.Videoonly_allVAF = vaf3;
    for i = 1:numel(famNames)
        nm = famNames{i}; idx = audioIndex.(nm);
        X = Xaud_true(idx,:); Y = Xaud_hat3(idx,:);
        prof.(['Videoonly_',nm]) = 100*(1 - sum((X(:)-Y(:)).^2)/sum(X(:).^2));
    end
end

%% ====================== Print summary ======================
fprintf('\n=== H3: MR+Video -> Audio | Sentence %d ===\n', ii);
fprintf('Overall Audio VAF (weighted/centred): %.2f %%\n', VAF_all);
fprintf('Per-family VAFs (%%):\n');
for i = 1:numel(famNames)
    nm = famNames{i};
    fprintf('  %-12s : %6.2f\n', nm, famVAF.(nm));
end
if opts.doDropOne
    fprintf('Drop-one ΔVAF (pp):\n');
    for i = 1:numel(famNames)
        nm = famNames{i};
        fprintf('  %-12s : %6.2f\n', nm, dropOne.(['drop_',nm]));
    end
end
if opts.mrOnlyAlso
    fprintf('\nMR-only -> Audio (overall VAF): %.2f %%\n', prof.MRonly_allVAF);
end
if opts.videoOnlyAlso
    fprintf('Video-only -> Audio (overall VAF): %.2f %%\n', prof.Videoonly_allVAF);
end

%% ====================== Save results ======================
H3 = struct();
H3.dataIdx    = ii;
H3.VAF_all    = VAF_all;
H3.famVAF     = famVAF;
H3.dropOne    = dropOne;
H3.profile    = prof;
H3.famNames   = famNames;
H3.weights    = struct('w_mr',w_mr,'w_vid',w_vid,'w_aud',w_aud);
H3.svd_k      = k;
H3.Dims       = struct('MR',size(thisMRWarp,1),'Video',size(thisVidWarp,1),'Audio',size(thisAudio,1),'T',T);

save(saveOutPath, 'H3', '-v7.3');
fprintf('\n[SAVE] H3 results saved to: %s\n', saveOutPath);
"""

# Write the script to the sandbox so the user can download it
path = "/mnt/data/H3_readout_ablation.m"
with open(path, "w", encoding="utf-8") as f:
    f.write(script)

path
