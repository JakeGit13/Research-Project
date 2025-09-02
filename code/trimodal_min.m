%% trimodal_min.m
% Minimal, straightforward PCA pipeline for H1/H2 with clear toggles.
% Assumes per-frame, time-aligned matrices:
%   thisMRWarp  : [p_mr  x T]
%   thisVidWarp : [p_vid x T]
%   thisAudio   : [p_aud x T]  (only when useTrimodal = true)
% Replace the LOADING SECTION with your actual .mat variables.

%% ===================== User Toggles =====================
% Hypothesis / target
analysisMode   = 'H1';           % 'H1' or 'H2'
reconstructInd = 3;              % 1=MR, 2=Video, 3=Audio (target to reconstruct)

% Modalities
useTrimodal    = true;           % true for MR+VID+AUD; false for MR+VID

% PCA & bootstrap
nComp          = 30;
nBoots         = 1000;
lambda_ridge   = 1e-3;
usePar         = true;

% Temporal context
useTemporalStack = true;         % true: stack T contiguous frames; false: T=1
Twin           = 40;             % temporal window size if useTemporalStack=true

% Preprocessing
preprocMode    = 'zscore_sqrtN'; % 'zscore_sqrtN' (recommended) or 'legacy'

%% ===================== LOADING SECTION =====================
% TODO: Replace this with your own data loading lines
% Example (single speaker/sentence already aligned):
load('mrAndVideoData.mat','thisMRWarp','thisVidWarp');
load('audioFeaturesData.mat','thisAudio');
error('Replace the LOADING SECTION with your data loads: thisMRWarp, thisVidWarp, thisAudio.');

%% ===================== Sanity checks =====================
if ~exist('thisMRWarp','var') || ~exist('thisVidWarp','var')
    error('Missing MR/Video variables.');
end
if useTrimodal && ~exist('thisAudio','var')
    error('useTrimodal=true but thisAudio is missing.');
end

% Ensure double precision
thisMRWarp  = double(thisMRWarp);
thisVidWarp = double(thisVidWarp);
if useTrimodal, thisAudio = double(thisAudio); end

% Ensure same number of frames
Tmr  = size(thisMRWarp,2);
Tvid = size(thisVidWarp,2);
if Tmr ~= Tvid, error('MR and Video have different frame counts.'); end
if useTrimodal && size(thisAudio,2) ~= Tmr
    error('Audio frames do not match MR/Video.');
end

%% ===================== Temporal stacking (optional) =====================
if ~useTemporalStack
    Twin = 1;
end

[thisMRWarp, thisVidWarp, thisAudio, validIdx] = stackT_all(thisMRWarp, thisVidWarp, ...
    iff(useTrimodal, thisAudio, []), Twin);

% After stacking, all modalities have T_eff frames
T_eff = size(thisMRWarp,2);

%% ===================== Preprocessing =====================
% Helper: per-row z-score with safe denominator
zscore_rows = @(X) (X - mean(X,2)) ./ max(std(X,0,2), 1e-12);

switch preprocMode
    case 'legacy'
        % Mean-centering will be done inside PCA; no per-row z-score, no block scaling
        mrW  = thisMRWarp;        vidW = thisVidWarp;
        if useTrimodal, audW = thisAudio; else, audW = []; end
        blockScales = [1,1,1];

        % For evaluation in z-space, create z-scored copies
        mrZ  = zscore_rows(thisMRWarp);
        vidZ = zscore_rows(thisVidWarp);
        if useTrimodal, audZ = zscore_rows(thisAudio); else, audZ = []; end

    case 'zscore_sqrtN'
        % Per-row z-score per modality
        mrZ  = zscore_rows(thisMRWarp);
        vidZ = zscore_rows(thisVidWarp);
        if useTrimodal, audZ = zscore_rows(thisAudio); else, audZ = []; end

        % Fail-fast if any non-finite (shouldn’t happen with safe z-score)
        if any(~isfinite(mrZ(:))) || any(~isfinite(vidZ(:))) || ...
           (useTrimodal && any(~isfinite(audZ(:))))
            error('NaNs/Infs detected after z-scoring.');
        end

        % Block scaling by inverse sqrt(#rows) to balance block trace
        wmr  = 1 / sqrt(size(mrZ,1));
        wvid = 1 / sqrt(size(vidZ,1));
        waud = iff(useTrimodal, 1 / sqrt(size(audZ,1)), 1);

        mrW  = wmr  * mrZ;
        vidW = wvid * vidZ;
        if useTrimodal, audW = waud * audZ; else, audW = []; end
        blockScales = [wmr, wvid, waud];

    otherwise
        error('Unknown preprocMode: %s', preprocMode);
end

% Concatenate for PCA
if useTrimodal
    X = [mrW; vidW; audW];      % (p_total x T_eff)
else
    X = [mrW; vidW];
end

% Element boundaries (for masking target rows)
p_mr  = size(mrW,1);
p_vid = size(vidW,1);
if useTrimodal, p_aud = size(audW,1); else, p_aud = 0; end
if useTrimodal
    elementBoundaries = [0, p_mr, p_mr+p_vid, p_mr+p_vid+p_aud];
else
    elementBoundaries = [0, p_mr, p_mr+p_vid];
end

%% ===================== Target, mask, and shuffle target =====================
tarIdx = (elementBoundaries(reconstructInd)+1) : elementBoundaries(reconstructInd+1);
M      = size(X,1);
obs_idx = true(M,1);   % observed rows mask
obs_idx(tarIdx) = false;

% Choose which block to shuffle in bootstrap
shuffleTarget = reconstructInd;                      % default for H1
if strcmpi(analysisMode,'H2') && useTrimodal && ismember(reconstructInd,[1,2])
    shuffleTarget = 3;  % shuffle Audio when testing whether Audio helps MR/Video
end

%% ===================== PCA fit on real data =====================
[PC, Xmean] = doPCA(X, nComp);

% Infer scores from observed rows; reconstruct in weighted z-space
A_real    = infer_scores_from_observed(PC, Xmean, X, obs_idx, lambda_ridge);
Xhat_real = PC * A_real + Xmean;

% Extract target block reconstruction (unweight for evaluation in z-space)
XhatB_z = Xhat_real(tarIdx,:) / blockScales(reconstructInd);

% Ground truth in z-space for the target block
switch reconstructInd
    case 1, XtrueB_z = mrZ;
    case 2, XtrueB_z = vidZ;
    case 3, XtrueB_z = audZ;
end
% Align to stacked range if needed (they already are)

% Vectorised and median row-wise r(z)
[r_vec, r_med] = corr_block(XtrueB_z, XhatB_z);

% Probe: no observed rows (should be ~0)
A_probe    = infer_scores_from_observed(PC, Xmean, X, false(M,1), lambda_ridge);
Xhat_probe = PC * A_probe + Xmean;
Xhat_probeB_z = Xhat_probe(tarIdx,:) / blockScales(reconstructInd);
[r_vec_probe, ~] = corr_block(XtrueB_z, Xhat_probeB_z);

%% ===================== Bootstrap (shuffle baseline) =====================
rng('default');
permIndexes = zeros(nBoots, T_eff);
for b = 1:nBoots, permIndexes(b,:) = randperm(T_eff); end

r_boot = zeros(nBoots,1);

nCores = feature('numcores');
useParEff = usePar && nCores > 2;

if useParEff
    pool = gcp('nocreate'); if isempty(pool), parpool(max(1,nCores-1)); end
    parfor b = 1:nBoots
        Xs = shuffle_block(X, permIndexes(b,:), shuffleTarget, elementBoundaries);
        [PCb, mb] = doPCA(Xs, nComp);
        Ab    = infer_scores_from_observed(PCb, mb, Xs, obs_idx, lambda_ridge);
        Xhatb = PCb * Ab + mb;
        XhatbB_z = Xhatb(tarIdx,:) / blockScales(reconstructInd);
        r_boot(b,1) = corr_vec(XtrueB_z(:), XhatbB_z(:));
    end
else
    for b = 1:nBoots
        Xs = shuffle_block(X, permIndexes(b,:), shuffleTarget, elementBoundaries);
        [PCb, mb] = doPCA(Xs, nComp);
        Ab    = infer_scores_from_observed(PCb, mb, Xs, obs_idx, lambda_ridge);
        Xhatb = PCb * Ab + mb;
        XhatbB_z = Xhatb(tarIdx,:) / blockScales(reconstructInd);
        r_boot(b,1) = corr_vec(XtrueB_z(:), XhatbB_z(:));
    end
end

ci = quantile(r_boot, [0.025 0.975]);
p_right = mean(r_boot >= r_vec);

%% ===================== Report =====================
fprintf('\nSanity: useTrimodal=%d | reconstructInd=%d | T=%d | nComp=%d | analysisMode=%s | preprocMode=%s\n', ...
    useTrimodal, reconstructInd, Twin, nComp, analysisMode, preprocMode);

blockNames = {'MR','Video','Audio'};
if useTrimodal
    fprintf('[Trimodal] Reconstruct %s from %s | median r(z)=%.4f | vectorised r(z)=%.4f\n', ...
        blockNames{reconstructInd}, strjoin(setdiff(blockNames,blockNames(reconstructInd)),'+'), r_med, r_vec);
else
    fprintf('[Bimodal] Reconstruct %s from %s | median r(z)=%.4f | vectorised r(z)=%.4f\n', ...
        blockNames{reconstructInd}, strjoin(setdiff(blockNames(1:2),blockNames(reconstructInd)),'+'), r_med, r_vec);
end

fprintf('Shuffle baseline: median=%.4f, CI=[%.4f, %.4f], p(R_boot >= R_real)=%.4f\n', ...
    median(r_boot), ci(1), ci(2), p_right);
fprintf('Probe (no observed rows): r_vec=%.4f\n', r_vec_probe);

%% ===================== Local functions =====================
function [PC, mu] = doPCA(X, nComp)
% PCA via time-cov SVD; returns feature-space components (PC) and mean
    mu = mean(X,2);
    Xc = X - mu;
    % Economy SVD on T x T space
    [U,S,~] = svd(Xc','econ');             % (T x r)
    r  = min([size(X,1), size(X,2), nComp]);
    U  = U(:,1:r);
    S  = S(1:r,1:r);
    PC = Xc * (U / S);                      % (p x r) feature-space components
end

function A = infer_scores_from_observed(PC, mu, X, obs_idx, lambda)
% Infer component scores for each frame using only observed rows (ridge LS)
    Xc = X - mu;                            % (p x T)
    P  = PC(obs_idx,:);                     % (p_obs x k)
    Y  = Xc(obs_idx,:);                     % (p_obs x T)
    % Solve (P'P + λI) A = P'Y  ->  A = (P'P + λI) \ (P'Y)
    PtP = P.' * P;
    k   = size(P,2);
    A   = (PtP + lambda*eye(k)) \ (P.' * Y); % (k x T)
end

function Xs = shuffle_block(X, permIdx, whichBlock, bounds)
% Permute frames within the chosen block; other blocks unchanged
    switch whichBlock
        case 1  % MR
            r = (bounds(1)+1):bounds(2);
        case 2  % Video
            r = (bounds(2)+1):bounds(3);
        case 3  % Audio
            r = (bounds(3)+1):bounds(4);
        otherwise
            error('Invalid block index.');
    end
    Xs = X;
    Xs(r,:) = X(r, permIdx);
end

function [r_vec, r_med] = corr_block(XtrueB_z, XhatB_z)
% Vectorised correlation and median row-wise correlation (safe)
    r_vec = corr_vec(XtrueB_z(:), XhatB_z(:));
    % Row-wise r
    r_rows = zeros(size(XtrueB_z,1),1);
    for i = 1:size(XtrueB_z,1)
        r_rows(i) = corr_vec(XtrueB_z(i,:).', XhatB_z(i,:).');
    end
    r_med = median(r_rows,'omitnan');
end

function r = corr_vec(x, y)
% Safe Pearson correlation with complete rows handling
    if isrow(x), x = x.'; end
    if isrow(y), y = y.'; end
    r = corr(x, y, 'Rows','complete');
end

function Y = iff(cond, a, b)
% Inline ternary
    if cond, Y = a; else, Y = b; end
end

function [MRs, VIDs, AUDs, validIdx] = stackT_all(MR, VID, AUD, T)
% Stack T contiguous frames along rows (temporal context).
% Outputs have (T * rows) x (T_frames) and T_frames = size(X,2) - T + 1
    if T <= 1
        MRs = MR; VIDs = VID; AUDs = AUD; validIdx = 1:size(MR,2);
        return;
    end
    MRs = stackT(MR,T);
    VIDs = stackT(VID,T);
    if ~isempty(AUD), AUDs = stackT(AUD,T); else, AUDs = []; end
    validIdx = T:size(MR,2);
end

function Xs = stackT(X,T)
% Create stacked features by concatenating T time-delayed copies along rows
    [p, n] = size(X);
    if n < T, error('Not enough frames (%d) for T=%d.', n, T); end
    Tn = n - T + 1;
    Xs = zeros(p*T, Tn, 'like', X);
    for t = 1:Tn
        % Concatenate frames t:t+T-1 vertically
        Xs(:,t) = reshape(X(:, t:t+T-1), [], 1);
    end
end
