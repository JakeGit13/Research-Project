function trimodalPCA_Clean
% Clean, toggle-driven PCA with block-masked reconstruction and bootstrap shuffles.
% Primary metric: block-level Pearson r in z-space (per your earlier weighting).
% Secondary (faithful) view: loadings scatter + shuffled histogram.
%
% Toggles:
%   useTrimodal    : false = MR+Video, true = MR+Video+Audio
%   reconstructInd : 1 = MR, 2 = Video, 3 = Audio
%   shuffleMode    : 'none' | 'shuffleMR' | 'shuffleVideo' | 'shuffleAudio'
%
% Notes:
% - Runs parallel only (opens a pool if needed).
% - Correlations are computed in z-space to avoid unit mismatch.
% - Block weights = 1/sqrt(p_block) applied to z-scored rows to equalise total variance.

clc; rng('default');

%% ---- User paths/files ----
dataDir   = '/Users/jaker/Research-Project/data';
dataFile  = 'mrAndVideoData.mat';                 % contains data(ii).mr_warp2D, data(ii).vid_warp2D
audioFile = 'audioFeaturesData_articulatory.mat'; % contains audioData(ii).audioFeatures_articulatory'

%% ---- Toggles / config ----
useTrimodal     = true;                % <— set for H1/H2 as described above
reconstructInd  = 3;                   % 1=MR, 2=Video, 3=Audio
shuffleMode     = 'shuffleAudio';      % 'none'|'shuffleMR'|'shuffleVideo'|'shuffleAudio'
nBoots          = 1000;
lambda          = 1e-3;                % ridge for score inference
itemIdx         = 9;                   % which sentence/item to run

%% ---- Load ----
addpath(dataDir);
load(dataFile,  'data');
load(audioFile, 'audioData');

% Open parallel pool
nCores = max(1, feature('numcores'));
poolOpen = gcp('nocreate');
if isempty(poolOpen)
    parpool(max(1, nCores-1));
end

%% ---- Select item ----
thisMRWarp = data(itemIdx).mr_warp2D;                 % (p_mr x T)
thisVidWarp = data(itemIdx).vid_warp2D;               % (p_vid x T)
thisAudio   = audioData(itemIdx).audioFeatures_articulatory'; % (p_aud x T)

fprintf('Item %d\nMR: %d×%d, Video: %d×%d, Audio: %d×%d\n', itemIdx, ...
    size(thisMRWarp,1), size(thisMRWarp,2), size(thisVidWarp,1), size(thisVidWarp,2), ...
    size(thisAudio,1), size(thisAudio,2));

%% ---- Z-score rows (retain z-space for metrics) ----
[mrZ, mu_mr, sd_mr]   = zscore_keep(thisMRWarp); %#ok<NASGU,ASGLU>
[vidZ, mu_vid, sd_vid]= zscore_keep(thisVidWarp); %#ok<NASGU,ASGLU>
[audZ, mu_aud, sd_aud]= zscore_keep(thisAudio);  %#ok<NASGU,ASGLU>

p_mr  = size(mrZ,  1);
p_vid = size(vidZ, 1);
p_aud = size(audZ, 1);
T     = size(mrZ,  2);

% Block weights (equalise total variance across blocks after z-scoring)
wmr  = 1/sqrt(max(p_mr,1));
wvid = 1/sqrt(max(p_vid,1));
waud = 1/sqrt(max(p_aud,1));

% Build weighted stack and boundaries once
if useTrimodal
    Xz = [mrZ; vidZ; audZ];
    scales = [wmr, wvid, waud];
    bounds = [0, p_mr, p_mr+p_vid, p_mr+p_vid+p_aud];
else
    if reconstructInd==3
        error('Audio (block 3) is not available when useTrimodal=false.');
    end
    Xz = [mrZ; vidZ];
    scales = [wmr, wvid];
    bounds = [0, p_mr, p_mr+p_vid];
    if strcmpi(shuffleMode,'shuffleAudio')
        warning('shuffleMode=shuffleAudio ignored in bimodal mode. Using ''none''.');
        shuffleMode = 'none';
    end
end

% Apply weights in z-space
Xw = Xz;
for b = 1:(numel(bounds)-1)
    idx = (bounds(b)+1):bounds(b+1);
    Xw(idx,:) = scales(b) * Xw(idx,:);
end

%% ---- Fit PCA (time-cov SVD, faithful style) ----
[prinComp, fitMean, origLoadings] = doPCA(Xw);   % prinComp: D×T, fitMean: D×1, origLoadings: T×T

%% ---- Observed mask (hide target block for score inference) ----
obsMask = true(size(Xw,1),1);
tarIdx  = block_idx(bounds, reconstructInd);
obsMask(tarIdx) = false;

%% ---- Infer scores from observed rows (ridge), reconstruct full X ----
scores = infer_scores_from_observed(prinComp, fitMean, Xw, obsMask, lambda); % T×T
Xhat_w = prinComp * scores + fitMean;                                        % D×T, weighted z-space

% Extract target block, undo ONLY block-weight -> back to z-space
XhatB_z   = Xhat_w(tarIdx,:) / scales(reconstructInd);
XtrueB_z  = Xz(tarIdx,:);

% Primary metric (z-space): Pearson r over all entries
r_real = corr(XtrueB_z(:), XhatB_z(:));

% Labeling
blkNames = {'MR','Video','Audio'};
obsNames = observed_list(useTrimodal, reconstructInd);
modeName = ternary(useTrimodal,'Trimodal (MR+Video+Audio)','Bimodal (MR+Video)');
fprintf('\n[%s] Reconstruct %s from %s | r_block(z) = %.4f\n', ...
    modeName, blkNames{reconstructInd}, obsNames, r_real);

%% ---- Bootstrap baseline (shuffle selected modality frames before the fit) ----
% Note: For H1 chance, use shuffleMode='shuffleAudio'.
%       For H2 control, compare trimodal real vs trimodal with shuffleAudio.
permMat = zeros(T, nBoots, 'uint32');
for k=1:nBoots, permMat(:,k) = randperm(T); end

r_boot = zeros(nBoots,1);
parfor k = 1:nBoots
    % Make a shuffled copy in weighted z-space
    Xw_k = applyShuffle(Xw, bounds, scales, shuffleMode, permMat(:,k));
    % Fit PCA on the shuffled data
    [PCk, mean_k] = doPCA(Xw_k); %#ok<ASGLU>
    % Infer scores using same observed mask definition (on Xw_k)
    scores_k = infer_scores_from_observed(PCk, mean_k, Xw_k, obsMask, lambda);
    Xhatk_w  = PCk * scores_k + mean_k;
    % Extract target block (same tarIdx) and undo weight -> z-space
    XhatkB_z = Xhatk_w(tarIdx,:) / scales(reconstructInd);
    % Correlate in z-units
    r_boot(k) = corr(XtrueB_z(:), XhatkB_z(:));
end

% Report simple stats
ci = quantile(r_boot, [0.025 0.975]);
p_right = mean(r_boot >= r_real);
fprintf('Bootstrap (mode=%s): median=%.4f, CI=[%.4f, %.4f], p(R_boot >= R_real)=%.4f\n', ...
    shuffleMode, median(r_boot), ci(1), ci(2), p_right);

% Plot bootstrap for block-r
figure('Name','Block-level r (z-space)');
histogram(r_boot, 50); hold on; yl = ylim;
plot([r_real r_real], yl, 'r-', 'LineWidth', 2);
xlabel(sprintf('r(z) for %s reconstruction', blkNames{reconstructInd}));
ylabel('Count');
title(sprintf('%s | shuffle=%s', modeName, shuffleMode));
grid on;

%% ---- (Faithful) loadings-level check & histogram (optional secondary view) ----
% Non-shuffled: zero target block, recompute mean on partial (faithful to original),
% project to get "reconstructed loadings" and compare to original loadings.
partial = Xw;
partial(tarIdx,:) = 0;
partialMean = mean(partial, 2);
partialCtr  = partial - partialMean;
partialLoads = partialCtr' * prinComp;     % T×T

% Scatter
figure('Name','Loadings scatter (faithful style)');
plot(origLoadings(:), partialLoads(:), '.'); hold on;
refline(1,0); xlabel('Original loadings'); ylabel('Reconstructed (masked) loadings');
title(sprintf('%s | Reconstruct %s', modeName, blkNames{reconstructInd}));
grid on;

% Bootstrap (faithful loadings R) under the same shuffleMode
R_boot = zeros(nBoots,1);
parfor k = 1:nBoots
    Xw_k = applyShuffle(Xw, bounds, scales, shuffleMode, permMat(:,k));
    [PCk, ~, Lk] = doPCA(Xw_k);
    part_k = Xw_k; part_k(tarIdx,:) = 0;
    mean_k = mean(part_k,2);
    ctr_k  = part_k - mean_k;
    Lk_rec = ctr_k' * PCk;
    R_boot(k) = corr(Lk(:), Lk_rec(:));
end
R_real = corr(origLoadings(:), partialLoads(:));

figure('Name','Loadings R bootstrap (faithful)');
histogram(R_boot, 50); hold on; yl = ylim;
plot([R_real R_real], yl, 'r-', 'LineWidth', 2);
xlabel('Loadings correlation (faithful)'); ylabel('Count');
title(sprintf('%s | shuffle=%s', modeName, shuffleMode));
grid on;

fprintf('\nDone.\n');
end % main

%% ---- Helpers ----
function [Z, mu, sd] = zscore_keep(X)
    mu = mean(X, 2);
    sd = std(X, 0, 2); sd(sd==0) = 1;
    Z  = (X - mu) ./ sd;
end

function idx = block_idx(bounds, b)
    idx = (bounds(b)+1):bounds(b+1);
end

function Xw_out = applyShuffle(Xw_in, bounds, scales, shuffleMode, perm)
% Shuffle frames (columns) within one chosen block, in weighted z-space.
    Xw_out = Xw_in;
    switch lower(shuffleMode)
        case 'none'
            return;
        case 'shufflemr',     b = 1;
        case 'shufflevideo',  b = 2;
        case 'shuffleaudio',  b = 3;
        otherwise
            error('Unknown shuffleMode: %s', shuffleMode);
    end
    if b > (numel(bounds)-1)
        % e.g., asked to shuffle Audio in bimodal mode
        return;
    end
    idx = (bounds(b)+1):bounds(b+1);
    Xw_out(idx,:) = Xw_in(idx, perm);
    % (No need to re-apply scales here; Xw_in is already weighted)
end

function [prinComp, MorphMean, loadings] = doPCA(dataW)
% Faithful to your original template: time-cov SVD; returns data-space PCs.
    MorphMean = mean(dataW, 2);
    Xc = bsxfun(@minus, dataW, MorphMean);
    xxt = Xc' * Xc;
    [~, S, V] = svd(xxt, 'econ');
    LInv      = 1 ./ sqrt(diag(S) + eps);
    prinComp  = Xc * V * diag(LInv);   % D×T
    loadings  = Xc' * prinComp;        % T×T
end

function scores = infer_scores_from_observed(prinComp, fitMean, Xw, obsMask, lambda)
% Solve for per-frame scores using only observed rows: ridge normal equations.
% P_obs: D_obs×T PCs; Y_obs: D_obs×T centered observed data.
    P_obs = prinComp(obsMask, :);                   % D_obs×T
    Y_obs = bsxfun(@minus, Xw(obsMask,:), fitMean(obsMask)); % D_obs×T
    A = (P_obs' * P_obs + lambda * eye(size(P_obs,2)));
    scores = A \ (P_obs' * Y_obs);                 % T×T
end

function out = observed_list(useTrimodal, reconstructInd)
    names = {'MR','Video','Audio'};
    if ~useTrimodal
        present = [true,true,false];
    else
        present = [true,true,true];
    end
    present(reconstructInd) = false;
    out = strjoin(names(present), '+');
end

function out = ternary(cond, a, b)
    if cond, out=a; else, out=b; end
end
