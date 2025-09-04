function finalTrimodalPCA
% Minimal extension of the original bimodal PCA+shuffle example to a
% trimodal setup (MR, Video, Audio) with block scaling so audio has a
% fair contribution. The PCA routine and the evaluation (loadings plots,
% shuffle bootstrap) are kept faithful to the original structure.

clc;

%% === Config ===
% Paths and files
dataDir  = '/Users/jaker/Research-Project/data';
dataFile = 'mrAndVideoData.mat';
audioFile = 'audioFeaturesData_articulatory.mat';

% Core parameters
useTrimodal   = true;    % false => bimodal (MR+Video), true => trimodal (MR+Video+Audio)
reconstructId = 2;       % 1 = MR, 2 = Video, 3 = Audio
shuffleTarget = 3;       % which block to shuffle
nComp         = 30;      % number of principal components to keep
nBoots        = 500;     % number of bootstraps
lambda        = 1e-3;    % ridge regularization
waudScale     = 1;       % audio weight multiplier for ablation studies
preprocMode   = 'zscore_sqrtN';  % kept for compatibility but only this mode is active
VERBOSE       = true;   % detailed diagnostics
randomSeed    = 'default';

% Reproducibility
rng(randomSeed);

% Check for Parallel Computing Toolbox
HAS_PAR = false;
try
    poolOpen = gcp('nocreate');
    if isempty(poolOpen)
        nCores = feature('numcores');
        if nCores > 2
            parpool(nCores-1);
            HAS_PAR = true;
        end
    else
        HAS_PAR = true;
    end
catch
    HAS_PAR = false;  % No parallel toolbox available
end

if HAS_PAR
    disp('Using parallel processing.');
else
    disp('NOT using parallel processing.');
end

%% === Load Data ===
addpath(dataDir);
load(dataFile,'data');         % MR + Video
load(audioFile,'audioData');   % Audio

% Process each item (adjust index as needed)
for ii = 9  % :length(data)
    
    clear results;
    
    %% === Load/Assemble Blocks ===
    thisMRWarp  = data(ii).mr_warp2D;                         % (p_mr x T)
    thisVidWarp = data(ii).vid_warp2D;                        % (p_vid x T)
    thisAudio   = audioData(ii).audioFeatures_articulatory';  % (p_aud x T)
    
    % Shape checks
    if size(thisMRWarp,2) ~= size(thisVidWarp,2) || size(thisMRWarp,2) ~= size(thisAudio,2)
        error('Frame count mismatch across modalities for item %d.', ii);
    end
    
    T = size(thisMRWarp, 2);
    fprintf('Item %d\n', ii);
    fprintf('MR:    %d features x %d frames\n', size(thisMRWarp,1), T);
    fprintf('Video: %d features x %d frames\n', size(thisVidWarp,1), T);
    fprintf('Audio: %d features x %d frames\n', size(thisAudio,1),   T);
    
    %% === Preprocessing (zscore + 1/sqrt(p)) ===
    % Safe per-row z-score helper
    zscore_rows = @(X) (X - mean(X,2)) ./ max(std(X,0,2), 1e-12);
    
    % Row-wise z-score per modality
    mrZ  = zscore_rows(thisMRWarp);
    vidZ = zscore_rows(thisVidWarp);
    if useTrimodal
        audZ = zscore_rows(thisAudio);
    else
        audZ = [];
    end
    
    % NaN/Inf guard
    if any(~isfinite(mrZ(:))) || any(~isfinite(vidZ(:))) || (~isempty(audZ) && any(~isfinite(audZ(:))))
        error('NaNs/Infs detected in features; fix extraction or impute before PCA.');
    end
    
    % Cull constant rows AFTER z-scoring
    isVar = @(Z) var(Z,0,2) > 1e-8;
    mrZ  = mrZ(isVar(mrZ), :);
    vidZ = vidZ(isVar(vidZ), :);
    if ~isempty(audZ), audZ = audZ(isVar(audZ), :); end
    
    if VERBOSE
        fprintf('VERBOSE: Z-score kept rows -> MR:%d, VID:%d, AUD:%d\n', ...
                size(mrZ,1), size(vidZ,1), size(audZ,1));
        bad = @(Z) sum(~isfinite(Z(:)));
        fprintf('VERBOSE: Non-finite (should be 0) -> MR:%d, VID:%d, AUD:%d\n', ...
                bad(mrZ), bad(vidZ), bad(audZ));
        vstats = @(Z) [min(var(Z,0,2)), median(var(Z,0,2)), max(var(Z,0,2))];
        vm = vstats(mrZ); vv = vstats(vidZ); 
        fprintf('VERBOSE: Row-var ranges (min|med|max) -> MR:%.3g|%.3g|%.3g  VID:%.3g|%.3g|%.3g', ...
                vm(1),vm(2),vm(3), vv(1),vv(2),vv(3));
        if ~isempty(audZ)
            va = vstats(audZ);
            fprintf('  AUD:%.3g|%.3g|%.3g', va(1),va(2),va(3));
        end
        fprintf('\n');
    end
    
    % Block scaling by 1/sqrt(#rows)
    wmr  = 1 / sqrt(size(mrZ, 1));
    wvid = 1 / sqrt(size(vidZ, 1));
    waud = 1;
    if useTrimodal
        waud = waudScale / sqrt(size(audZ, 1));
    end
    
    % Weighted stacks
    mrW  = wmr  * mrZ;
    vidW = wvid * vidZ;
    if useTrimodal
        audW = waud * audZ;
    else
        audW = [];
    end
    
    blockScales = [wmr, wvid, waud];
    
    if VERBOSE
        fprintf('VERBOSE: Block weights -> wMR=%.4g  wVID=%.4g  wAUD=%.4g\n', wmr, wvid, waud);
        tr = @(Z) sum(var(Z,0,2));
        trZ = [tr(mrZ), tr(vidZ), 0];
        trW = [tr(mrW), tr(vidW), 0];
        if useTrimodal
            trZ(3) = tr(audZ);
            trW(3) = tr(audW);
        end
        fprintf('VERBOSE: Trace pre-weight -> MR:%.3g VID:%.3g AUD:%.3g | post-weight -> MR:%.3g VID:%.3g AUD:%.3g\n',...
                trZ(1),trZ(2),trZ(3), trW(1),trW(2),trW(3));
    end
    
    % Concatenate for PCA
    if useTrimodal
        mixWarpsW = [mrW; vidW; audW];
    else
        mixWarpsW = [mrW; vidW];
    end
    
    % Set element boundaries
    p_mr  = size(mrW, 1);
    p_vid = size(vidW, 1);
    
    if useTrimodal
        p_aud = size(audW, 1);
        elementBoundaries = [0, p_mr, p_mr + p_vid, p_mr + p_vid + p_aud];
    else
        elementBoundaries = [0, p_mr, p_mr + p_vid];
    end
    
    if VERBOSE
        [D,T] = size(mixWarpsW);
        fprintf('VERBOSE: Concat -> D=%d rows, T=%d frames. Boundaries: ', D, T);
        fprintf('%d ', elementBoundaries); fprintf('\n');
        fprintf('VERBOSE: Block sizes -> MR:%d VID:%d AUD:%d (0 if not used)\n', p_mr, p_vid, size(audW,1));
        maxPC = min(D, max(1, T-1));
        fprintf('VERBOSE: Component ceiling -> min(D=%d, T-1=%d) = %d; using nComp=%d\n', D, T-1, maxPC, nComp);
    end
    
    % Sanity check
    fprintf('Sanity: useTrimodal=%d | reconstructInd=%d | frames T=%d | nComp=%d\n', ...
            useTrimodal, reconstructId, T, nComp);
    if nComp >= T
        error('Degenerate setup: nComp (%d) >= T (%d). Choose a smaller nComp.', nComp, T);
    end
    
    %% === PCA (time-cov SVD) ===
    [origPCA, origMorphMean, origloadings] = doPCA(mixWarpsW, nComp, VERBOSE);
    
    %% === Mask & Score Inference ===
    % Hide the target block when inferring scores
    tarIdx  = (elementBoundaries(reconstructId)+1) : elementBoundaries(reconstructId+1);
    obs_idx = true(size(mixWarpsW,1),1);
    obs_idx(tarIdx) = false;
    
    if VERBOSE
        fprintf('VERBOSE: Mask: hidden rows=%d (target block), observed rows=%d\n', ...
                numel(tarIdx), sum(obs_idx));
        % Leak check
        if any(obs_idx(tarIdx)), error('Mask error: target rows marked as observed.'); end
    end
    
    %% === Reconstruction & Metrics ===
    A_obs  = infer_scores_from_observed(origPCA, origMorphMean, mixWarpsW, obs_idx, lambda, VERBOSE);
    Xhat_w = origPCA * A_obs + origMorphMean;  % reconstructed full data in WEIGHTED z-space
    
    % Undo block weight -> back to z-space for target block
    XhatB_z = Xhat_w(tarIdx,:) / blockScales(reconstructId);
    
    % Ground-truth in z-space
    switch reconstructId
        case 1, XtrueB_z = mrZ;
        case 2, XtrueB_z = vidZ;
        case 3, XtrueB_z = audZ;
    end
    
    % Row-wise correlations and robust aggregation
    r_rows = arrayfun(@(i) corr(XtrueB_z(i,:).', XhatB_z(i,:).'), 1:size(XtrueB_z,1));
    r_med  = median(r_rows, 'omitnan');
    r_vec  = corr(XtrueB_z(:), XhatB_z(:), 'Rows','complete');
    
    if VERBOSE
        q = quantile(r_rows(~isnan(r_rows)), [0.1 0.5 0.9]);
        fracNeg = mean(r_rows < 0);
        fprintf('VERBOSE: Row-r summary: 10th=%.3f, 50th=%.3f, 90th=%.3f | %%rows<0: %.2f%% | NaN rows: %d\n', ...
                q(1), q(2), q(3), 100*fracNeg, sum(isnan(r_rows)));
    end
    
    % Dynamic label
    blkNames = {'MR','Video','Audio'};
    modeName = 'Bimodal (MR+Video)';
    if useTrimodal, modeName = 'Trimodal (MR+Video+Audio)'; end
    present  = [true,true,useTrimodal];
    present(reconstructId) = false;
    fromList = strjoin(blkNames(present), '+');
    
    fprintf('\n[%s] Reconstruct %s from %s | median r(z)=%.4f | vectorised r(z)=%.4f\n', ...
            modeName, blkNames{reconstructId}, fromList, r_med, r_vec);
    
    % Non-shuffled reconstruction (for loadings comparison)
    partial_data = mixWarpsW;
    partial_data(elementBoundaries(reconstructId)+1 : elementBoundaries(reconstructId+1), :) = 0;
    
    partialMorphMean = mean(partial_data, 2);
    partial_centered = bsxfun(@minus, partial_data, partialMorphMean);
    partial_loading  = partial_centered' * origPCA;
    
    results.nonShuffledLoadings     = origloadings;
    results.nonShuffledReconLoadings = partial_loading;
    
    % Display
    figure;
    plot(results.nonShuffledLoadings, results.nonShuffledReconLoadings, '.');
    hline = refline(1,0); hline.Color = 'k';
    xlabel('Original loadings'); ylabel('Reconstructed loadings');
    title(sprintf('Non-shuffled (reconstruct modality %d: 1=MR,2=Video,3=Audio)', reconstructId));
    
    %% === Shuffle Baseline ===
    nFrames = T;
    permIndexes = NaN(nBoots, nFrames);
    for bootI = 1:nBoots
        permIndexes(bootI,:) = randperm(nFrames);
    end
    
    if VERBOSE
        fprintf('VERBOSE: Shuffle target: %s\n', blkNames{shuffleTarget});
        idperm = mean(permIndexes(1,:) == 1:size(permIndexes,2));
        dispMeanJump = mean(abs(permIndexes(1,:) - (1:size(permIndexes,2))));
        fprintf('VERBOSE: Shuffle check (boot 1): fixed-point frac=%.3f, mean index displacement=%.1f\n', ...
                idperm, dispMeanJump);
    end
    
    % Preallocate
    r_boot = zeros(nBoots,1);
    allShuffledOrigLoad  = cell(nBoots,1);
    allShuffledReconLoad = cell(nBoots,1);
    
    % Get ground truth for target block
    switch reconstructId
        case 1, XtrueB_z = mrZ;
        case 2, XtrueB_z = vidZ;
        case 3, XtrueB_z = audZ;
    end
    
    % Bootstrap loop
    tic
    if HAS_PAR
        parfor bootI = 1:nBoots
            [r_boot(bootI), allShuffledOrigLoad{bootI}, allShuffledReconLoad{bootI}] = ...
                run_one_bootstrap(bootI, permIndexes(bootI,:), shuffleTarget, ...
                                  mrW, vidW, audW, nComp, tarIdx, obs_idx, ...
                                  blockScales, reconstructId, XtrueB_z, lambda, false);
        end
    else
        for bootI = 1:nBoots
            [r_boot(bootI), allShuffledOrigLoad{bootI}, allShuffledReconLoad{bootI}] = ...
                run_one_bootstrap(bootI, permIndexes(bootI,:), shuffleTarget, ...
                                  mrW, vidW, audW, nComp, tarIdx, obs_idx, ...
                                  blockScales, reconstructId, XtrueB_z, lambda, false);
        end
    end
    toc
    
    % Report baseline
    ci = quantile(r_boot,[.025 .975]);
    p_right = mean(r_boot >= r_vec);
    fprintf('Block-level shuffle baseline: median=%.4f, CI=[%.4f, %.4f], p(R_boot >= R_real)=%.4f\n', ...
            median(r_boot), ci(1), ci(2), p_right);
    
    if VERBOSE
        fprintf('VERBOSE: Shuffle r(z): median=%.4f, IQR=[%.4f, %.4f], %%<0 = %.2f%%\n', ...
                median(r_boot), quantile(r_boot,0.25), quantile(r_boot,0.75), 100*mean(r_boot<0));
    end
    
    results.allShuffledOrigLoad  = allShuffledOrigLoad;
    results.allShuffledReconLoad = allShuffledReconLoad;
    
    %% === Reporting ===
    % Statistics on loadings
    loadings1D      = results.nonShuffledLoadings(:);
    reconLoadings1D = results.nonShuffledReconLoadings(:);
    
    SSE = sum((loadings1D - reconLoadings1D).^2);
    [R,~] = corr(loadings1D, reconLoadings1D);
    pfit = polyfit(loadings1D, reconLoadings1D, 1);
    
    unshuffstats = [R, pfit(1), SSE];
    
    shuffstats = NaN(3, nBoots);
    for bootI = 1:nBoots
        loadings1D_s      = results.allShuffledOrigLoad{bootI}(:);
        reconLoadings1D_s = results.allShuffledReconLoad{bootI}(:);
        SSE_s = sum((loadings1D_s - reconLoadings1D_s).^2);
        [R_s, ~] = corr(loadings1D_s, reconLoadings1D_s);
        pfit_s = polyfit(loadings1D_s, reconLoadings1D_s, 1);
        shuffstats(:,bootI) = [R_s, pfit_s(1), SSE_s];
    end
    
    fprintf('\n=== LOADINGS-BASED METRICS (faithful to original) ===\n');
    fprintf('Correlation (R): %.6f\n', R);
    fprintf('Linear fit gradient: %.4f\n', pfit(1));
    fprintf('Sum of squared error (SSE): %.4f\n', SSE);
    
    % Probe with NO observed rows
    obs_idx_none = false(size(mixWarpsW,1),1);
    A0 = infer_scores_from_observed(origPCA, origMorphMean, mixWarpsW, obs_idx_none, 1e-3, false);
    Xhat0_w  = origPCA * A0 + origMorphMean;
    Xhat0B_z = Xhat0_w(tarIdx,:) / blockScales(reconstructId);
    r_none = corr(XtrueB_z(:), Xhat0B_z(:));
    fprintf('Probe: r with NO observed rows (expect near 0): %.4f\n', r_none);
    
    % Display histograms
    figure;
    subplot(2,3,2);
    plot(results.nonShuffledLoadings, results.nonShuffledReconLoadings, '.');
    hline = refline(1,0); hline.Color = 'k';
    xlabel('Original loadings'); ylabel('Reconstructed loadings');
    title('Non-shuffled');
    
    statStrings = {'Correlation coefficient','Linear Fit Gradient','SSE'};
    for statI = 1:3
        subplot(2,3,statI+3);
        histogram(shuffstats(statI,:), 50); hold on; axis tight;
        plot(unshuffstats([statI statI]), ylim, 'r--', 'linewidth', 2);
        xlabel(statStrings{statI}); ylabel('Frequency');
        title('Shuffled bootstrap');
    end
end
end

%% === Helper Functions ===

function [prinComp, MorphMean, loadings] = doPCA(dataW, nComp, VERBOSE)
% Time-cov SVD with enforced component cap and reporting.
% Inputs:  dataW (p x T), nComp (scalar), VERBOSE (logical)
% Outputs: prinComp (p x nComp), MorphMean (p x 1), loadings (T x nComp)

    MorphMean = mean(dataW, 2);
    Xc = bsxfun(@minus, dataW, MorphMean);
    xxt = Xc' * Xc;                          % T×T
    [~, S, V] = svd(xxt, 'econ');            % V: T×T
    T = size(V,2);
    
    if nComp >= T
        error('nComp (%d) must be < T (%d) to avoid trivial reconstruction. Reduce nComp.', nComp, T);
    end
    
    K = nComp;
    Vk   = V(:, 1:K);
    LInv = 1 ./ sqrt(diag(S(1:K,1:K)) + eps);
    prinComp = Xc * Vk * diag(LInv);         % p×K
    loadings = Xc' * prinComp;               % T×K
    
    if VERBOSE
        sk = diag(S);
        top = min(5, numel(sk));
        frac = sum(sk(1:K)) / sum(sk);
        fprintf('VERBOSE: PCA top-%d singular values: ', top);
        fprintf('%.3g ', sk(1:top)); fprintf('\n');
        fprintf('VERBOSE: PCA cumulative time-cov energy captured by K=%d: %.2f%%\n', K, 100*frac);
        orthoErr = norm(prinComp' * prinComp - eye(K), 'fro');
        fprintf('VERBOSE: PCA ||P''P - I||_F = %.2e (expect ~0)\n', orthoErr);
    end
end

function A = infer_scores_from_observed(P, mu, X, obs_idx, lambda, VERBOSE)
% Infer PCA scores from observed rows only
% Inputs:  P (p x q), mu (p x 1), X (p x T), obs_idx (p x 1 logical), lambda (scalar), VERBOSE (logical)
% Output:  A (q x T)

    Xc = bsxfun(@minus, X, mu);
    Pobs = P(obs_idx,:);                 % (p_obs x q)
    Xtobs = Xc(obs_idx,:);               % (p_obs x T)
    
    if VERBOSE
        q = size(P,2); T = size(X,2);
        fprintf('VERBOSE: Scoring: q=%d comps, obs_rows=%d, T=%d, lambda=%.2e\n', ...
                q, sum(obs_idx), T, lambda);
    end
    
    M = (Pobs.'*Pobs) + lambda*eye(size(P,2));  % (q x q)
    
    if VERBOSE
        e = eig(M); e = sort(real(e),'ascend');
        fprintf('VERBOSE: Scoring eig(M) min=%.3g med=%.3g max=%.3g\n', ...
                e(1), median(e), e(end));
    end
    
    A = M \ (Pobs.' * Xtobs);            % (q x T)
end

function [r_val, orig_load, recon_load] = run_one_bootstrap(bootI, perm, shuffleTarget, ...
                                                             mrW, vidW, audW, nComp, tarIdx, obs_idx, ...
                                                             blockScales, reconstructId, XtrueB_z, lambda, VERBOSE)
% Run a single bootstrap iteration
% Returns: r_val (scalar correlation), orig_load (T x nComp), recon_load (T x nComp)

    % Build shuffled stack
    switch shuffleTarget
        case 1  % shuffle MR frames
            shuffWarps = [mrW(:,perm); vidW; audW];
        case 2  % shuffle Video frames
            shuffWarps = [mrW; vidW(:,perm); audW];
        case 3  % shuffle Audio frames
            shuffWarps = [mrW; vidW; audW(:,perm)];
        otherwise
            error('Invalid shuffleTarget.');
    end
    
    % Fit PCA on shuffled data
    [PCk, mean_k, loadings] = doPCA(shuffWarps, nComp, VERBOSE);
    
    % r(z) side: infer scores from observed rows
    Ak = infer_scores_from_observed(PCk, mean_k, shuffWarps, obs_idx, lambda, VERBOSE);
    Xhatk = PCk * Ak + mean_k;                              % weighted z-space
    XhatkB_z = Xhatk(tarIdx,:) / blockScales(reconstructId);
    r_val = corr(XtrueB_z(:), XhatkB_z(:), 'Rows','complete');
    
    % Loadings side: replicate original partial-loadings method
    partial_data_s = shuffWarps;
    partial_data_s(tarIdx, :) = 0;                          % hide target block
    partialMorphMean_s = mean(partial_data_s, 2);
    partial_centered_s = bsxfun(@minus, partial_data_s, partialMorphMean_s);
    partial_loading_s  = partial_centered_s' * PCk;
    
    orig_load = loadings;
    recon_load = partial_loading_s;
end