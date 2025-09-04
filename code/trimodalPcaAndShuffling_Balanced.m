function trimodalPcaAndShuffling_Balanced
% Minimal extension of the original bimodal PCA+shuffle example to a
% trimodal setup (MR, Video, Audio) with **block scaling** so audio has a
% fair contribution. The PCA routine and the evaluation (loadings plots,
% shuffle bootstrap) are kept faithful to the original structure.


clc;

% ---- User paths/files ----
dataDir  = '/Users/jaker/Research-Project/data';
dataFile = 'mrAndVideoData.mat';
audioFile = 'audioFeaturesData_articulatory.mat';

% ---- Load data ----
addpath(dataDir);
load(dataFile,'data');            % MR + Video
load(audioFile,'audioData');      % Audio


% ---- Controls  ----
usePar        = true;               % parallel processing if available
useTrimodal = true;                 % false => bimodal (MR+Video), true => trimodal (MR+Video+Audio)
reconstructId = 1;                 % 1 = MR, 2 = Video, 3 = Audio
shuffleTarget = 1;     % chose which block to shuffle 
nComp = 39;                         % number of principal components to keep (try 20, 40, 80)
nBoots = 500;                       % # bootstraps
lambda = 1e-3;
% (Diagnostic) Audio weight multiplier for H1 ablation; set to 1 for default runs.
waudScale = 1;


global VERBOSE; 

VERBOSE = false;


% === PREPROCESSING MODE ===
% 'legacy'        -> faithful to original bimodal (mean-center only; no z-score; no block scaling)
% 'zscore_sqrtN'  -> your current default (row-wise z-score; block scaling by 1/sqrt(#rows))
preprocMode = 'zscore_sqrtN';   % change to 'legacy' for apples-to-apples with the original SHOULD REPLACE THIS WITH Z SCORE OR SOMETHING 

% ---- Reproducibility ----
rng('default');



% ---- Iterate (kept identical loop style; adjust index as needed) ----
for ii = 9 % :length(data)

    clear results;

    % ---- Select this item ----
    thisMRWarp = data(ii).mr_warp2D;            % (p_mr x T)
    thisVidWarp = data(ii).vid_warp2D;          % (p_vid x T)
    thisAudio   = audioData(ii).audioFeatures_articulatory'; % (p_aud x T)

    % ---- Basic shape checks ----
    if size(thisMRWarp,2) ~= size(thisVidWarp,2) || size(thisMRWarp,2) ~= size(thisAudio,2)
        error('Frame count mismatch across modalities for item %d.', ii);
    end

    % ---- Report shapes ----
    fprintf('Item %d\n', ii);
    fprintf('MR:    %d features x %d frames\n', size(thisMRWarp,1), size(thisMRWarp,2));
    fprintf('Video: %d features x %d frames\n', size(thisVidWarp,1), size(thisVidWarp,2));
    fprintf('Audio: %d features x %d frames\n', size(thisAudio,1),   size(thisAudio,2));


    %% === PREPROCESSING ===

   
    % Safe per-row z-score helper (used only where needed)
    zscore_rows = @(X) (X - mean(X,2)) ./ max(std(X,0,2), 1e-12);
    
    % Sizes
    nMR  = size(thisMRWarp, 1);
    nVID = size(thisVidWarp, 1);
    useAudio = (exist('useTrimodal','var') && useTrimodal) && exist('thisAudio','var') && ~isempty(thisAudio);
    nAUD = useAudio * size(thisAudio, 1);  % 0 if no audio
    
    switch preprocMode  
        case 'legacy'       % Not even sure if this is used at all 
            % === PCA input (no z-score, no block scaling) ===
            mrW  = thisMRWarp;
            vidW = thisVidWarp;
            if useAudio, audW = thisAudio; else, audW = []; end
    
            % Scales kept for downstream "unweighting" convenience
            blockScales = [1, 1, 1];
    
            % Optional: z-scored copies only for evaluation (e.g., r(z))
            mrZ  = zscore_rows(thisMRWarp);
            vidZ = zscore_rows(thisVidWarp);
            if useAudio, audZ = zscore_rows(thisAudio); else, audZ = []; end
            
            % --- NaN/Inf guard (fail fast if any remain after feature extraction) ---
            if any(~isfinite(mrZ(:))) || any(~isfinite(vidZ(:))) || (~isempty(audZ) && any(~isfinite(audZ(:))))
                error('NaNs/Infs detected in features; fix extraction or impute before PCA.');
            end
            
            % --- Constant-row guard (prevents NaNs in correlations/p-values) ---
            keep = @(Z) Z(var(Z,0,2) > 1e-12, :);
            mrZ  = keep(mrZ);
            vidZ = keep(vidZ);
            if ~isempty(audZ), audZ = keep(audZ); end

    
        case 'zscore_sqrtN'
            % === Row-wise z-score per modality ===
            mrZ  = zscore_rows(thisMRWarp);
            vidZ = zscore_rows(thisVidWarp);
            if useAudio, audZ = zscore_rows(thisAudio); else, audZ = []; end

            % --- NaN/Inf guard (fail fast) ---
            if any(~isfinite(mrZ(:))) || any(~isfinite(vidZ(:))) || (~isempty(audZ) && any(~isfinite(audZ(:))))
                error('NaNs/Infs detected in features; fix extraction or impute before PCA.');
            end

            % === Constant-row cull AFTER z-scoring ===
            isVar = @(Z) var(Z,0,2) > 1e-8;
            mrZ  = mrZ(isVar(mrZ), :);
            vidZ = vidZ(isVar(vidZ), :);
            if ~isempty(audZ), audZ = audZ(isVar(audZ), :); end

            if VERBOSE
                fprintf('Z-score: kept rows -> MR:%d, VID:%d, AUD:%d\n', size(mrZ,1), size(vidZ,1), size(audZ,1));
                bad = @(Z) sum(~isfinite(Z(:)));
                fprintf('Non-finite (should be 0) -> MR:%d, VID:%d, AUD:%d\n', bad(mrZ), bad(vidZ), bad(audZ));
                vstats = @(Z) [min(var(Z,0,2)), median(var(Z,0,2)), max(var(Z,0,2))];
                vm = vstats(mrZ); vv = vstats(vidZ); va = vstats(audZ);
                fprintf('Row-var ranges (min|med|max) -> MR:%.3g|%.3g|%.3g  VID:%.3g|%.3g|%.3g  AUD:%.3g|%.3g|%.3g\n',...
                    vm(1),vm(2),vm(3), vv(1),vv(2),vv(3), va(1),va(2),va(3));
            end


            % === Block scaling by 1/sqrt(#rows) (use counts AFTER cull) ===
            wmr  = 1 / sqrt(size(mrZ, 1));
            wvid = 1 / sqrt(size(vidZ, 1));
            waud = 1; if useAudio, waud = 1 / sqrt(size(audZ, 1)); end

            % (Optional audio-weight ablation multiplier; see Change 2)
            if exist('waudScale','var') && ~isempty(waudScale), waud = waud * waudScale; end        % Not sure about this  




            % Weighted stacks (z-space × weights)
            mrW  = wmr  * mrZ;
            vidW = wvid * vidZ;
            if useAudio, audW = waud * audZ; else, audW = []; end

            if VERBOSE
                fprintf('Block weights -> wMR=%.4g  wVID=%.4g  wAUD=%.4g\n', wmr, wvid, waud);
                % Energy (trace) before/after weighting
                tr = @(Z) sum(var(Z,0,2));
                trZ = [tr(mrZ), tr(vidZ), tr(audZ)];
                trW = [tr(mrW), tr(vidW), tr(audW)];
                fprintf('Trace pre-weight -> MR:%.3g VID:%.3g AUD:%.3g | post-weight -> MR:%.3g VID:%.3g AUD:%.3g\n',...
                    trZ(1),trZ(2),trZ(3), trW(1),trW(2),trW(3));
            end


            blockScales = [wmr, wvid, waud];

    
        otherwise
            error('Unknown preprocMode: %s', preprocMode);
    end
    
    % === Concatenate for PCA ===
    if useAudio
        mixWarpsW = [mrW; vidW; audW];
    else
        mixWarpsW = [mrW; vidW];
    end

    


    % ---- Set element boundaries ----
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
        fprintf('Concat -> D=%d rows, T=%d frames. Boundaries: ', D, T);
        fprintf('%d ', elementBoundaries); fprintf('\n');
        p_mr  = size(mrW,1); p_vid = size(vidW,1); p_aud = size(audW,1);
        fprintf('Block sizes -> MR:%d VID:%d AUD:%d (0 if not used)\n', p_mr, p_vid, p_aud);
    end




    % ---- Sanity: report frames and enforce nComp<T at the call site too ----
    T = size(mixWarpsW, 2);
    fprintf('Sanity: useTrimodal=%d | reconstructInd=%d | frames T=%d | nComp=%d\n', useTrimodal, reconstructId, T, nComp);
    if nComp >= T
        error('Degenerate setup: nComp (%d) >= T (%d). Choose a smaller nComp.', nComp, T);
    end

    if VERBOSE
        D = size(mixWarpsW,1); T = size(mixWarpsW,2);
        maxPC = min(D, max(1, T-1));
        fprintf('Component ceiling -> min(D=%d, T-1=%d) = %d; using nComp=%d\n', D, T-1, maxPC, nComp);
    end





    % ---- PCA on hybrid data (kept identical) ----
    [origPCA,origMorphMean,origloadings] = doPCA(mixWarpsW, nComp); 

  

    % --- Objective block reconstruction metric (z-space; unit-consistent) ---
    % Hide the target block when inferring scores, using boundaries defined earlier.
    tarIdx  = (elementBoundaries(reconstructId)+1) : elementBoundaries(reconstructId+1);
    obs_idx = true(size(mixWarpsW,1),1);
    obs_idx(tarIdx) = false;
    
    if VERBOSE
        fprintf('Mask: hidden rows=%d (target block), observed rows=%d\n', ...
            numel(tarIdx), sum(obs_idx));
        % Simple leak check: ensure hidden block really excluded
        if any(obs_idx(tarIdx)), error('Mask error: target rows marked as observed.'); end
    end


    A_obs  = infer_scores_from_observed(origPCA, origMorphMean, mixWarpsW, obs_idx, lambda);
    Xhat_w = origPCA * A_obs + origMorphMean;     % reconstructed full data in WEIGHTED z-space
    
    % Undo only the block weight -> back to z-space for the target block
    XhatB_z = Xhat_w(tarIdx,:) / blockScales(reconstructId);
    
    % Ground-truth in z-space for that block
    switch reconstructId
        case 1, XtrueB_z = mrZ;
        case 2, XtrueB_z = vidZ;
        case 3, XtrueB_z = audZ;
    end
    
    % Row-wise correlations and robust aggregation (median); also report vectorised r in z-space
    r_rows   = arrayfun(@(i) corr(XtrueB_z(i,:).', XhatB_z(i,:).'), 1:size(XtrueB_z,1));
    r_med    = median(r_rows, 'omitnan');
    r_vec = corr(XtrueB_z(:), XhatB_z(:), 'Rows','complete');

    if VERBOSE
        q = quantile(r_rows(~isnan(r_rows)), [0.1 0.5 0.9]);
        fracNeg = mean(r_rows < 0);
        fprintf('Row-r summary: 10th=%.3f, 50th=%.3f, 90th=%.3f | %%rows<0: %.2f%% | NaN rows: %d\n', ...
            q(1), q(2), q(3), 100*fracNeg, sum(isnan(r_rows)));
    end

    
    % Dynamic, accurate label
    blkNames = {'MR','Video','Audio'};
    modeName = 'Bimodal (MR+Video)'; if useTrimodal, modeName = 'Trimodal (MR+Video+Audio)'; end
    present  = [true,true,useTrimodal]; present(reconstructId) = false;
    fromList = strjoin(blkNames(present), '+');
    
    fprintf('\n[%s] Reconstruct %s from %s | median r(z)=%.4f | vectorised r(z)=%.4f\n', ...
        modeName, blkNames{reconstructId}, fromList, r_med, r_vec);

    % ===============================
    % Non-shuffled reconstruction (faithful to original pattern)
    nFrames = size(thisMRWarp, 2);
    partial_data = mixWarpsW;
    % Zero the block we intend to "reconstruct"
    partial_data(elementBoundaries(reconstructId)+1 : elementBoundaries(reconstructId+1), :) = 0;

    % Recompute mean on the partially zeroed data,
    % then recenter and project to get "reconstructed loadings"
    partialMorphMean = mean(partial_data, 2);
    partial_centered = bsxfun(@minus, partial_data, partialMorphMean);
    partial_loading  = partial_centered' * origPCA;

    % Store results 
    results.nonShuffledLoadings     = origloadings;
    results.nonShuffledReconLoadings = partial_loading;

    % ---- Display (faithful scatter with unity line) ----
    figure;
    plot(results.nonShuffledLoadings, results.nonShuffledReconLoadings, '.');
    hline = refline(1,0); hline.Color = 'k';
    xlabel('Original loadings'); ylabel('Reconstructed loadings');
    title(sprintf('Non-shuffled (reconstruct modality %d: 1=MR,2=Video,3=Audio)', reconstructId));


    %% Shuffled reconstructions (bootstrap) — unified loop for r(z) and loadings
    permIndexes = NaN(nBoots, nFrames);
    for bootI = 1:nBoots
        permIndexes(bootI,:) = randperm(nFrames);
    end

    if VERBOSE          % This one is temporary for some reason 
        blkNames = {'MR','Video','Audio'};
        fprintf('Shuffle target: %s\n', blkNames{shuffleTarget});
        idperm = mean(permIndexes(1,:) == 1:size(permIndexes,2));   % fraction fixed points in first perm
        dispMeanJump = mean(abs(permIndexes(1,:) - (1:size(permIndexes,2))));
        fprintf('Shuffle check (boot 1): fixed-point frac=%.3f, mean index displacement=%.1f\n', idperm, dispMeanJump);
    end

    
    % Hidden block index range
    tarIdx  = (elementBoundaries(reconstructId)+1) : elementBoundaries(reconstructId+1);
    switch reconstructId
        case 1, XtrueB_z = mrZ;
        case 2, XtrueB_z = vidZ;
        case 3, XtrueB_z = audZ;
    end
    
    % Choose which block to shuffle.
    % Default (H1): shuffle the block being reconstructed.

    % --- To run H2 control (e.g., reconstruct MR/Video but shuffle AUDIO), you may set:
    % if reconstructInd ~= 3 && useTrimodal, shuffleTarget = 3; end
    
    % Preallocate for both families of metrics
    r_boot = zeros(nBoots,1);
    allShuffledOrigLoad  = cell(nBoots,1);
    allShuffledReconLoad = cell(nBoots,1);
    
    nCores = feature('numcores');
    tic
    if usePar && nCores > 2
        disp('Using parallel processing.');
        poolOpen = gcp('nocreate'); if isempty(poolOpen), parpool(nCores-1); end
    
        parfor bootI = 1:nBoots
            % ---- Build shuffled stack (only selected block) ----
            switch shuffleTarget
                case 1 % shuffle MR frames
                    shuffWarps = [mrW(:,permIndexes(bootI,:)); vidW; audW];
                case 2 % shuffle Video frames
                    shuffWarps = [mrW; vidW(:,permIndexes(bootI,:)); audW];
                case 3 % shuffle Audio frames
                    shuffWarps = [mrW; vidW; audW(:,permIndexes(bootI,:))];
                otherwise
                    error('Invalid shuffleTarget.');
            end
    
            % ---- Fit PCA on shuffled data ----
            [PCk, mean_k, loadings] = doPCA(shuffWarps, nComp);
    
            % ---- r(z) side: infer scores from observed rows, reconstruct hidden block ----
            Ak    = infer_scores_from_observed(PCk, mean_k, shuffWarps, obs_idx, 1e-3); % lambda as used above
            Xhatk = PCk * Ak + mean_k;                              % weighted z-space
            XhatkB_z = Xhatk(tarIdx,:) / blockScales(reconstructId);
            r_boot(bootI) = corr(XtrueB_z(:), XhatkB_z(:), 'Rows','complete');
    
            % ---- loadings side: replicate original partial-loadings method ----
            partial_data_s = shuffWarps;
            partial_data_s(tarIdx, :) = 0;                          % hide target block
            partialMorphMean_s = mean(partial_data_s, 2);
            partial_centered_s = bsxfun(@minus, partial_data_s, partialMorphMean_s);
            partial_loading_s  = partial_centered_s' * PCk;
    
            allShuffledOrigLoad{bootI}  = loadings;
            allShuffledReconLoad{bootI} = partial_loading_s;
        end
    else
        disp('NOT using parallel processing.');
        for bootI = 1:nBoots
            switch shuffleTarget
                case 1
                    shuffWarps = [mrW(:,permIndexes(bootI,:)); vidW; audW];
                case 2
                    shuffWarps = [mrW; vidW(:,permIndexes(bootI,:)); audW];
                case 3
                    shuffWarps = [mrW; vidW; audW(:,permIndexes(bootI,:))];
            end
    
            [PCk, mean_k, loadings] = doPCA(shuffWarps, nComp);
    
            Ak    = infer_scores_from_observed(PCk, mean_k, shuffWarps, obs_idx, 1e-3);
            Xhatk = PCk * Ak + mean_k;
            XhatkB_z = Xhatk(tarIdx,:) / blockScales(reconstructId);
            r_boot(bootI) = corr(XtrueB_z(:), XhatkB_z(:), 'Rows','complete');
    
            partial_data_s = shuffWarps;
            partial_data_s(tarIdx, :) = 0;
            partialMorphMean_s = mean(partial_data_s, 2);
            partial_centered_s = bsxfun(@minus, partial_data_s, partialMorphMean_s);
            partial_loading_s  = partial_centered_s' * PCk;
    
            allShuffledOrigLoad{bootI}  = loadings;
            allShuffledReconLoad{bootI} = partial_loading_s;
        end
    end
    toc
    
    % Report baseline vs real
    ci = quantile(r_boot,[.025 .975]);
    p_right = mean(r_boot >= r_vec);  % or compare to r_med if preferred
    fprintf('Block-level shuffle baseline: median=%.4f, CI=[%.4f, %.4f], p(R_boot >= R_real)=%.4f\n', ...
            median(r_boot), ci(1), ci(2), p_right);
    

    if VERBOSE
        fprintf('Shuffle r(z): median=%.4f, IQR=[%.4f, %.4f], %%<0 = %.2f%%\n', ...
            median(r_boot), quantile(r_boot,0.25), quantile(r_boot,0.75), 100*mean(r_boot<0));
    end

    
    % Store both families of bootstrap artefacts for the later stats/plots
    results.allShuffledOrigLoad  = allShuffledOrigLoad;
    results.allShuffledReconLoad = allShuffledReconLoad;


    %% Statistics 
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

    % ---- Probe: with NO observed rows, r should be ~0 ----
    obs_idx_none = false(size(mixWarpsW,1),1);
    A0 = infer_scores_from_observed(origPCA, origMorphMean, mixWarpsW, obs_idx_none, 1e-3);
    Xhat0_w  = origPCA * A0 + origMorphMean;
    tarIdx   = (elementBoundaries(reconstructId)+1) : elementBoundaries(reconstructId+1);
    Xhat0B_z = Xhat0_w(tarIdx,:) / blockScales(reconstructId);
    switch reconstructId
        case 1, XtrueB_z = mrZ; case 2, XtrueB_z = vidZ; case 3, XtrueB_z = audZ;
    end
    r_none = corr(XtrueB_z(:), Xhat0B_z(:));
    fprintf('Probe: r with NO observed rows (expect near 0): %.4f\n', r_none);


    % ---- Display histograms ----
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





function [prinComp, MorphMean, loadings] = doPCA(dataW, nComp)

VERBOSE = true;
% Time-cov SVD with enforced component cap and reporting.
    MorphMean = mean(dataW, 2);
    Xc = bsxfun(@minus, dataW, MorphMean);
    xxt = Xc' * Xc;                          % T×T
    [~, S, V] = svd(xxt, 'econ');            % V: T×T
    T = size(V,2);
    
    if nComp >= T
        error('nComp (%d) must be < T (%d) to avoid trivial reconstruction. Reduce nComp.', nComp, T);  % Think this is wrong
    end
    K = nComp;
    Vk   = V(:, 1:K);
    LInv = 1 ./ sqrt(diag(S(1:K,1:K)) + eps);
    prinComp = Xc * Vk * diag(LInv);         % D×K
    loadings = Xc' * prinComp;               % T×K

    
    % After computing prinComp and loadings:
    if exist('VERBOSE','var') && VERBOSE
        sk = diag(S); 
        K  = size(prinComp,2);
        top = min(5, numel(sk));
        frac = sum(sk(1:K)) / sum(sk);
        fprintf('PCA: top-%d singular values: ', top); fprintf('%.3g ', sk(1:top)); fprintf('\n');
        fprintf('PCA: cumulative time-cov energy captured by K=%d: %.2f%%\n', K, 100*frac);
        orthoErr = norm(prinComp' * prinComp - eye(K), 'fro');
        fprintf('PCA: ||P''P - I||_F = %.2e (expect ~0)\n', orthoErr);
    end


end





function A = infer_scores_from_observed(P, mu, X, obs_idx, lambda)

VERBOSE = true;
% P: (p x q) principal components matrix returned as `prinComp` from doPCA
% mu: (p x 1) mean used in that PCA
% X:  (p x T) data you want to score (same variables, weighted)
% obs_idx: logical (p x 1), TRUE for observed rows, FALSE for the "held-out" block
% lambda: small ridge (e.g., 1e-3) for stability
%
% Returns A: (q x T) scores given only the observed rows.

    Xc = bsxfun(@minus, X, mu);
    Pobs = P(obs_idx,:);                 % (p_obs x q)
    Xtobs = Xc(obs_idx,:);               % (p_obs x T)

    if exist('VERBOSE','var') && VERBOSE
        q = size(P,2); T = size(X,2);
        fprintf('Scoring: q=%d comps, obs_rows=%d, T=%d, lambda=%.2e\n', q, sum(obs_idx), T, lambda);
    end



    M = (Pobs.'*Pobs) + lambda*eye(size(P,2));  % (q x q)

    if exist('VERBOSE','var') && VERBOSE
        % Conditioning diagnostic (avoid exact cond(M) if too slow)
        e = eig(M); e = sort(real(e),'ascend');
        fprintf('Scoring: eig(M) min=%.3g med=%.3g max=%.3g\n', e(1), median(e), e(end));
    end
    A = M \ (Pobs.' * Xtobs);            % (q x T)
end
