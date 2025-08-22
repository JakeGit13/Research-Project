function trimodalPcaAndShuffling_Balanced
% Minimal extension of the original bimodal PCA+shuffle example to a
% trimodal setup (MR, Video, Audio) with **block scaling** so audio has a
% fair contribution. The PCA routine and the evaluation (loadings plots,
% shuffle bootstrap) are kept faithful to the original structure.
%
% Assumptions:
% - mrAndVideoData.mat contains struct array `data` with fields:
%     .mr_warp2D  (features x frames)
%     .vid_warp2D (features x frames)
% - audioFeaturesData_articulatory.mat contains struct array `audioData`
%     with field:
%     .audioFeatures_articulatory (frames x features) — will be transposed
%
% Only intentional changes vs original:
%   1) Per-feature z-scoring within each modality (as usual)
%   2) **Block scaling**: divide each modality by sqrt(#features)
%   3) Shuffle the **reconstructed** modality (generalised from the original)

clc;

% ---- User paths/files ----
dataDir  = '/Users/jaker/Research-Project/data';
dataFile = 'mrAndVideoData.mat';
audioFile = 'audioFeaturesData_articulatory.mat';

% ---- Controls (kept from original style) ----
usePar        = true;     % parallel processing if available
reconstructInd = 3;       % 1 = MR, 2 = Video, 3 = Audio
nBoots        = 1000;     % # bootstraps

% ---- Reproducibility ----
rng('default');

% ---- Load data ----
addpath(dataDir);
load(dataFile,'data');            % MR + Video
load(audioFile,'audioData');      % Audio

actors    = [data.actor]; %#ok<NASGU>
sentences = [data.sentence]; %#ok<NASGU>

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

    % ---- Per-feature z-scoring within each modality (as usual) ----
    zscore_rows = @(X) (X - mean(X,2)) ./ max(std(X,0,2), 1e-8);
    mrZ  = zscore_rows(thisMRWarp);
    vidZ = zscore_rows(thisVidWarp);
    audZ = zscore_rows(thisAudio);

    % ---- Block scaling: equalise total variance contribution per block ----
    p_mr  = size(mrZ, 1);
    p_vid = size(vidZ, 1);
    p_aud = size(audZ, 1);

    wmr  = 1 / sqrt(p_mr);
    wvid = 1 / sqrt(p_vid);
    waud = 1 / sqrt(p_aud);

    mrW  = wmr  * mrZ;
    vidW = wvid * vidZ;
    audW = waud * audZ;

    % ---- Concatenate (weighted) ----
    mixWarpsW = [mrW; vidW; audW];

    % ---- Boundaries (no hard-coding) ----
    b1 = p_mr;
    b2 = p_mr + p_vid;
    b3 = p_mr + p_vid + p_aud;
    elementBoundaries = [0 b1 b2 b3];

    % ---- PCA on hybrid data (kept identical) ----
    [origPCA,origMorphMean,origloadings] = doPCA(mixWarpsW); 

  

    % --- Objective block reconstruction metric (minimal add-on) ---
    p_mr  = size(thisMRWarp,1);
    p_vid = size(thisVidWarp,1);
    p_aud = size(thisAudio,1);
    b1 = p_mr; b2 = p_mr+p_vid; b3 = p_mr+p_vid+p_aud;
    elementBoundaries = [0 b1 b2 b3];
    
    % We will "hide" the block indicated by reconstructInd when inferring scores
    obs_idx = true(size(mixWarpsW,1),1);
    obs_idx(elementBoundaries(reconstructInd)+1 : elementBoundaries(reconstructInd+1)) = false;
    
    lambda = 1e-3;
    A_obs = infer_scores_from_observed(origPCA, origMorphMean, mixWarpsW, obs_idx, lambda);
    Xhat  = origMorphMean + origPCA * A_obs;  % (p x T) reconstructed full data
    
    switch reconstructInd
        case 1  % MR from Video+Audio
            mr_hat  = Xhat(1:b1,:)          / (1/sqrt(p_mr));   % unweight
            mr_true = thisMRWarp;
            r_block = corr(mr_true(:), mr_hat(:));
            fprintf('MR reconstruction from Video+Audio (feature corr): %.4f\n', r_block);
    
        case 2  % Video from MR+Audio
            vid_hat  = Xhat(b1+1:b2,:)      / (1/sqrt(p_vid));
            vid_true = thisVidWarp;
            r_block = corr(vid_true(:), vid_hat(:));
            fprintf('Video reconstruction from MR+Audio (feature corr): %.4f\n', r_block);
    
        case 3  % Audio from MR+Video
            aud_hatW = Xhat(b2+1:b3,:);
            aud_hat  = aud_hatW            / (1/sqrt(p_aud));
            aud_true = thisAudio;
            r_block = corr(aud_true(:), aud_hat(:));
            fprintf('Audio reconstruction from MR+Video (feature corr): %.4f\n', r_block);
    end
    
    % --- Shuffle control for the same block (should ~kill r_block) ---
    T = size(mixWarpsW,2);
    perm = randperm(T);
    mixWarpsW_shuf = mixWarpsW;
    switch reconstructInd
        case 1, mixWarpsW_shuf(1:b1,:)      = mixWarpsW_shuf(1:b1,perm);
        case 2, mixWarpsW_shuf(b1+1:b2,:)   = mixWarpsW_shuf(b1+1:b2,perm);
        case 3, mixWarpsW_shuf(b2+1:b3,:)   = mixWarpsW_shuf(b2+1:b3,perm);
    end
    
    [Pshuf, mushuf, ~] = doPCA(mixWarpsW_shuf);               % same PCA routine
    A_obs_sh = infer_scores_from_observed(Pshuf, mushuf, mixWarpsW_shuf, obs_idx, lambda);
    Xhat_sh  = mushuf + Pshuf * A_obs_sh;
    
    switch reconstructInd
        case 1
            mr_hat_sh = Xhat_sh(1:b1,:)        / (1/sqrt(p_mr));
            r_block_sh = corr(thisMRWarp(:), mr_hat_sh(:));
        case 2
            vid_hat_sh = Xhat_sh(b1+1:b2,:)    / (1/sqrt(p_vid));
            r_block_sh = corr(thisVidWarp(:), vid_hat_sh(:));
        case 3
            aud_hat_sh = Xhat_sh(b2+1:b3,:)    / (1/sqrt(p_aud));
            r_block_sh = corr(thisAudio(:), aud_hat_sh(:));
    end
    fprintf('Shuffle baseline (same block): %.4f\n', r_block_sh);


    % ===============================
    % Non-shuffled reconstruction (faithful to original pattern)
    % ===============================
    nFrames = size(thisMRWarp, 2);
    partial_data = mixWarpsW;
    % Zero the block we intend to "reconstruct"
    partial_data(elementBoundaries(reconstructInd)+1 : elementBoundaries(reconstructInd+1), :) = 0;

    % As in the original: recompute mean on the partially zeroed data,
    % then recenter and project to get "reconstructed loadings"
    partialMorphMean = mean(partial_data, 2);
    partial_centered = bsxfun(@minus, partial_data, partialMorphMean);
    partial_loading  = partial_centered' * origPCA;

    % Store results (faithful to original)
    results.nonShuffledLoadings     = origloadings;
    results.nonShuffledReconLoadings = partial_loading;

    % ---- Display (faithful scatter with unity line) ----
    figure;
    plot(results.nonShuffledLoadings, results.nonShuffledReconLoadings, '.');
    hline = refline(1,0); hline.Color = 'k';
    xlabel('Original loadings'); ylabel('Reconstructed loadings');
    title(sprintf('Non-shuffled (reconstruct modality %d: 1=MR,2=Video,3=Audio)', reconstructInd));

    % ===============================
    % Shuffled reconstructions (bootstrap), faithful to original
    % but shuffle the **reconstructed** modality
    % ===============================
    permIndexes = NaN(nBoots, nFrames);
    for bootI = 1:nBoots
        permIndexes(bootI,:) = randperm(nFrames);
    end

    allShuffledOrigLoad  = cell(nBoots,1);
    allShuffledReconLoad = cell(nBoots,1);

    nCores = feature('numcores');
    tic
    if usePar && nCores > 2
        disp('Using parallel processing...');
        poolOpen = gcp('nocreate');
        if isempty(poolOpen), parpool(nCores-1); end

        parfor bootI = 1:nBoots
            % Shuffle the modality being reconstructed
            switch reconstructInd
                case 1 % MR shuffled
                    shuffWarps = [mrW(:,permIndexes(bootI,:));           vidW;                           audW];
                case 2 % Video shuffled
                    shuffWarps = [mrW;                                    vidW(:,permIndexes(bootI,:));   audW];
                case 3 % Audio shuffled
                    shuffWarps = [mrW;                                    vidW;                           audW(:,permIndexes(bootI,:))];
                otherwise
                    error('Invalid reconstructInd (use 1, 2, or 3).');
            end

            [PCA,MorphMean,loadings] = doPCA(shuffWarps); %#ok<ASGLU>

            partial_data_s = shuffWarps;
            partial_data_s(elementBoundaries(reconstructInd)+1 : elementBoundaries(reconstructInd+1), :) = 0;

            partialMorphMean_s = mean(partial_data_s, 2);
            partial_centered_s = bsxfun(@minus, partial_data_s, partialMorphMean_s);
            partial_loading_s  = partial_centered_s' * PCA;

            allShuffledOrigLoad{bootI}  = loadings;
            allShuffledReconLoad{bootI} = partial_loading_s;
        end
    else
        disp('NOT using parallel processing...');
        for bootI = 1:nBoots
            switch reconstructInd
                case 1 % MR shuffled
                    shuffWarps = [mrW(:,permIndexes(bootI,:));           vidW;                           audW];
                case 2 % Video shuffled
                    shuffWarps = [mrW;                                    vidW(:,permIndexes(bootI,:));   audW];
                case 3 % Audio shuffled
                    shuffWarps = [mrW;                                    vidW;                           audW(:,permIndexes(bootI,:))];
            end

            [PCA,MorphMean,loadings] = doPCA(shuffWarps); %#ok<ASGLU>

            partial_data_s = shuffWarps;
            partial_data_s(elementBoundaries(reconstructInd)+1 : elementBoundaries(reconstructInd+1), :) = 0;

            partialMorphMean_s = mean(partial_data_s, 2);
            partial_centered_s = bsxfun(@minus, partial_data_s, partialMorphMean_s);
            partial_loading_s  = partial_centered_s' * PCA;

            allShuffledOrigLoad{bootI}  = loadings;
            allShuffledReconLoad{bootI} = partial_loading_s;
        end
    end
    toc

    results.allShuffledOrigLoad  = allShuffledOrigLoad;
    results.allShuffledReconLoad = allShuffledReconLoad;

    % ===============================
    % Statistics (faithful to original)
    % ===============================
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

    % ---- Display histograms (faithful to original) ----
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
end % end trimodalPcaAndShuffling_Balanced

%% doPCA (unchanged from the original template)
function [prinComp,MorphMean,loadings] = doPCA(data)
% Mean each row (across frames)
MorphMean = mean(data, 2);
% Subtract overall mean from each frame
data = bsxfun(@minus, data, MorphMean);
xxt        = data'*data;
[~,LSq,V]  = svd(xxt);
LInv       = 1./sqrt(diag(LSq));
prinComp   = data * V * diag(LInv);
loadings   = (data')*prinComp;
end


function A = infer_scores_from_observed(P, mu, X, obs_idx, lambda)
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
    M = (Pobs.'*Pobs) + lambda*eye(size(P,2));  % (q x q)
    A = M \ (Pobs.' * Xtobs);            % (q x T)
end
