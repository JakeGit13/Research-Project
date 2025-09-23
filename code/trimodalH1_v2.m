function results = trimodalH1_v2(data, audioFeatures, dataIdx, opts)
    % Inputs:
    %   data           : struct array with .mr_warp2D, .vid_warp2D 
    %   audioFeatures  : [F × T] matrix for this sentence 
    %   dataIdx        : index into 'data'
    
    %% Default options  =======================================================================================
    arguments
        data
        audioFeatures
        dataIdx 
        opts.reconstructId (1,1) double = 3     % MR = 1, VIDEO = 2, AUDIO = 3
        opts.nBoots (1,1) double = 1000
        opts.VERBOSE (1,1) logical = false 
        opts.genFigures (1,1) logical = false
        opts.ifNormalise (1,1) logical = false

        %% >>> NEW: diagnostics controls (no normalization applied) <<< NOT SURE IF THESE ARE NEEDED TBH 
        opts.targetAudioShare (1,1) double = 0.15   % e.g., 0.10–0.20 is a good range

    end

    reconstructId = opts.reconstructId;
    nBoots      = opts.nBoots;
    VERBOSE     = opts.VERBOSE;    
    genfigures  = opts.genFigures;
    normalise   = opts.ifNormalise;

    targetAudioShare = opts.targetAudioShare;
    

    % Reset random seed
    rng('default');
    usePar = true;  % set to false if parallel processing isn't required/working
            
    %% Select out data from this actor/sentence
    thisMRWarp  = data(dataIdx).mr_warp2D;   % [p_MR x T]
    thisVidWarp = data(dataIdx).vid_warp2D;  % [p_VID x T]
    thisAudio   = audioFeatures;             % [p_AUD(=5) x T]

    mrFrameCount    = size(thisMRWarp,  2); 
    vidFrameCount   = size(thisVidWarp, 2); 
    audioFrameCount = size(thisAudio,    2);

    if VERBOSE
        fprintf('MR:    %d features x %d frames\n', size(thisMRWarp,1), mrFrameCount);
        fprintf('Video: %d features x %d frames\n', size(thisVidWarp,1), vidFrameCount);
        fprintf('Audio: %d features x %d frames\n\n', size(thisAudio,1),   audioFrameCount);
    end

    % --- Ensure equal frame counts (conservative: crop to the min) ---
    T = min([mrFrameCount, vidFrameCount, audioFrameCount]);
    if any([mrFrameCount, vidFrameCount, audioFrameCount] ~= T)
        if VERBOSE
            warning('Frame count mismatch (MR=%d, VID=%d, AUD=%d). Cropping all to T=%d.', ...
                mrFrameCount, vidFrameCount, audioFrameCount, T);
        end
        thisMRWarp  = thisMRWarp(:,  1:T);
        thisVidWarp = thisVidWarp(:, 1:T);
        thisAudio   = thisAudio(:,   1:T);
    end


    %% BLOCK VARIANCE DIAGNOSTICS (pre-normalisation)
    % =========================
    % Per-feature variance across time, then sum per block
    mrVarVec  = var(thisMRWarp,  0, 2);  % [p_MR x 1]
    vidVarVec = var(thisVidWarp, 0, 2);  % [p_VID x 1]
    audVarVec = var(thisAudio,   0, 2);  % [p_AUD x 1]

    vMR  = nansum(mrVarVec);
    vVID = nansum(vidVarVec);
    vAUD = nansum(audVarVec);
    vTot = vMR + vVID + vAUD;

    shareMR  = vMR  / vTot;
    shareVID = vVID / vTot;
    shareAUD = vAUD / vTot;

    if VERBOSE
        fprintf('--- Block variance (pre-PCA, no normalisation) ---\n');
        fprintf('p_MR=%d,  p_VID=%d,  p_AUD=%d\n', size(thisMRWarp,1), size(thisVidWarp,1), size(thisAudio,1));
        fprintf('SumVar  MR:  %.3e  (share = %.2f%%)\n', vMR,  100*shareMR);
        fprintf('SumVar  VID: %.3e  (share = %.2f%%)\n', vVID, 100*shareVID);
        fprintf('SumVar  AUD: %.3e  (share = %.2f%%)\n', vAUD, 100*shareAUD);
    end

    % --- Recommend simple block weights for a target audio share ---
    % Target shares: audio fixed, MR and VID split remaining equally
    sAUD = max(min(opts.targetAudioShare, 0.99), 0.01);  % clamp to (1%, 99%) for safety
    sRem = 1 - sAUD;
    sMR  = 0.5 * sRem;
    sVID = 0.5 * sRem;

    % Avoid divide-by-zero if a block is (near) constant
    epsVar = 1e-12;
    wMR  = sqrt(sMR  / max(vMR,  epsVar));
    wVID = sqrt(sVID / max(vVID, epsVar));
    wAUD = sqrt(sAUD / max(vAUD, epsVar));

    % Predicted shares *after* applying these scalars (purely diagnostic)
    vMR_w  = (wMR^2)  * vMR;
    vVID_w = (wVID^2) * vVID;
    vAUD_w = (wAUD^2) * vAUD;
    vTot_w = vMR_w + vVID_w + vAUD_w;

    shareMR_pred  = vMR_w  / vTot_w;
    shareVID_pred = vVID_w / vTot_w;
    shareAUD_pred = vAUD_w / vTot_w;

    if VERBOSE
        fprintf('\n--- Target shares ---\n');
        fprintf('Target:  MR=%.1f%%, VID=%.1f%%, AUD=%.1f%%\n', 100*sMR, 100*sVID, 100*sAUD);
        fprintf('Weights: wMR=%.3g, wVID=%.3g, wAUD=%.3g (relative scalars per block)\n', wMR, wVID, wAUD);
        fprintf('Predicted shares after weighting (pre-centering): MR=%.1f%%, VID=%.1f%%, AUD=%.1f%%\n\n', ...
            100*shareMR_pred, 100*shareVID_pred, 100*shareAUD_pred);
    end

    % Save diagnostics
    results.blockDiagnostics = struct( ...
        'pMR', size(thisMRWarp,1), 'pVID', size(thisVidWarp,1), 'pAUD', size(thisAudio,1), ...
        'sumVar', [vMR, vVID, vAUD], ...
        'share',  [shareMR, shareVID, shareAUD], ...
        'targetShare', [sMR, sVID, sAUD], ...
        'weightsRecommended', [wMR, wVID, wAUD], ...
        'predictedShareAfterWeight', [shareMR_pred, shareVID_pred, shareAUD_pred], ...
        'T_used', T );



    %% Z-SCORING
    %% WITHIN-BLOCK NORMALIZATION (per-feature z-score across time)
    % =========================
    if normalise
        if VERBOSE, fprintf('Applying within-block per-feature z-scoring...\n'); end

        % Compute per-feature mean/std across time (dimension 2)
        % MR
        mr_mu  = mean(thisMRWarp,  2, 'omitnan');
        mr_sd  = std( thisMRWarp,  0, 2, 'omitnan');    % unbiased by default
        mr_sd( mr_sd < 1e-12 ) = 1e-12;                 % guard against zero-variance
        thisMRWarp  = (thisMRWarp  - mr_mu) ./ mr_sd;

        % Video
        vid_mu = mean(thisVidWarp, 2, 'omitnan');
        vid_sd = std( thisVidWarp, 0, 2, 'omitnan');
        vid_sd( vid_sd < 1e-12 ) = 1e-12;
        thisVidWarp = (thisVidWarp - vid_mu) ./ vid_sd;

        % Audio
        aud_mu = mean(thisAudio,   2, 'omitnan');
        aud_sd = std( thisAudio,   0, 2, 'omitnan');
        aud_sd( aud_sd < 1e-12 ) = 1e-12;
        thisAudio   = (thisAudio   - aud_mu) ./ aud_sd;

        % Report post-normalization block variances/shares (diagnostic)
        mrVarVec_z  = var(thisMRWarp,  0, 2);
        vidVarVec_z = var(thisVidWarp, 0, 2);
        audVarVec_z = var(thisAudio,   0, 2);

        vMR_z  = sum(mrVarVec_z);
        vVID_z = sum(vidVarVec_z);
        vAUD_z = sum(audVarVec_z);
        vTot_z = vMR_z + vVID_z + vAUD_z;

        shareMR_z  = vMR_z  / vTot_z;
        shareVID_z = vVID_z / vTot_z;
        shareAUD_z = vAUD_z / vTot_z;

        if VERBOSE
            fprintf('--- Block variance AFTER within-block z-scoring ---\n');
            fprintf('SumVar  MR_z:  %.3e  (share = %.2f%%)\n', vMR_z,  100*shareMR_z);
            fprintf('SumVar  VID_z: %.3e  (share = %.2f%%)\n', vVID_z, 100*shareVID_z);
            fprintf('SumVar  AUD_z: %.3e  (share = %.2f%%)\n\n', vAUD_z, 100*shareAUD_z);
        end

        % Save normalization params (useful if you later want to back-transform)
        results.normParams = struct( ...
            'mr_mu', mr_mu, 'mr_sd', mr_sd, ...
            'vid_mu', vid_mu, 'vid_sd', vid_sd, ...
            'aud_mu', aud_mu, 'aud_sd', aud_sd );
    else
        % If not normalising, clear to avoid confusion downstream
        results.normParams = [];
        vMR_z  = vMR;   vVID_z = vVID;   vAUD_z = vAUD;   % carry forward raw sums
        shareMR_z  = shareMR; shareVID_z = shareVID; shareAUD_z = shareAUD;
    end



    %% BLOCK WEIGHTING
    %% === Block weighting AFTER z-scoring (or raw if normalise=false) ===
    % Choose target shares: audio ~10–20% is a good starting window
    sAUD = max(min(opts.targetAudioShare, 0.99), 0.01);  % e.g., 0.15
    sRem = 1 - sAUD;
    sMR  = 0.5 * sRem;
    sVID = 0.5 * sRem;

    % Use the observed (post z-score) block sums you already computed:
    % vMR_z, vVID_z, vAUD_z  (if not z-scored, these are the raw sums vMR,vVID,vAUD)
    epsVar = 1e-12;
    wMR  = sqrt( sMR  / max(vMR_z,  epsVar) );
    wVID = sqrt( sVID / max(vVID_z, epsVar) );
    wAUD = sqrt( sAUD / max(vAUD_z, epsVar) );

    % Optional: print predicted shares (pre-centering) for sanity
    if VERBOSE
        vMR_w  = (wMR^2)  * vMR_z;
        vVID_w = (wVID^2) * vVID_z;
        vAUD_w = (wAUD^2) * vAUD_z;
        vTot_w = vMR_w + vVID_w + vAUD_w;
        fprintf('Predicted shares after weighting: MR=%.1f%%, VID=%.1f%%, AUD=%.1f%%\n', ...
            100*vMR_w/vTot_w, 100*vVID_w/vTot_w, 100*vAUD_w/vTot_w);
    end

    %% Apply weights 
    if opts.ifNormalise
        thisMRWarp  = thisMRWarp  * wMR;
        thisVidWarp = thisVidWarp * wVID;
        thisAudio   = thisAudio   * wAUD;
    

        %% --- Diagnostics: actual block variances AFTER weighting (pre-PCA) ---
        mrVarVec_w  = var(thisMRWarp,  0, 2);
        vidVarVec_w = var(thisVidWarp, 0, 2);
        audVarVec_w = var(thisAudio,   0, 2);
    
        vMR_w  = sum(mrVarVec_w);
        vVID_w = sum(vidVarVec_w);
        vAUD_w = sum(audVarVec_w);
        vTot_w = vMR_w + vVID_w + vAUD_w;
    
        shareMR_w  = vMR_w  / vTot_w;
        shareVID_w = vVID_w / vTot_w;
        shareAUD_w = vAUD_w / vTot_w;
    
        if VERBOSE
            fprintf('--- Block variance AFTER block weighting ---\n');
            fprintf('SumVar  MR_w:  %.3e  (share = %.2f%%)\n', vMR_w,  100*shareMR_w);
            fprintf('SumVar  VID_w: %.3e  (share = %.2f%%)\n', vVID_w, 100*shareVID_w);
            fprintf('SumVar  AUD_w: %.3e  (share = %.2f%%)\n\n', vAUD_w, 100*shareAUD_w);

        end

    end



    %% CONCATENATE MR, VIDEO AND AUDIO (weighted or raw depending on flag)
    mixWarps = [thisMRWarp; thisVidWarp; thisAudio];

    % Perform a PCA on the hybrid data
    [origPCA, origMorphMean, origloadings] = doPCA(mixWarps); 




    % Do the non-shuffled reconstruction for the original order **********************************************************************
    
    % Indexes of boundaries between MR, video and audio

    nMR = size(thisMRWarp,1); 
    nVID = size(thisVidWarp,1); 
    nAUD = size(thisAudio,1);
    
    elementBoundaries = [0, nMR, nMR + nVID, nMR + nVID + nAUD]; % Element boundaries based on the rows
    
    % --- Audio share in the PCs (unshuffled) ---
    nMR  = size(thisMRWarp,1);
    nVID = size(thisVidWarp,1);
    nAUD = size(thisAudio,1);
    b = [0, nMR, nMR+nVID, nMR+nVID+nAUD];
    audRows = (b(3)+1):b(4);
    
    pcAudioWeight = sum(origPCA(audRows,:).^2, 1);   % fraction per PC in audio rows
    fprintf('[PC|Audio] median=%.3e  max=%.3e\n', median(pcAudioWeight), max(pcAudioWeight));



    if VERBOSE
        fprintf('Element boundaries (row indices): %d %d %d %d\n', elementBoundaries);
        fprintf('MR rows:    1–%d\n', nMR);
        fprintf('Video rows: %d–%d\n', nMR+1, nMR+nVID);
        fprintf('Audio rows: %d–%d\n\n', nMR+nVID+1, nMR+nVID+nAUD);
    end

    partial_data = mixWarps;    % make copy of mixWarps
    partial_data(elementBoundaries(reconstructId)+1:elementBoundaries(reconstructId+1),:) = 0; % Set one modality to 0
    partialMorphMean = mean(partial_data, 2); % get row means 
    partial_centered = bsxfun(@minus, partial_data, partialMorphMean);
    partial_loading = partial_centered'*origPCA;

    
    % Store the loadings for further processing
    results.nonShuffledLoadings = origloadings;
    results.nonShuffledReconLoadings = partial_loading;


    %% Just for audio
    if reconstructId == 3
        % --- Reconstruct data and read out Audio rows (unshuffled) ---
        recon_full  = origPCA * (partial_loading') + origMorphMean;   % rows x T
        recon_audio = recon_full(audRows,:);                           % Audio-only
        orig_audio  = mixWarps(audRows,:);
    
        % centre per feature (rows) before comparison
        ra = recon_audio - mean(recon_audio, 2);
        oa = orig_audio  - mean(orig_audio,  2);
    
        R_audio_true   = corr(oa(:), ra(:), 'rows','complete');
        SSE_audio_true = sum((oa(:) - ra(:)).^2);
        fprintf('[Audio|data] Unshuffled: R=%.3f  SSE=%.3e\n', R_audio_true, SSE_audio_true);
    end

    
    %% Display ************************************************************************************************************************
    
    figure;
    
    % Original and reconstructed loadings
    plot(results.nonShuffledLoadings,results.nonShuffledReconLoadings,'.');
    
    % Unity line
    hline=refline(1,0);
    hline.Color = 'k';
    
    xlabel('Original loadings');ylabel('Reconstructed loadings');



    %% Do the shuffled reconstruction *************************************************************************************************
    

    if reconstructId == 3
        R_audio_shuff   = NaN(1, nBoots);
        SSE_audio_shuff = NaN(1, nBoots);
    end

    
    T = size(mixWarps, 2);

    % Create indexes for nBoot random permutations using a loop
    permIndexes = NaN(nBoots, T);
    for bootI = 1:nBoots
        permIndexes(bootI,:) = randperm(T);
    end

    
    allShuffledOrigLoad = cell(nBoots,1);
    allShuffledReconLoad = cell(nBoots,1);
    
    % Do PCA on one shuffled combo
    nCores = feature('numcores');
    tic
    if usePar && nCores>2
        disp('Using parallel processing...');
        
        poolOpen = gcp('nocreate');
        if isempty(poolOpen)
            pp = parpool(nCores-1); % Leave one core free
        end
        
        parfor bootI = 1:nBoots

            % p is the permutation of frame indices for this bootstrap
            p = permIndexes(bootI,:);
            
            % Build the shuffled hybrid by shuffling columns in the target block only
            if reconstructId == 1          % MR
                shuffWarps = [ thisMRWarp(:,p);  thisVidWarp;        thisAudio ];
            elseif reconstructId == 2      % VIDEO
                shuffWarps = [ thisMRWarp;        thisVidWarp(:,p);  thisAudio ];
            elseif reconstructId == 3      % AUDIO
                shuffWarps = [ thisMRWarp;        thisVidWarp;        thisAudio(:,p) ];
            else
                error('reconstructId must be 1 (MR), 2 (VIDEO), or 3 (AUDIO).');
            end
            
            % Refit PCA on the shuffled hybrid
            [PCA, MorphMean, loadings] = doPCA(shuffWarps);

            
            partial_data = shuffWarps;
            partial_data(elementBoundaries(reconstructId)+1:elementBoundaries(reconstructId+1),:) = 0; % surely we must change it? 
            partialMorphMean = mean(partial_data, 2);
            partial_centered = bsxfun(@minus, partial_data, partialMorphMean); % resizes partialMorphMean to make subtraction possible (could use matrix maths?)
            partial_loading = partial_centered'*PCA;
            
            allShuffledOrigLoad{bootI} = loadings;
            allShuffledReconLoad{bootI} = partial_loading;



            %% Just for audio (shuffled)
            if reconstructId == 3
                % --- Reconstruct shuffled data and read out Audio rows ---
                recon_full  = PCA * (partial_loading') + MorphMean;
                recon_audio = recon_full(audRows,:);
                orig_audio  = shuffWarps(audRows,:);    % compare within the shuffled pairing
            
                ra = recon_audio - mean(recon_audio, 2);
                oa = orig_audio  - mean(orig_audio,  2);
            
                R_audio_shuff(bootI)   = corr(oa(:), ra(:), 'rows','complete');
                SSE_audio_shuff(bootI) = sum((oa(:) - ra(:)).^2);
            end


        end
    end
    results.allShuffledOrigLoad = allShuffledOrigLoad;
    results.allShuffledReconLoad = allShuffledReconLoad;
    toc

    %% Just for audio 
    if reconstructId == 3
        muR = mean(R_audio_shuff); sdR = std(R_audio_shuff);
        p_R_audio = (1 + sum(R_audio_shuff >= R_audio_true)) / (nBoots + 1);
        fprintf('[Audio|data] Null: R mean=%.3f±%.3f | p=%.4g (one-sided)\n', muR, sdR, p_R_audio);
    end



    
    % Statistics ************************************************************************************************************************
    
    % Unshuffled
    loadings1D = results.nonShuffledLoadings(:);
    reconLoadings1D = results.nonShuffledReconLoadings(:);
    
    SSE = sum((loadings1D-reconLoadings1D).^2); % sum of squared error
    [R,~] = corr(loadings1D,reconLoadings1D);   % Pearson correlation
    p = polyfit(loadings1D,reconLoadings1D,1); % linear fit
    
    unshuffstats = [R p(1) SSE];
    
    % Shuffled
    shuffstats = NaN(3,nBoots);
    for bootI=1:nBoots
        loadings1D = results.allShuffledOrigLoad{bootI}(:);
        reconLoadings1D = results.allShuffledReconLoad{bootI}(:);
        
        SSE = sum((loadings1D-reconLoadings1D).^2); % sum of squared error
        [R,~] = corr(loadings1D,reconLoadings1D);   % Pearson correlation
        p = polyfit(loadings1D,reconLoadings1D,1); % linear fit
        
        shuffstats(:,bootI) = [R p(1) SSE];
    end



    %% NOVEL STATS  
    % --- Descriptive labels ---
    modNames = {'MR','Video','Audio'};
    tgt = reconstructId;                          % 1=MR, 2=Video, 3=Audio
    obs = setdiff(1:3, tgt);                      % observed modalities
    obsLabel = strjoin(modNames(obs), '+');
    
    % --- Unshuffled metrics (true pairing) ---
    R_true     = unshuffstats(1);
    slope_true = unshuffstats(2);
    SSE_true   = unshuffstats(3);
    
    % --- Shuffled (null) summaries ---
    R_shuff     = shuffstats(1,:);
    slope_shuff = shuffstats(2,:);
    SSE_shuff   = shuffstats(3,:);
    
    muR   = mean(R_shuff);   sdR   = std(R_shuff);
    p95R  = prctile(R_shuff,95);   p99R = prctile(R_shuff,99);
    
    muG   = mean(slope_shuff); sdG  = std(slope_shuff);
    p95G  = prctile(slope_shuff,95); p99G = prctile(slope_shuff,99);
    
    muE   = mean(SSE_shuff);  sdE  = std(SSE_shuff);
    p05E  = prctile(SSE_shuff,5);   p01E = prctile(SSE_shuff,1);   % lower is better
    
    % --- Percentile of the true value within the null ---
    pct_R     = 100 * mean(R_shuff <  R_true);
    pct_slope = 100 * mean(slope_shuff < slope_true);
    pct_SSE   = 100 * mean(SSE_shuff  > SSE_true);   % SSE lower is better
    
    % --- Effect sizes (optional) ---
    z_R     = (R_true     - muR) / max(sdR,eps);
    z_slope = (slope_true - muG) / max(sdG,eps);
    z_SSE   = (muE - SSE_true) / max(sdE,eps);       % inverted so larger=better
    
    % --- Header ---
    fprintf('\n=== Trimodal PCA (H1): Reconstruct %s from %s ===\n', modNames{tgt}, obsLabel);
    
    % --- True pairing summary ---
    fprintf('Unshuffled:  R = %.3f,  gain (slope) = %.3f,  SSE = %.3e\n', R_true, slope_true, SSE_true);
    
    % --- Null summaries ---
    fprintf('Null (shuffle %s):  R  mean = %.3f ± %.3f, 95th = %.3f, 99th = %.3f\n', ...
            modNames{tgt}, muR, sdR, p95R, p99R);
    fprintf('                     gain mean = %.3f ± %.3f, 95th = %.3f, 99th = %.3f\n', ...
            muG, sdG, p95G, p99G);
    fprintf('                     SSE mean  = %.3e ± %.3e,  5th = %.3e,  1st = %.3e\n', ...
            muE, sdE, p05E, p01E);
    
    % --- Simple interpretation cue ---
    if R_true > p95R
        fprintf('Result: R exceeds the 95th percentile of the null ⇒ evidence supporting H1.\n');
    else
        fprintf('Result: R does not exceed the 95th percentile of the null.\n');
    end

    
    % Display ************************************************************************************************************************
    
    figure;
    
    % Original and reconstructed loadings
    subplot(2,3,2);
    plot(results.nonShuffledLoadings,results.nonShuffledReconLoadings,'.');
    
    % Unity line
    hline=refline(1,0);
    hline.Color = 'k';
    
    xlabel('Original loadings');ylabel('Reconstructed loadings');
    
    statStrings = {'Correlation coefficient','Linear Fit Gradient','SSE'};
    
    for statI = 1:3
        
        % Shuffled distributions
        subplot(2,3,statI+3);
        
        histogram(shuffstats(statI,:),50);hold on
        axis tight
        plot(unshuffstats([statI statI]),ylim,'r--','linewidth',2);
        
        xlabel(statStrings{statI});ylabel('Frequency');
        
    end

end % end pcaAndShufflingExample

%% doPCA
function [prinComp,MorphMean,loadings] = doPCA(data)

    % Mean each row (across frames)
    MorphMean = mean(data, 2);
    
    % Subtract overall mean from each frame
    data = bsxfun(@minus, data, MorphMean);
    xxt        = data'*data;
    [~,LSq,V]  = svd(xxt);
    LInv       = 1./sqrt(diag(LSq));
    prinComp  = data * V * diag(LInv);
    loadings = (data')*prinComp;
end % end doPCA