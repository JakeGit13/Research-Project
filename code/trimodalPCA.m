function resultsForCSV = trimodalPCA(data, audioFeatures, dataIdx, opts)
    % Inputs:
    %   data           : struct array with .mr_warp2D, .vid_warp2D 
    %   audioFeatures  : [F × T] matrix for this sentence 
    %   dataIdx        : index into 'data'
    
    %% Options set to H1 as default =======================================================================================
    arguments
        data
        audioFeatures
        dataIdx 
        opts.reconstructId (1,1) double = 3     % MR = 1, VIDEO = 2, AUDIO = 3
        opts.nBoots (1,1) double = 1000
        opts.VERBOSE (1,1) logical = false 
        opts.genFigures (1,1) logical = false
        opts.ifNormalise (1,1) logical = true;
        opts.includeAudio (1,1) logical = true;
        opts.h1Source (1,1) string = "MRVID"   % "MRVID" | "MR" | "VID"

        opts.targetAudioShare (1,1) double = 0.15   % e.g., 0.10–0.20 is a good range

    end

    reconstructId = opts.reconstructId;
    nBoots      = opts.nBoots;
    VERBOSE     = opts.VERBOSE;    
    genfigures  = opts.genFigures;
    ifNormalise   = opts.ifNormalise;
    includeAudio   = opts.includeAudio;

    targetAudioShare = opts.targetAudioShare;
    h1Source = opts.h1Source;

    

    

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


    %{

    %% BLOCK VARIANCE DIAGNOSTICS (pre-normalisation)
    % =========================
    % Per-feature variance across time, then sum per block
    mrVarVec  = var(thisMRWarp,  0, 2);  % [p_MR x 1]
    vidVarVec = var(thisVidWarp, 0, 2);  % [p_VID x 1]
    audVarVec = var(thisAudio,   0, 2);  % [p_AUD x 1]

    vMR  = sum(mrVarVec);
    vVID = sum(vidVarVec);
    vAUD = sum(audVarVec);
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


    %}


    %% Z-SCORING
    %% WITHIN-BLOCK NORMALIZATION (per-feature z-score across time)
    % =========================
    if ifNormalise
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



    %% CONCATENATE MR, VIDEO AND AUDIO (depending on H1 or H2)
    
    if includeAudio 
        mixWarps = [thisMRWarp; thisVidWarp; thisAudio]; 
    else 
        mixWarps = [thisMRWarp; thisVidWarp]; 
    end

    %% Not sure what this does now 
    pMR  = size(thisMRWarp,  1);
    pVID = size(thisVidWarp, 1);
    pAUD = opts.includeAudio * size(thisAudio, 1);  % 0 if no audio in fit
    
    idxMR  = 1:pMR;
    idxVID = pMR + (1:pVID);
    idxAUD = pMR + pVID + (1:pAUD);  % empty if pAUD==0
    
    % Projection input: start all zeros, then fill only the allowed block(s)
    XprojInput = zeros(size(mixWarps), 'like', mixWarps);
    
    switch reconstructId
        case 1  % target MR: feed Video-only
            XprojInput(idxVID, :) = thisVidWarp;   % MR and (if present) Audio stay zero

        case 2  % target Video: feed MR-only
            XprojInput(idxMR, :)  = thisMRWarp;    % Video and (if present) Audio stay zero

        case 3  % H1 target Audio: feed MR+Video-only
            switch upper(h1Source)
                case 'MRVID'   % default: MR+VID
                    fprintf("RECONSTRUCTING FROM %s\n", h1Source)
                    XprojInput(idxMR,:)  = thisMRWarp;
                    XprojInput(idxVID,:) = thisVidWarp;
                case 'MR'      % MR-only
                    fprintf("RECONSTRUCTING FROM %s\n", h1Source)
                    XprojInput(idxMR,:)  = thisMRWarp;
                    % VID remains zero
                case 'VID'     % VID-only
                    fprintf("RECONSTRUCTING FROM %s\n", h1Source)
                    XprojInput(idxVID,:) = thisVidWarp;
                    % MR remains zero
                otherwise
                            error('opts.h1Source must be "MRVID", "MR", or "VID".');
            end

        otherwise
            error('reconstructId must be 1 (MR), 2 (Video), or 3 (Audio).');
    end


    % Perform a PCA on the hybrid data
    [origPCA, origMorphMean, origloadings] = doPCA(mixWarps); 


    % Do the non-shuffled reconstruction for the original order **********************************************************************
    
    % Indexes of boundaries between MR, video and audio

    % Boundaries must match the FIT MATRIX (mixWarps), not raw blocks
    nMR  = size(thisMRWarp, 1);
    nVID = size(thisVidWarp, 1);
    nAUD = includeAudio * size(thisAudio,1);   % 0 if audio was excluded at fit
    
    elementBoundaries = [0, nMR, nMR + nVID, nMR + nVID + nAUD];   % Element boundaries based on the rows

    
    % Audio rows only exist if audio was included in the fit
    b = elementBoundaries;


    if includeAudio && (b(4) > b(3))
        audRows = (b(3)+1):b(4);
    else
        audRows = [];
    end





    if VERBOSE
        fprintf('Element boundaries (row indices): %d %d %d %d\n', elementBoundaries);
        fprintf('MR rows:    1–%d\n', nMR);
        fprintf('Video rows: %d–%d\n', nMR+1, nMR+nVID);
        fprintf('Audio rows: %d–%d\n\n', nMR+nVID+1, nMR+nVID+nAUD);
    end

    % Project FROM the source-only input you just built (XprojInput)
    Xctr_source     = XprojInput - origMorphMean;
    partial_loading = Xctr_source' * origPCA;   % scores (T×K)
    

%{

    % Store the loadings for further processing
    results.nonShuffledLoadings      = origloadings;      % scores from the fit data (T×K)
    results.nonShuffledReconLoadings = partial_loading;   % scores from source-only input (T×K)
    
%}



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

    %% Primary H2 metric: native target space (MR/Video)
    if reconstructId ~= 3
        % Reconstruct the full data from the source-only scores
        recon_full = origPCA * (partial_loading') + origMorphMean;   % rows x T
    
        % Target rows (already defined earlier as idxMR / idxVID)
        if reconstructId == 1
            tgtRows = idxMR;   tgtName = 'MR';
        else % reconstructId == 2
            tgtRows = idxVID;  tgtName = 'Video';
        end
    
        recon_tgt = recon_full(tgtRows,:);
        orig_tgt  = mixWarps(tgtRows,:);
    
        % Centre per feature (rows) before comparison
        rt = recon_tgt - mean(recon_tgt, 2);
        ot = orig_tgt  - mean(orig_tgt,  2);
    
        R_native   = corr(ot(:), rt(:), 'rows','complete');
        SSE_native = sum((ot(:) - rt(:)).^2);
        muY        = mean(ot(:));
        SST        = sum((ot(:) - muY).^2);
        R2_native  = 1 - SSE_native / max(SST, eps);
    
        % Store & print
        results.native_R   = R_native;
        results.native_SSE = SSE_native;
        results.native_R2  = R2_native;
    
        fprintf('Primary (%s space): SSE=%.3e, R^2=%.3f\n', tgtName, SSE_native, R2_native);
    end





    
    %% Display ************************************************************************************************************************
    if genfigures
        figure;
        
        % Original and reconstructed loadings
        plot(results.nonShuffledLoadings,results.nonShuffledReconLoadings,'.');
        
        % Unity line
        hline=refline(1,0);
        hline.Color = 'k';
        
        xlabel('Original loadings');ylabel('Reconstructed loadings');
    end 


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

    
    %{

    allShuffledOrigLoad = cell(nBoots,1);
    allShuffledReconLoad = cell(nBoots,1);

    %}
    
    % Do PCA on one shuffled combo
    nCores = feature('numcores');
    tic

    % Only build a null for H1 (Audio target). H2 runs should set nBoots=0 upstream.
    if reconstructId == 3 && usePar && nCores>2
        disp('Using parallel processing...');
        
        poolOpen = gcp('nocreate');
        if isempty(poolOpen)
            parpool; 
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
            
            % Build the matching SOURCE-ONLY projection input for THIS shuffled fit
            XprojInputB = zeros(size(shuffWarps), 'like', shuffWarps);
            switch reconstructId
                case 1   % target MR: feed VID-only
                    XprojInputB( (elementBoundaries(1)+1) : elementBoundaries(2), : ) = 0; % MR rows zero (redundant)
                    XprojInputB( (elementBoundaries(2)+1) : elementBoundaries(3), : ) = thisVidWarp;
                case 2   % target VID: feed MR-only
                    XprojInputB( (elementBoundaries(1)+1) : elementBoundaries(2), : ) = thisMRWarp;
                    XprojInputB( (elementBoundaries(2)+1) : elementBoundaries(3), : ) = 0; % VID rows zero
                case 3   % target AUD: feed MR+VID / MR-only / VID-only based on h1Source
                    switch upper(h1Source)
                        case 'MRVID'
                            XprojInputB( (elementBoundaries(1)+1) : elementBoundaries(2), : ) = thisMRWarp;
                            XprojInputB( (elementBoundaries(2)+1) : elementBoundaries(3), : ) = thisVidWarp;
                        case 'MR'
                            XprojInputB( (elementBoundaries(1)+1) : elementBoundaries(2), : ) = thisMRWarp;
                        case 'VID'
                            XprojInputB( (elementBoundaries(2)+1) : elementBoundaries(3), : ) = thisVidWarp;
                    end
            end
            % Always zero audio rows at projection; H1/H2 never feed audio as input
            if includeAudio && elementBoundaries(4) > elementBoundaries(3)
                XprojInputB( (elementBoundaries(3)+1) : elementBoundaries(4), : ) = 0;
            end
            
            % Project FROM the source-only input in the shuffled fit
            XctrB          = XprojInputB - MorphMean;
            partial_loading = XctrB' * PCA;
            

            %{
            allShuffledOrigLoad{bootI}  = loadings;
            allShuffledReconLoad{bootI} = partial_loading;

            %}


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
    %{
    results.allShuffledOrigLoad = allShuffledOrigLoad;
    results.allShuffledReconLoad = allShuffledReconLoad;
    %}

    toc

    %% Just for audio 
    if reconstructId == 3
        muR = mean(R_audio_shuff); sdR = std(R_audio_shuff);
        p_R_audio = (1 + sum(R_audio_shuff >= R_audio_true)) / (nBoots + 1);
        fprintf('[Audio|data] Null: R mean=%.3f±%.3f | p=%.4g (one-sided)\n', muR, sdR, p_R_audio);
    end



    
    % Statistics ************************************************************************************************************************
    

    %{
    % Unshuffled
    loadings1D = results.nonShuffledLoadings(:);
    reconLoadings1D = results.nonShuffledReconLoadings(:);
    
    SSE = sum((loadings1D-reconLoadings1D).^2); % sum of squared error
    [R,~] = corr(loadings1D,reconLoadings1D);   % Pearson correlation
    p = polyfit(loadings1D,reconLoadings1D,1); % linear fit
    
    unshuffstats = [R p(1) SSE];
    
    % Shuffled (null) – only for H1
    if reconstructId == 3 && nBoots > 0 && ~isempty(results.allShuffledOrigLoad)
        nBootsEff = numel(results.allShuffledOrigLoad);
        shuffstats = NaN(3, nBootsEff);
        for bootI = 1:nBootsEff
            loadings1D      = results.allShuffledOrigLoad{bootI}(:);
            reconLoadings1D = results.allShuffledReconLoad{bootI}(:);
            if isempty(loadings1D) || isempty(reconLoadings1D)
                shuffstats(:,bootI) = [NaN NaN NaN];
            else
                SSE = sum((loadings1D - reconLoadings1D).^2);
                R   = corr(loadings1D, reconLoadings1D, 'rows','complete');
                p   = polyfit(loadings1D, reconLoadings1D, 1);
                shuffstats(:,bootI) = [R p(1) SSE];
            end
        end
    else
        shuffstats = [];
    end

    %}



    %{

    %% NOVEL STATS  
    % --- Descriptive labels ---
    modNames = {'MR','Video','Audio'};
    tgt = reconstructId;  % 1=MR, 2=Video, 3=Audio
    
    % Human-readable source label
    switch tgt
        case 1, obsLabel = 'Video';           % Video->MR (H2)
        case 2, obsLabel = 'MR';              % MR->Video (H2)
        case 3                                 % Audio target (H1)
            switch upper(h1Source)
                case 'MR',   obsLabel = 'MR';
                case 'VID',  obsLabel = 'Video';
                otherwise,   obsLabel = 'MR+Video';
            end
    end

    
    % --- Unshuffled metrics (true pairing) ---
    R_true     = unshuffstats(1);
    slope_true = unshuffstats(2);
    SSE_true   = unshuffstats(3);
    
    % --- Header ---
    fprintf('\n=== Trimodal PCA: Reconstruct %s from %s ===\n', modNames{tgt}, obsLabel);
    
    % --- True pairing summary ---
    fprintf('Unshuffled:  R = %.3f,  gain (slope) = %.3f,  SSE = %.3e\n', R_true, slope_true, SSE_true);
    
    % --- Shuffled (null) summaries — H1 only ---
    if reconstructId == 3 && ~isempty(shuffstats)

        
        R_shuff     = shuffstats(1,:);
        slope_shuff = shuffstats(2,:);
        SSE_shuff   = shuffstats(3,:);
    
        muR   = mean(R_shuff);   sdR   = std(R_shuff);
        p95R  = prctile(R_shuff,95);   p99R = prctile(R_shuff,99);
    
        muG   = mean(slope_shuff); sdG  = std(slope_shuff);
        p95G  = prctile(slope_shuff,95); p99G = prctile(slope_shuff,99);
    
        muE   = mean(SSE_shuff);  sdE   = std(SSE_shuff);
        p05E  = prctile(SSE_shuff,5);   p01E = prctile(SSE_shuff,1);
    
        % --- Percentiles of true vs null
        pct_R     = 100 * mean(R_shuff     <  R_true);
        pct_slope = 100 * mean(slope_shuff <  slope_true);
        pct_SSE   = 100 * mean(SSE_shuff   >  SSE_true);   % lower is better
    
        % --- Effect sizes (optional)
        z_R     = (R_true     - muR) / max(sdR,eps);
        z_slope = (slope_true - muG) / max(sdG,eps);
        z_SSE   = (muE - SSE_true) / max(sdE,eps);
    
        % --- Null summaries
        fprintf('Null (shuffle %s):  R  mean = %.3f ± %.3f, 95th = %.3f, 99th = %.3f\n', ...
                modNames{tgt}, muR, sdR, p95R, p99R);
        fprintf('                     gain mean = %.3f ± %.3f, 95th = %.3f, 99th = %.3f\n', ...
                muG, sdG, p95G, p99G);
        fprintf('                     SSE mean  = %.3e ± %.3e,  5th = %.3e,  1st = %.3e\n', ...
                muE, sdE, p05E, p01E);
    
        if R_true > p95R
            fprintf('Result (loadings): R exceeds the 95th percentile of the null.\n');
        else
            fprintf('Result (loadings): R does not exceed the 95th percentile of the null.\n');
        end
    end
    

    %}
        






    
    % --- Primary H1 cue: audio native-space SSE / R^2 ---
    if reconstructId == 3
        % Empirical one-sided p for SSE (better = smaller)
        p_SSE = (1 + sum(SSE_audio_shuff <= SSE_audio_true)) / (nBoots + 1);
        % R^2 for the audio block (optional and quick)
        muY  = mean(orig_audio(:));
        SST  = sum((orig_audio(:) - muY).^2);
        R2_audio_true = 1 - SSE_audio_true / max(SST,eps);
    
        fprintf('Primary (audio space): SSE=%.3e, R^2=%.3f, p_SSE=%.4g (vs lag-shuffle audio)\n', ...
                SSE_audio_true, R2_audio_true, p_SSE);
    end


    %{
    % Display ************************************************************************************************************************
    if genfigures
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
            
            % Shuffled distributions (H1 only)
            subplot(2,3,statI+3);
            if reconstructId == 3 && ~isempty(shuffstats)
                histogram(shuffstats(statI,:),50); hold on
                axis tight
                plot(unshuffstats([statI statI]), ylim, 'r--', 'linewidth', 2);
                xlabel(statStrings{statI}); ylabel('Frequency');
            else
                axis off
            end


            
        end
    end

    %}


    %% STORE RESULTS ===============================================================
    %% === STORE RESULTS (minimal CSV) =====================================
    
    % --- identifiers / config ---
    resultsForCSV.data_idx           = dataIdx;
    resultsForCSV.reconstruct_id     = reconstructId;               % 1=MR, 2=Video, 3=Audio
    resultsForCSV.source_label       = h1Source;              % 'MR','Video','MR+Video' (for H1)
    resultsForCSV.include_audio      = double(includeAudio);        % 0/1
    resultsForCSV.targetAudioShare   = targetAudioShare;
    

   
    
    % --- a single loadings-space diagnostic (legacy comparability) ---
    % resultsForCSV.R_load = unshuffstats(1);    % correlation in loadings space
    
    % --- primary native-space metrics ---
    if reconstructId == 3
        % H1: Audio target
        resultsForCSV.audio_R_native   = R_audio_true;
        resultsForCSV.audio_SSE_native = SSE_audio_true;
        muY  = mean(orig_audio(:));
        SST  = sum((orig_audio(:) - muY).^2);
        resultsForCSV.audio_R2_native  = 1 - SSE_audio_true/max(SST,eps);
    
        % null-only if we actually built it
        if nBoots > 0
            resultsForCSV.p_SSE_audio = p_SSE;   % one-sided (SSE smaller is better)
        else
            resultsForCSV.p_SSE_audio = NaN;
        end
    else
        % H2: MR/Video target
        resultsForCSV.native_R   = R_native;
        resultsForCSV.native_SSE = SSE_native;
        resultsForCSV.native_R2  = R2_native;
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