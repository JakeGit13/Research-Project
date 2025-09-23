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
        opts.reconstructId = 3;     % MR = 1, VIDEO = 2, AUDIO = 3
        opts.nBoots = 1000
        opts.VERBOSE = false 
        opts.genFigures = false;
        opts.normalise = false;
    end

    reconstructId = opts.reconstructId;
    nBoots      = opts.nBoots;
    
    VERBOSE     = opts.VERBOSE;    
    genfigures = opts.genFigures;
    normalise = opts.normalise;


    % Reset random seed
    rng('default');
    usePar = true; % set to false if parallel processing isn't required/working
            
    
    %% PCA on hybrid facial video and vocal-tract MR images
   
    
    % Select out data from this actor/sentence
    thisMRWarp = data(dataIdx).mr_warp2D;
    thisVidWarp = data(dataIdx).vid_warp2D;
    thisAudio = audioFeatures;

    mrFrameCount = size(thisMRWarp, 2); 
    vidFrameCount = size(thisVidWarp, 2); 
    audioFrameCount = size(thisAudio, 2);

    if VERBOSE
        fprintf('MR:    %d features x %d frames\n', size(thisMRWarp,1), mrFrameCount);
        fprintf('Video: %d features x %d frames\n', size(thisVidWarp,1), vidFrameCount);
        fprintf('Audio: %d features x %d frames\n\n', size(thisAudio,1),   audioFrameCount);
    end



    %% CONCATENATE MR, VIDEO AND AUDIO
    mixWarps = [thisMRWarp; thisVidWarp; thisAudio];
    
    % Perform a PCA on the hybrid data
    [origPCA,origMorphMean,origloadings] = doPCA(mixWarps);
    
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
    [R,~] = corr(loadings1D, reconLoadings1D, 'rows','complete');   % Pearson correlation
    p = polyfit(loadings1D,reconLoadings1D,1); % linear fit
    
    unshuffstats = [R p(1) SSE];
    
    % Shuffled
    shuffstats = NaN(3,nBoots);
    for bootI=1:nBoots
        loadings1D = results.allShuffledOrigLoad{bootI}(:);
        reconLoadings1D = results.allShuffledReconLoad{bootI}(:);
        
        SSE = sum((loadings1D-reconLoadings1D).^2); % sum of squared error
        [R,~] = corr(loadings1D, reconLoadings1D, 'rows','complete');   % Pearson correlation
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