function trimodalFinalAttempt

clc;

dataDir = '/Users/jaker/Research-Project/data';
dataFile = 'mrAndVideoData.mat';
audioFile = 'audioFeaturesData_articulatory.mat';


usePar = true; % remove this as no option for series processing, not sure if that's a bad thing
VERBOSE = true;


reconstructId = 3; % 1 = MR, 2 = video, 3 = audio
shuffleTarget = 3;   % 1 = MR, 2 = video, 3 = audio
blockNames = {'MR','Video','Audio'};


nBoots = 200; % # bootstraps
% ******************************************************************************************************************************************************


% Reset random seed
rng('default');

dataFile = 'mrAndVideoData.mat';

addpath(dataDir) % Add the user-defined data directory to the path 
load(dataFile,'data');          % MR + video
load(audioFile,'audioData');   % Audio


actors = [data.actor]; % Array of actor numbers
sentences = [data.sentence]; % Array of sentence numbers

%% PCA on hybrid facial video and vocal-tract MR images

for ii = 9%:length(actors)
    
    clear results;
    
    %% === Load/Assemble Blocks ===
    thisMRWarp  = data(ii).mr_warp2D;                         % (p_mr x T)
    thisVidWarp = data(ii).vid_warp2D;                        % (p_vid x T)
    thisAudio   = audioData(ii).audioFeatures_articulatory';  % (p_aud x T)
    
    T = size(thisMRWarp, 2);    % Amount of frames

    if VERBOSE
        fprintf('Item %d\n', ii);
        fprintf('MR:    %d features x %d frames\n', size(thisMRWarp,1), T);
        fprintf('Video: %d features x %d frames\n', size(thisVidWarp,1), T);
        fprintf('Audio: %d features x %d frames\n', size(thisAudio,1),   T);
    end



    %% === Trimodal block balancing (MFA-style) ===============================
    % Diagnostics BEFORE weighting
    [lam1_mr,  fro_mr]   = block_scale_stats(thisMRWarp);
    [lam1_vid, fro_vid]  = block_scale_stats(thisVidWarp);
    [lam1_aud, fro_aud]  = block_scale_stats(thisAudio);
    fprintf('Before weighting:  PC1 λ  MR=%.4g | Video=%.4g | Audio=%.4g   | Fro MR=%.3g | Fro Video=%.3g | Fro Audio=%.3g\n', ...
            lam1_mr, lam1_vid, lam1_aud, fro_mr, fro_vid, fro_aud);

    % Weights to equalise each block's dominant variance scale
    w_mr  = 1/sqrt(lam1_mr);
    w_vid = 1/sqrt(lam1_vid);
    w_aud = 1/sqrt(lam1_aud);

    % Apply weights
    thisMRWarpW   =  w_mr * thisMRWarp;
    thisVidWarpW  =  w_vid * thisVidWarp;
    thisAudioW    =  w_aud * thisAudio;

    % Diagnostics AFTER weighting
    [lam1_mr_a,  fro_mr_a]   = block_scale_stats(thisMRWarpW);
    [lam1_vid_a, fro_vid_a]  = block_scale_stats(thisVidWarpW);
    [lam1_aud_a, fro_aud_a]  = block_scale_stats(thisAudioW);
    fprintf('After  weighting:  PC1 λ  MR=%.4g | Video=%.4g | Audio=%.4g   | Fro MR=%.3g | Fro Video=%.3g | Fro Audio=%.3g\n\n', ...
            lam1_mr_a, lam1_vid_a, lam1_aud_a, fro_mr_a, fro_vid_a, fro_aud_a);

    % Use weighted blocks for PCA and reconstruction
    mixWarps = [thisMRWarpW; thisVidWarpW; thisAudioW];



%% ============================================================================

    
    % Perform a PCA on the hybrid data
    [origPCA,origMorphMean,origloadings] = doPCA(mixWarps);
    
    % Do the non-shuffled reconstruction for the original order **********************************************************************
    
    % Indexes of boundaries between MR and video
    bMR  = size(thisMRWarpW, 1);
    bVID = size(thisVidWarpW, 1);
    bAUD = size(thisAudioW,  1);

    elementBoundaries = [0, bMR, bMR + bVID, bMR + bVID + bAUD];
    nFrames = T;  % already defined above


    if VERBOSE
        fprintf('Hidden/target block: %s | Shuffle target: %s\n', ...
            blockNames{reconstructId}, blockNames{shuffleTarget});
    end


    partial_data = mixWarps;
    partial_data(elementBoundaries(reconstructId)+1:elementBoundaries(reconstructId+1),:) = 0; % Set one modality to 0
    partialMorphMean = mean(partial_data, 2);
    partial_centered = bsxfun(@minus, partial_data, partialMorphMean);
    partial_loading = partial_centered'*origPCA;

    %% === H1 Audio feature-space metrics (UNSHUFFLED) =======================
    idx1 = elementBoundaries(reconstructId)+1;
    idx2 = elementBoundaries(reconstructId+1);
    
    % Audio rows of PCA basis (k columns), and centred Audio (match projection centring)
    P_audio       = origPCA(idx1:idx2, :);                                      % p_aud x k
    X_audio_true  = bsxfun(@minus, mixWarps(idx1:idx2, :), partialMorphMean(idx1:idx2));
    X_audio_hat   = P_audio * (partial_loading.');                               % p_aud x T
    
    % Vectorised correlation (primary H1 metric)
    a = zscore(X_audio_true(:));
    b = zscore(X_audio_hat(:));
    vecR_real = corr(a, b);
    
    % Row-wise correlations (omit constant rows)
    rowR = nan(size(X_audio_true,1),1);
    for rr = 1:size(X_audio_true,1)
        aa = X_audio_true(rr,:).';  bb = X_audio_hat(rr,:).';
        if std(aa)>0 && std(bb)>0
            rowR(rr) = corr(aa,bb);
        end
    end
    medRowR_real = median(rowR,'omitnan');
    
    % SSE in audio block
    SSE_real = sum((X_audio_true(:) - X_audio_hat(:)).^2);

    SSE0_real = sum(X_audio_true(:).^2);
    VAF_real  = 1 - SSE_real / SSE0_real;   % in the weighted/centred space
    
    results.h1_VAF_real = VAF_real;
    
    if VERBOSE
        fprintf('H1 (Audio) UNSHUFFLED: VAF=%.1f%% (space=weighted/centred)\n', 100*VAF_real);
    end

    
    % Store + print
    results.h1_vecR_real    = vecR_real;
    results.h1_medRowR_real = medRowR_real;
    results.h1_SSE_real     = SSE_real;
    
    if VERBOSE
        fprintf('H1 (Audio) UNSHUFFLED: vecR=%.4f | median rowR=%.4f | SSE=%.3e\n', ...
                vecR_real, medRowR_real, SSE_real);
    end


    %% === H1 feature-space shuffle-null (EVALUATION-ONLY; NO REFIT) ==========
    % Keep origPCA and partial_loading fixed; only permute the TRUE audio.
    if VERBOSE
        fprintf('H1 eval-only null: model fixed; permuting audio frames at evaluation only.\n');
    end
    
    % Build permutations over the same nFrames
    permIdx_eval = NaN(nBoots, nFrames);
    for b = 1:nBoots
        permIdx_eval(b,:) = randperm(nFrames);
    end
    
    % Compute vectorised r for each evaluation-only shuffle
    shuffAudioVecR_eval = nan(1, nBoots);
    shuffAudioVAF_eval  = nan(1, nBoots); 
    
    % Use parallel if available (cheap either way)
    nCores = feature('numcores');
    if usePar && nCores > 2
        parfor b = 1:nBoots
            Xa_true_sh = X_audio_true(:, permIdx_eval(b,:));  % permuted true audio
            aa = zscore(Xa_true_sh(:));
            bb = zscore(X_audio_hat(:));                      % fixed reconstruction from MR+Video
            shuffAudioVecR_eval(b) = corr(aa, bb);

            % after you build Xa_true_sh
            SSE_sh = sum((Xa_true_sh(:) - X_audio_hat(:)).^2);
            VAF_sh = 1 - SSE_sh / SSE0_real;
            shuffAudioVAF_eval(b) = VAF_sh;

        end
    end
    
    % Summarise this H1-appropriate null
    realVecR_eval = vecR_real;
    sh_med_eval   = median(shuffAudioVecR_eval, 'omitnan');
    sh_ci_eval    = prctile(shuffAudioVecR_eval, [2.5 97.5]);
    p_vecR_eval   = mean(shuffAudioVecR_eval >= realVecR_eval);  % one-sided (>=)
    
    VAF_med_eval = median(shuffAudioVAF_eval,'omitnan');
    VAF_ci_eval  = prctile(shuffAudioVAF_eval,[2.5 97.5]);
    p_VAF_eval   = mean(shuffAudioVAF_eval >= VAF_real);   % one-sided (>=)
    
    results.h1_eval_VAF_real   = VAF_real;
    results.h1_eval_VAF_shuffs = shuffAudioVAF_eval;
    results.h1_eval_VAF_p      = p_VAF_eval;
    results.h1_eval_VAF_ci     = VAF_ci_eval;
    
    if VERBOSE
        fprintf(['H1 (Audio) VAF — EVAL-ONLY: real=%.1f%% | shuffle median=%.1f%% | ' ...
                 '95%% CI=[%.1f%%, %.1f%%] | p=%.3g\n'], ...
                100*VAF_real, 100*VAF_med_eval, 100*VAF_ci_eval(1), 100*VAF_ci_eval(2), p_VAF_eval);
    end






    
    % Store the loadings for further processing
    results.nonShuffledLoadings = origloadings;
    results.nonShuffledReconLoadings = partial_loading;
    
    % Display ************************************************************************************************************************
    
    figure;
    
    % Original and reconstructed loadings
    plot(results.nonShuffledLoadings,results.nonShuffledReconLoadings,'.');
    
    % Unity line
    hline=refline(1,0);
    hline.Color = 'k';
    
    xlabel('Original loadings');ylabel('Reconstructed loadings');

    % Do the shuffled reconstruction *************************************************************************************************
    
    % Create indexes for nBoot random permutations using a loop
    permIndexes = NaN(nBoots,nFrames);
    for bootI = 1:nBoots
        permIndexes(bootI,:) = randperm(nFrames);
    end
    
    shuffAudioVecR = nan(1,nBoots);   % H1 metric per shuffle (vectorised r)



    allShuffledOrigLoad  = cell(nBoots,1);
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
            
            %% SHUFFLE WARPS  --- Build shuffled dataset: permute only the selected block ---

            switch shuffleTarget
                case 1  % shuffle MR frames
                    shMR  = thisMRWarpW(:, permIndexes(bootI,:));
                    shVID = thisVidWarpW;
                    shAUD = thisAudioW;
                case 2  % shuffle Video frames
                    shMR  = thisMRWarpW;
                    shVID = thisVidWarpW(:, permIndexes(bootI,:));
                    shAUD = thisAudioW;
                case 3  % shuffle Audio frames
                    shMR  = thisMRWarpW;
                    shVID = thisVidWarpW;
                    shAUD = thisAudioW(:, permIndexes(bootI,:));
                otherwise
                    error('shuffleTarget must be 1 (MR), 2 (Video), or 3 (Audio).');
            end
            shuffWarps = [shMR; shVID; shAUD];

            if VERBOSE && bootI==1
                fprintf('Shuffle check: permuting %s frames only.\n', blockNames{shuffleTarget});
            end


            [PCA,MorphMean,loadings] = doPCA(shuffWarps);
            
            partial_data = shuffWarps;
            partial_data(elementBoundaries(reconstructId)+1:elementBoundaries(reconstructId+1),:) = 0; % zero the target block
            partialMorphMean = mean(partial_data, 2);
            partial_centered = bsxfun(@minus, partial_data, partialMorphMean); % resizes partialMorphMean to make subtraction possible (could use matrix maths?)
            partial_loading = partial_centered'*PCA;


            %% === H1 Audio feature-space metric (SHUFFLED) ==========================
            idx1 = elementBoundaries(reconstructId)+1;
            idx2 = elementBoundaries(reconstructId+1);
            
            P_audio       = PCA(idx1:idx2, :);
            X_audio_true  = bsxfun(@minus, shuffWarps(idx1:idx2, :), partialMorphMean(idx1:idx2));
            X_audio_hat   = P_audio * (partial_loading.');
            
            aa = zscore(X_audio_true(:));
            bb = zscore(X_audio_hat(:));
            shuffAudioVecR(bootI) = corr(aa, bb);

            
            allShuffledOrigLoad{bootI} = loadings;
            allShuffledReconLoad{bootI} = partial_loading;
        end
    else
        disp('NOT using parallel processing...');
    end
    
    results.allShuffledOrigLoad = allShuffledOrigLoad;
    results.allShuffledReconLoad = allShuffledReconLoad;
    toc 
    
    %% Statistics ************************************************************************************************************************
    
    % === H1 Audio vectorised r: shuffle summary ============================
    realVecR = results.h1_vecR_real;
    sh_med   = median(shuffAudioVecR,'omitnan');
    sh_ci    = prctile(shuffAudioVecR,[2.5 97.5]);
    p_vecR   = mean(shuffAudioVecR >= realVecR);   % one-sided: shuffle >= real
    
    results.h1_vecR_shuff_all = shuffAudioVecR;
    results.h1_vecR_p         = p_vecR;
    results.h1_vecR_ci        = sh_ci;
    
    if VERBOSE
        fprintf('H1 (Audio) vectorised r: real=%.4f | shuffle median=%.4f | 95%% CI=[%.4f, %.4f] | p=%.3g\n', ...
                realVecR, sh_med, sh_ci(1), sh_ci(2), p_vecR);
    end




    % Unshuffled
    loadings1D = results.nonShuffledLoadings(:);
    reconLoadings1D = results.nonShuffledReconLoadings(:);
    
    SSE = sum((loadings1D-reconLoadings1D).^2); % sum of squared error
    [R,~] = corr(loadings1D,reconLoadings1D); % Pearson correlation
    p = polyfit(loadings1D,reconLoadings1D,1); % linear fit
    
    unshuffstats = [R p(1) SSE];
    
    % Shuffled
    shuffstats = NaN(3,nBoots);
    for bootI=1:nBoots
        loadings1D = results.allShuffledOrigLoad{bootI}(:);
        reconLoadings1D = results.allShuffledReconLoad{bootI}(:);
        
        SSE = sum((loadings1D-reconLoadings1D).^2); % sum of squared error
        [R,~] = corr(loadings1D,reconLoadings1D); % Pearson correlation
        p = polyfit(loadings1D,reconLoadings1D,1); % linear fit
        
        shuffstats(:,bootI) = [R p(1) SSE];
    end


    % After computing unshuffstats (R, slope, SSE) and shuffstats:
    p_R    = mean(shuffstats(1,:) >= unshuffstats(1));
    p_slope= mean(shuffstats(2,:) >= unshuffstats(2));
    p_SSE  = mean(shuffstats(3,:) <= unshuffstats(3)); % SSE lower is better
    
    if VERBOSE
        fprintf('Loadings-space stats (unshuffled): R=%.4f, slope=%.4f, SSE=%.3e\n', ...
                unshuffstats(1), unshuffstats(2), unshuffstats(3));
        fprintf('Permutation p-values: p_R=%.3g, p_slope=%.3g, p_SSE=%.3g\n', p_R, p_slope, p_SSE);
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


function [lam1, fro] = block_scale_stats(X)
% Row-centre within block (match the main script’s convention)
Xc  = bsxfun(@minus, X, mean(X,2));
% Leading eigenvalue of Xc'Xc equals the square of the top singular value
% Using economy SVD is fine here given T << p in your data layout
[~, S, ~] = svd(Xc' * Xc, 'econ');
lam1 = S(1,1);
fro  = norm(Xc, 'fro');
end