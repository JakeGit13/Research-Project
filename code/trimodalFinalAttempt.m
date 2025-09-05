function trimodalFinalAttempt

clc;

% THINGS THAT YOU MAY WANT TO CHANGE *******************************************************************************************************************
% Point to the location of the mat file 'mrAndVideoData.mat' 
dataDir = '/Users/jaker/Research-Project/data';
dataFile = 'mrAndVideoData.mat';
audioFile = 'audioFeaturesData_articulatory.mat';


usePar = true; % set to false if parallel processing isn't required/working
VERBOSE = true;


reconstructId = 1; % 1 = MR, 2 = video, 3 = audio

nBoots = 500; % # bootstraps
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


    partial_data = mixWarps;
    partial_data(elementBoundaries(reconstructId)+1:elementBoundaries(reconstructId+1),:) = 0; % Set one modality to 0
    partialMorphMean = mean(partial_data, 2);
    partial_centered = bsxfun(@minus, partial_data, partialMorphMean);
    partial_loading = partial_centered'*origPCA;
    
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
    
    allShuffledOrigLoad = cell(1000,1);
    allShuffledReconLoad = cell(1000,1);
    
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
            % ************ Shuffle the AUDIO block only ************
            % (MR and Video stay time-aligned; Audio is time-permuted)
            shuffWarps = [ w_mr  * thisMRWarp, ;  % unchanged MR
                           w_vid * thisVidWarp;   % unchanged Video
                           w_aud * thisAudio(:,permIndexes(bootI,:)) ]; % shuffled Audio


            [PCA,MorphMean,loadings] = doPCA(shuffWarps);
            
            partial_data = shuffWarps;
            partial_data(elementBoundaries(reconstructId)+1:elementBoundaries(reconstructId+1),:) = 0; % set the MR section to 0
            partialMorphMean = mean(partial_data, 2);
            partial_centered = bsxfun(@minus, partial_data, partialMorphMean); % resizes partialMorphMean to make subtraction possible (could use matrix maths?)
            partial_loading = partial_centered'*PCA;
            
            allShuffledOrigLoad{bootI} = loadings;
            allShuffledReconLoad{bootI} = partial_loading;
        end
    else
        disp('NOT using parallel processing...');
    end
    results.allShuffledOrigLoad = allShuffledOrigLoad;
    results.allShuffledReconLoad = allShuffledReconLoad;
    toc 
    
    % Statistics ************************************************************************************************************************
    
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
