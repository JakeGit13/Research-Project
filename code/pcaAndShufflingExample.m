function results = pcaAndShufflingExample(data, dataIdx, opts)
    % Inputs:
    %   data           : struct array with .mr_warp2D, .vid_warp2D 
    %   dataIdx        : index into 'data'
    
    %% Default options =======================================================================================
    arguments
        data
        dataIdx 
        opts.reconstructId (1,1) double = 1     % MR = 1, VIDEO = 2
        opts.nBoots (1,1) double = 1000
        opts.VERBOSE (1,1) logical = false 
        opts.genFigures (1,1) logical = false


    end

    reconstructId = opts.reconstructId;
    nBoots      = opts.nBoots;
    VERBOSE     = opts.VERBOSE;    
    genfigures  = opts.genFigures;
   
    usePar = true;  % set to false if parallel processing isn't required/working
            
    %% Select out data from this actor/sentence


    % Reset random seed
    rng('default');

    
    %% PCA on hybrid facial video and vocal-tract MR images
    
        
    % Select out data fro this actor/sentence
    thisMRWarp  = data(dataIdx).mr_warp2D;   % [p_MR x T]
    thisVidWarp = data(dataIdx).vid_warp2D;  % [p_VID x T]
    
    % Concatentate the MR and video data
    mixWarps = [thisMRWarp; thisVidWarp];
    
    % Perform a PCA on the hybrid data
    [origPCA,origMorphMean,origloadings] = doPCA(mixWarps);
    
    % Do the non-shuffled reconstruction for the original order **********************************************************************
    
    % Indexes of boundaries between MR and video
    warpSize = size(thisMRWarp);
    elementBoundaries = [0 warpSize(1) 2*warpSize(1)]; % this works because MR and video frames have the same number of pixels
    nFrames = warpSize(2);

    partial_data = mixWarps;
    partial_data(elementBoundaries(reconstructInd)+1:elementBoundaries(reconstructInd+1),:) = 0; % Set one modality to 0
    partialMorphMean = mean(partial_data, 2);
    partial_centered = bsxfun(@minus, partial_data, partialMorphMean);
    partial_loading = partial_centered'*origPCA;
    
    % Store the loadings for further processing
    results.nonShuffledLoadings = origloadings;
    results.nonShuffledReconLoadings = partial_loading;
    
    % Display ************************************************************************************************************************
    if genfigures
        figure;
        
        % Original and reconstructed loadings
        plot(results.nonShuffledLoadings,results.nonShuffledReconLoadings,'.');
        
        % Unity line
        hline=refline(1,0);
        hline.Color = 'k';
        
        xlabel('Original loadings');ylabel('Reconstructed loadings');
    end

    % Do the shuffled reconstruction *************************************************************************************************
    
    % Create indexes for nBoot random permutations using a loop
    permIndexes = NaN(nBoots,nFrames);
    for bootI = 1:nBoots
        permIndexes(bootI,:) = randperm(nFrames);
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
            % ************ Shuffle the MR warps ************ but keep the (time-matched) warps from the video
            shuffWarps = [thisMRWarp(:,permIndexes(bootI,:)); thisVidWarp];
            [PCA,MorphMean,loadings] = doPCA(shuffWarps);
            
            partial_data = shuffWarps;
            partial_data(elementBoundaries(reconstructInd)+1:elementBoundaries(reconstructInd+1),:) = 0; % set the MR section to 0
            partialMorphMean = mean(partial_data, 2);
            partial_centered = bsxfun(@minus, partial_data, partialMorphMean); % resizes partialMorphMean to make subtraction possible (could use matrix maths?)
            partial_loading = partial_centered'*PCA;
            
            allShuffledOrigLoad{bootI} = loadings;
            allShuffledReconLoad{bootI} = partial_loading;
        end
    else
        disp('NOT using parallel processing...');
        
        for bootI = 1:nBoots
            % ************ Shuffle the MR warps ************ but keep the (time-matched) warps from the video
            shuffWarps = [thisMRWarp(:,permIndexes(bootI,:)); thisVidWarp];
            [PCA,MorphMean,loadings] = doPCA(shuffWarps);
            
            partial_data = shuffWarps;
            partial_data(elementBoundaries(reconstructInd)+1:elementBoundaries(reconstructInd+1),:) = 0; % set the MR section to 0
            partialMorphMean = mean(partial_data, 2);
            partial_centered = bsxfun(@minus, partial_data, partialMorphMean); % resizes partialMorphMean to make subtraction possible (could use matrix maths?)
            partial_loading = partial_centered'*PCA;
            
            allShuffledOrigLoad{bootI} = loadings;
            allShuffledReconLoad{bootI} = partial_loading;
        end
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