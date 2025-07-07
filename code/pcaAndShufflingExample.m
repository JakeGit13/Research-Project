function pcaAndShufflingExample
% pcaAndShufflingExample
% SCRIPT FOR THE PAPER 'The inter-relationship between the face and vocal-tract configuration during audio-visual speech'
%   A PCA is performed on the hybrid MR/video, the MR is reconstructed, and the original and reconstructed loadings are displayed
%   The same process is then performed 1000 times when the MR frame order has been randomly-shuffled and loading comparison statistics are displayed  
%
%   Only one actor/sentence example is shown by default, as it takes some time to compute 1000 shuffled PCAs
%   Shuffling:
%       If more than one core is available then parallel processing will be used (this can be switched on/off using usePar)
%       On my ~2015 Mac Book Pro it takes around 3-4 minutes to run the shuffling for one actor/sentence, without using parallel processing
%       If the shuffling is taking too long then reduce the number of bootstraps (nBoots)
%
%   A note on the data:
%   data is a structure with an entry for each actor and sentence. It can be indexed into (e.g. data(1)) and the dot notation used to
%   access specific fields. For example, data(1).mr_frames gives a cell array containing the MR frames for actor 1 / sentence 252 and
%   data(6).vid_warp2D gives the video input to the PCA for actor 8 / sentence 256
%   Fields:
%   mr_frames - cell array of MR frames
%   video_frames - cell array of video frames
%   mr_warp2D - MR input to the PCA
%   vid_warp2D - video input to the PCA
%   actor - actor number ([1, 4, 6, 7, 8, 10, 12, 13, 14])
%   sentence - sentence number (1-10)
%
% Author: Chris Scholes / University of Nottingham / Department of Psychology
% Date created:  8th Dec 2019
% Last modified: 22nd Sept 2019


% THINGS THAT YOU MAY WANT TO CHANGE *******************************************************************************************************************
% Point to the location of the mat file 'mrAndVideoData.mat' 
dataDir = '/Users/lpzcs/Documents/MATLAB/';

usePar = true; % set to false if parallel processing isn't required/working

reconstructInd = 1; % 1 = MR, 2 = video

nBoots = 1000; % # bootstraps
% ******************************************************************************************************************************************************


% Reset random seed
rng('default');

dataFile = 'mrAndVideoData.mat';

addpath(dataDir) % Add the user-defined data directory to the path 
load(dataFile,'data');

actors = [data.actor]; % Array of actor numbers
sentences = [data.sentence]; % Array of sentence numbers

%% PCA on hybrid facial video and vocal-tract MR images

for ii = 1%:length(actors)
    
    % Select out data fro this actor/sentence
    thisMRWarp = data(ii).mr_warp2D;
    thisVidWarp = data(ii).vid_warp2D;
    
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
            % ************ Shuffle the MR warps ************ but keep the (time-matched) warps from the video
            shuffWarps = [thisMRWarp(:,permIndexes(bootI,:)); thisVidWarp];
            [PCA,MorphMean,loadings] = pcaScholesParallel(shuffWarps);
            
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
            [PCA,MorphMean,loadings] = pcaScholesParallel(shuffWarps);
            
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