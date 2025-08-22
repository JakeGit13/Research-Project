function trimodalPcaAndShufflingExample

clc;

% Point to the location of the mat file 'mrAndVideoData.mat' 
dataDir = '/Users/jaker/Research-Project/data';

usePar = true; % set to false if parallel processing isn't required/working

reconstructInd = 3; % 1 = MR, 2 = video, 3 = audio

nBoots = 1000; % # bootstraps
% ******************************************************************************************************************************************************

% Reset random seed
rng('default');

dataFile = 'mrAndVideoData.mat';
audioFile = 'audioFeaturesData_articulatory.mat';

addpath(dataDir) % Add the data directory to the path 
load(dataFile,'data');  % MR + video
load(audioFile,'audioData');  % Audio


%% Trimodal PCA on hybrid facial video and vocal-tract MR images + audio

for ii = 9 %:length(actors)

    clear results; 
    
    % Select out data fro this actor/sentence
    thisMRWarp = data(ii).mr_warp2D;
    thisVidWarp = data(ii).vid_warp2D;
    thisAudio = audioData(ii).audioFeatures_articulatory'; % Transposed, should fix when actually extracting 


    % thisAudio = thisAudio(:, randperm(size(thisAudio,2))); % To shift
    % audio randomly 
    
    % Shapes
    fprintf('MR:   %d features x %d frames\n', size(thisMRWarp,1),  size(thisMRWarp,2));
    fprintf('Video:%d features x %d frames\n', size(thisVidWarp,1),  size(thisVidWarp,2));
    fprintf('Audio:%d features x %d frames\n', size(thisAudio,1),    size(thisAudio,2));
    
   % --- Row-wise z-scoring (each feature across frames, safe for zero-variance rows) ---
    zscore_rows = @(X) (X - mean(X,2)) ./ max(std(X,0,2), 1e-8);
    mrZ  = zscore_rows(thisMRWarp);
    vidZ = zscore_rows(thisVidWarp);
    audZ = zscore_rows(thisAudio);
    
    % --- Compute effective "block RMS" (root mean square per block) ---
    blockScale = @(X) sqrt(mean(X(:).^2));
    
    mrScale  = blockScale(mrZ);
    vidScale = blockScale(vidZ);
    audScale = blockScale(audZ);
    
    fprintf('Block RMS BEFORE weighting:\n');
    fprintf('  MR   RMS: %.3e\n', mrScale);
    fprintf('  VideoRMS: %.3e\n', vidScale);
    fprintf('  AudioRMS: %.3e\n', audScale);
    
    % --- Weight each block so average feature contribution is balanced ---
    wmr  = 1 / (mrScale + 1e-12);
    wvid = 1 / (vidScale + 1e-12);
    waud = 1 / (audScale + 1e-12);
    
    mrW  = wmr  * mrZ;
    vidW = wvid * vidZ;
    audW = waud * audZ;
    
    % --- Check after-weighting scales ---
    mrScaleW  = blockScale(mrW);
    vidScaleW = blockScale(vidW);
    audScaleW = blockScale(audW);
    
    fprintf('AFTER weighting (balanced per-feature scale):\n');
    fprintf('  MR   RMS: %.3e\n', mrScaleW);
    fprintf('  VideoRMS: %.3e\n', vidScaleW);
    fprintf('  AudioRMS: %.3e\n', audScaleW);
    
    % --- Concatenate across features ---
    mixWarpsW = [mrW; vidW; audW];


    % Perform a PCA on the hybrid data
    [origPCA,origMorphMean,origloadings] = doPCA(mixWarpsW);

    % After computing PCA (line 55), check variance explained
    variances = diag(origloadings' * origloadings);
    cumVar = cumsum(variances) / sum(variances);
    fprintf('\nVariance explained by first 10 PCs: %.2f%%\n', cumVar(10)*100);
    fprintf('Variance explained by first 40 PCs: %.2f%%\n', cumVar(40)*100);
    
    % How many PCs to explain 95% variance?
    pcs95 = find(cumVar > 0.95, 1);
    fprintf('PCs needed for 95%% variance: %d\n', pcs95);
        
    % Do the non-shuffled reconstruction for the original order **********************************************************************
    
    % Indexes of boundaries between MR and video
    warpSize = size(thisMRWarp);
    nAudioFeatures = size(thisAudio, 1);
    elementBoundaries = [0 warpSize(1) 2*warpSize(1) 2*warpSize(1)+nAudioFeatures];% Audio boundary should be 193,534
    nFrames = warpSize(2);

    fprintf('\n=== DEBUGGING RECONSTRUCTION ===\n');
    fprintf('MR features: %d\n', warpSize(1));
    fprintf('Video features: %d\n', size(thisVidWarp,1));
    fprintf('Audio features: %d\n', size(thisAudio,1));
    fprintf('Total features in mixWarps: %d x %d frames\n', size(mixWarpsW,1), size(mixWarpsW,2));
    fprintf('Element boundaries: [%d, %d, %d, %d]\n', elementBoundaries);
    fprintf('Reconstructing modality %d (1=MR, 2=Video, 3=Audio)\n', reconstructInd);

    partial_data = mixWarpsW;
    partial_data(elementBoundaries(reconstructInd)+1:elementBoundaries(reconstructInd+1),:) = 0;
    partialMorphMean = mean(partial_data, 2);
    partial_centered = bsxfun(@minus, partial_data, partialMorphMean);
    partial_loading = partial_centered'*origPCA;
    
    % Store the loadings for further processing
    results.nonShuffledLoadings = origloadings;
    results.nonShuffledReconLoadings = partial_loading;

    % Reconstruct the zeroed modality properly
    reconstructed_full = origPCA * partial_loading';
    
    if reconstructInd == 1  % MR reconstruction
        reconstructedMRW = reconstructed_full(1:96000, :);
        reconstructedMR = reconstructedMRW / wmr;  % Undo MR weighting
        originalMR = thisMRWarp;  % Original unweighted MR
        modalityReconCorr = corr(originalMR(:), reconstructedMR(:));
        fprintf('MR reconstruction from Video+Audio: %.4f\n', modalityReconCorr);
        
    elseif reconstructInd == 2  % Video reconstruction  
        reconstructedVidW = reconstructed_full(96001:192000, :);
        reconstructedVid = reconstructedVidW / wvid;  % Undo video weighting
        originalVid = thisVidWarp;  % Original unweighted video
        modalityReconCorr = corr(originalVid(:), reconstructedVid(:));
        fprintf('Video reconstruction from MR+Audio: %.4f\n', modalityReconCorr);
        
    elseif reconstructInd == 3  % Audio reconstruction
        reconstructedAudioW = reconstructed_full(elementBoundaries(3)+1:elementBoundaries(4), :); 
        reconstructedAudio = reconstructedAudioW / waud;  % Undo audio weighting
        originalAudio = thisAudio;  % Original unweighted audio
        modalityReconCorr = corr(originalAudio(:), reconstructedAudio(:));
        fprintf('Audio reconstruction from MR+Video: %.4f\n', modalityReconCorr);
    end
    
    results.modalityReconstructionCorr = modalityReconCorr;
        
  

    % Check what got zeroed
    zeroedRange = elementBoundaries(reconstructInd)+1:elementBoundaries(reconstructInd+1);
    fprintf('\nZeroing features %d to %d (Audio)\n', zeroedRange(1), zeroedRange(end));
    fprintf('Sum of audio region BEFORE zeroing: %.2f\n', sum(sum(abs(mixWarpsW(zeroedRange,:)))));
    fprintf('Sum of audio region AFTER zeroing: %.2f (should be 0)\n', sum(sum(abs(partial_data(zeroedRange,:)))));
   
    
    % Check other regions aren't zeroed
    fprintf('Sum of MR region after zeroing: %.2f (should be > 0)\n', sum(sum(abs(partial_data(1:96000,:)))));
    fprintf('Sum of Video region after zeroing: %.2f (should be > 0)\n', sum(sum(abs(partial_data(96001:192000,:)))));
       
    
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
            % ************ Shuffle the MR warps ************ but keep the (time-matched) warps from the videoshuffWarps
            shuffWarps = [mrW(:,permIndexes(bootI,:)); vidW; audW];  % Using the weighted versions                          
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
            shuffWarps = [mrW(:,permIndexes(bootI,:)); vidW; audW];
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


    % Print detailed reconstruction metrics
    fprintf('\n=== TRIMODAL RECONSTRUCTION RESULTS (Audio from MR+Video) ===\n');
    fprintf('Correlation (R): %.6f\n', R);
    fprintf('Sum of Squared Error (SSE): %.4f\n', SSE);
    fprintf('Linear fit gradient: %.4f\n', p(1));
    
    % Check if perfect reconstruction (BAD!)
    if R > 0.9999
        fprintf('WARNING: Near-perfect correlation suggests audio not being reconstructed!\n');
    end
    
    % Save results for comparison
    save('trimodal_results.mat', 'R', 'SSE', 'p', 'results');


    
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