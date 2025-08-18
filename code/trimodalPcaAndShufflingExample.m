


function trimodalPcaAndShufflingExample
% pcaAndShufflingExample
% ADD DESCRIPTION LATER

% Point to the location of the mat file 'mrAndVideoData.mat' 
dataDir = '/Users/jaker/Research-Project/data';

usePar = true; % set to false if parallel processing isn't required/working

reconstructInd = 1; % 1 = MR, 2 = video, 3 = audio

nBoots = 1000; % # bootstraps
% ******************************************************************************************************************************************************


% Reset random seed
rng('default');

dataFile = 'mrAndVideoData.mat';
audioFile = 'audioFeaturesData.mat';

addpath(dataDir) % Add the data directory to the path 
load(dataFile,'data');  % MR + video
load(audioFile,'audioData');  % Audio




actors = [data.actor]; % Array of actor numbers
sentences = [data.sentence]; % Array of sentence numbers

%% PCA on hybrid facial video and vocal-tract MR images

for ii = 9 %:length(actors)

    clear results; 
    
    % Select out data fro this actor/sentence
    thisMRWarp = data(ii).mr_warp2D;
    thisVidWarp = data(ii).vid_warp2D;
    thisAudio = audioData(ii).audioFeatures; 

    % Check if audio is just constant or has meaningful variation
    fprintf('\n=== AUDIO DIAGNOSTIC ===\n');
    fprintf('Audio shape: %d frequencies x %d frames\n', size(thisAudio));
    fprintf('Audio range: [%.2f, %.2f]\n', min(thisAudio(:)), max(thisAudio(:)));
    fprintf('Audio variance per frame:\n');
    frameVars = var(thisAudio);
    fprintf('Min variance: %.4f, Max variance: %.4f\n', min(frameVars), max(frameVars));
    
    % Check if audio changes over time
    audioCorr = corr(thisAudio);
    fprintf('Mean correlation between frames: %.4f\n', mean(audioCorr(~eye(size(audioCorr)))));
    if mean(audioCorr(~eye(size(audioCorr)))) > 0.95
        warning('Audio frames are highly correlated - might be misaligned or constant!');
    end
    
    % Plot audio to visually inspect
    figure('Name', 'Audio Spectrogram Check');
    imagesc(thisAudio);
    colorbar;
    title('Audio Features - Should show variation across frames');
    xlabel('Frame'); ylabel('Frequency bin');

    % Normalise each modality to have similar scales
    thisMRWarp_norm = thisMRWarp / std(thisMRWarp(:));
    thisVidWarp_norm = thisVidWarp / std(thisVidWarp(:));
    thisAudio_norm = thisAudio / std(thisAudio(:));
    
    % Concatenate normalised versions
    mixWarps = [thisMRWarp_norm; thisVidWarp_norm; thisAudio_norm];
    
    fprintf('After normalisation:\n');
    fprintf('MR std: %.2f\n', std(thisMRWarp_norm(:)));
    fprintf('Video std: %.2f\n', std(thisVidWarp_norm(:)));
    fprintf('Audio std: %.2f\n', std(thisAudio_norm(:)));


    
    % Perform a PCA on the hybrid data
    [origPCA,origMorphMean,origloadings] = doPCA(mixWarps);
    
    % Do the non-shuffled reconstruction for the original order **********************************************************************
    
    % Indexes of boundaries between MR and video
    warpSize = size(thisMRWarp);
    nAudioFeatures = size(thisAudio, 1);
    elementBoundaries = [0 warpSize(1) 2*warpSize(1) 2*warpSize(1)+nAudioFeatures];
    nFrames = warpSize(2);

    fprintf('\n=== DEBUGGING RECONSTRUCTION ===\n');
    fprintf('MR features: %d\n', warpSize(1));
    fprintf('Video features: %d\n', size(thisVidWarp,1));
    fprintf('Audio features: %d\n', size(thisAudio,1));
    fprintf('Total features in mixWarps: %d x %d frames\n', size(mixWarps,1), size(mixWarps,2));
    fprintf('Element boundaries: [%d, %d, %d, %d]\n', elementBoundaries);
    fprintf('Reconstructing modality %d (1=MR, 2=Video, 3=Audio)\n', reconstructInd);

    partial_data = mixWarps;
    partial_data(elementBoundaries(reconstructInd)+1:elementBoundaries(reconstructInd+1),:) = 0; % Set one modality to 0
    partialMorphMean = mean(partial_data, 2);
    partial_centered = bsxfun(@minus, partial_data, partialMorphMean);
    partial_loading = partial_centered'*origPCA;
    
    % Store the loadings for further processing
    results.nonShuffledLoadings = origloadings;
    results.nonShuffledReconLoadings = partial_loading;

    % Extract the reconstructed audio from the loadings
    reconstructedAudio = origPCA(192001:192257, :) * partial_loading';
    originalAudio = mixWarps(192001:192257, :);
    
    % Compare
    audioReconCorr = corr(originalAudio(:), reconstructedAudio(:));
    fprintf('Direct audio reconstruction correlation: %.4f\n', audioReconCorr);
    
    % Plot comparison
    figure('Name', 'Audio Reconstruction Check');
    subplot(1,3,1); imagesc(originalAudio); title('Original Audio'); colorbar;
    subplot(1,3,2); imagesc(reconstructedAudio); title('Reconstructed Audio'); colorbar;
    subplot(1,3,3); imagesc(originalAudio - reconstructedAudio); title('Difference'); colorbar;

    % Check what got zeroed
    zeroedRange = elementBoundaries(reconstructInd)+1:elementBoundaries(reconstructInd+1);
    fprintf('\nZeroing features %d to %d (Audio)\n', zeroedRange(1), zeroedRange(end));
    fprintf('Sum of audio region BEFORE zeroing: %.2f\n', sum(sum(abs(mixWarps(zeroedRange,:)))));
    fprintf('Sum of audio region AFTER zeroing: %.2f (should be 0)\n', sum(sum(abs(partial_data(zeroedRange,:)))));
    
    % Verify the zeroing worked
    if sum(sum(abs(partial_data(zeroedRange,:)))) > 0.001
        error('AUDIO WAS NOT PROPERLY ZEROED!');
    end
    
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
            % ************ Shuffle the MR warps ************ but keep the (time-matched) warps from the video
            shuffWarps = [thisMRWarp_norm(:,permIndexes(bootI,:)); thisVidWarp_norm; thisAudio_norm];  % Added audio                             
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
            shuffWarps = [thisMRWarp_norm(:,permIndexes(bootI,:)); thisVidWarp_norm; thisAudio_norm];
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