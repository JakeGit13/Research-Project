function trimodalPcaWithTrainTest

    clc;
    
    % Point to the location of the mat files
    dataDir = '/Users/jaker/Research-Project/data';
    
    usePar = true; % set to false if parallel processing isn't required/working
    reconstructInd = 3; % 1 = MR, 2 = video, 3 = audio
    trainFrac = 0.8; % proportion of frames used for training PCA
    nBoots = 100; % # bootstraps for shuffle test (kept small for debugging)
    
    rng('default'); % Reset random seed
    
    dataFile = 'mrAndVideoData.mat';
    audioFile = 'audioFeaturesData_articulatory.mat';
    
    addpath(dataDir);
    load(dataFile,'data');      % MR + video
    load(audioFile,'audioData'); % Audio
    
    
    %% Trimodal PCA
    for ii = 9 %:length(actors)
    
        % Select out data for this actor/sentence
        thisMRWarp  = data(ii).mr_warp2D;
        thisVidWarp = data(ii).vid_warp2D;
        thisAudio   = audioData(ii).audioFeatures_articulatory'; % features x frames
    
        % Shapes sanity check
        assert(size(thisMRWarp,2)==size(thisVidWarp,2),'MR/Video frames mismatch');
        assert(size(thisMRWarp,2)==size(thisAudio,2),'Audio frames mismatch');
        nFrames = size(thisMRWarp,2);
    
        fprintf('\nActor %d\n', ii);
        fprintf('MR   [features x frames]=[%d x %d]\n',size(thisMRWarp));
        fprintf('Video[features x frames]=[%d x %d]\n',size(thisVidWarp));
        fprintf('Audio[features x frames]=[%d x %d]\n',size(thisAudio));
    
        % --- Row-wise z-scoring ---
        zscore_rows = @(X) (X - mean(X,2)) ./ max(std(X,0,2), 1e-8);
        mrZ  = zscore_rows(thisMRWarp);
        vidZ = zscore_rows(thisVidWarp);
        audZ = zscore_rows(thisAudio);
    
        % --- Block weighting (equalise RMS) ---
        blockScale = @(X) sqrt(mean(X(:).^2));
        wmr  = 1 / (blockScale(mrZ)  + 1e-12);
        wvid = 1 / (blockScale(vidZ) + 1e-12);
        waud = 1 / (blockScale(audZ) + 1e-12);
    
        mrW  = wmr  * mrZ;
        vidW = wvid * vidZ;
        audW = waud * audZ;
    
        % --- Concatenate across features ---
        mixWarpsW = [mrW; vidW; audW];
        nMR     = size(thisMRWarp,1);
        nVideo  = size(thisVidWarp,1);
        nAudio  = size(thisAudio,1);
        elementBoundaries = [0 nMR nMR+nVideo nMR+nVideo+nAudio];
    
        % ---------------- Train/test split ----------------
        frameIdx = randperm(nFrames);
        nTrain   = round(trainFrac*nFrames);
        trainIdx = frameIdx(1:nTrain);
        testIdx  = frameIdx(nTrain+1:end);
    
        trainData = mixWarpsW(:,trainIdx);
        testData  = mixWarpsW(:,testIdx);
    
        % ---------------- Train PCA ----------------
        [prinComp, MorphMean, ~] = doPCA(trainData);
    
        % ---------------- Reconstruction on test data ----------------
        % Zero out target modality
        partial_data = testData;
        zeroedRange = elementBoundaries(reconstructInd)+1:elementBoundaries(reconstructInd+1);
        partial_data(zeroedRange,:) = 0;
    
        % Project into PC space and back
        partial_centered = bsxfun(@minus, partial_data, MorphMean);
        partial_loading  = partial_centered' * prinComp;
        reconstructed_full = prinComp * partial_loading';
    
        % Undo weighting
        if reconstructInd == 1
            reconstructed = reconstructed_full(1:nMR,:) / wmr;
            original      = thisMRWarp(:,testIdx);
            label = 'MR';
        elseif reconstructInd == 2
            reconstructed = reconstructed_full(nMR+1:nMR+nVideo,:) / wvid;
            original      = thisVidWarp(:,testIdx);
            label = 'Video';
        elseif reconstructInd == 3
            reconstructed = reconstructed_full(nMR+nVideo+1:end,:) / waud;
            original      = thisAudio(:,testIdx);
            label = 'Audio';
        end
    
        % ---------------- Evaluation ----------------
        originalVec      = original(:);
        reconstructedVec = reconstructed(:);
    
        R   = corr(originalVec, reconstructedVec);
        SSE = sum((originalVec - reconstructedVec).^2);
        p   = polyfit(originalVec, reconstructedVec, 1);
    
        fprintf('\n=== Trimodal Reconstruction (%s from others) ===\n', label);
        fprintf('Correlation (R): %.4f\n', R);
        fprintf('SSE: %.2f\n', SSE);
        fprintf('Linear fit gradient: %.4f\n', p(1));
    
        % ---------------- Shuffle control ----------------
        shuffR = zeros(1,nBoots);
        for b = 1:nBoots
            permIdx = randperm(length(testIdx));
            shuffOriginal = original(:,permIdx);
            shuffR(b) = corr(shuffOriginal(:), reconstructed(:));
        end
    
        figure;
        histogram(shuffR,50); hold on
        plot([R R], ylim, 'r--','LineWidth',2);
        xlabel('Shuffled correlation'); ylabel('Frequency');
        title(sprintf('Shuffle control (%s)', label));
    end
end


%% doPCA
function [prinComp,MorphMean,~] = doPCA(data)
MorphMean = mean(data, 2);
data = bsxfun(@minus, data, MorphMean);
xxt = data'*data;
[~,LSq,V] = svd(xxt);
LInv = 1./sqrt(diag(LSq));
prinComp = data * V * diag(LInv);
end
