%% Trimodal PCA Test - Extending the Shuffling Example
clear all; close all; clc;

fprintf('=== TRIMODAL PCA RECONSTRUCTION TEST ===\n\n');

%% 1. Load all data
% Audio
audioFile = '/Users/jaker/Research-Project/data/audio/sub1_sen_252_1_svtimriMANUAL.wav';
[Y, FS] = audioread(audioFile);
[lpCleanAudio, lpFs] = processAudio(Y, FS);

% Extract spectrograms
audioFeatures = extractSpectrograms(lpCleanAudio, lpFs);

% MR and Video
load('/Users/jaker/Research-Project/data/mrAndVideoData.mat', 'data');
thisMRWarp = data(5).mr_warp2D;
thisVideoWarp = data(5).vid_warp2D;

%% 2. Align and apply offset
numFrames = min([size(audioFeatures,2), size(thisMRWarp,2), size(thisVideoWarp,2)]);
offset = 8;

% Apply offset to audio
audioFeatures_shifted = [zeros(size(audioFeatures,1), offset), ...
                        audioFeatures(:, 1:end-offset)];

% Align all to same length
audioFeatures = audioFeatures_shifted(:, 1:numFrames);
thisMRWarp = thisMRWarp(:, 1:numFrames);
thisVideoWarp = thisVideoWarp(:, 1:numFrames);

%% 3. Store modality boundaries (crucial for reconstruction)
audioSize = size(audioFeatures, 1);
mrSize = size(thisMRWarp, 1);
videoSize = size(thisVideoWarp, 1);

% Boundaries: [start_audio | end_audio/start_MR | end_MR/start_video | end_video]
modalityBoundaries = [0, audioSize, audioSize+mrSize, audioSize+mrSize+videoSize];

fprintf('Modality sizes:\n');
fprintf('  Audio: %d features\n', audioSize);
fprintf('  MR: %d features\n', mrSize);
fprintf('  Video: %d features\n', videoSize);
fprintf('  Total: %d features × %d frames\n\n', modalityBoundaries(4), numFrames);

%% 4. Test all reconstruction combinations

% Bimodal baseline (Video → MR)
fprintf('BIMODAL BASELINE\n');
bimodalData = [thisMRWarp; thisVideoWarp];
[pca_bi, ~, loadings_bi] = doPCA(bimodalData);
partial_bi = bimodalData;
partial_bi(1:mrSize, :) = 0;  % Zero MR
recon_bi = reconstructFromPartial(partial_bi, pca_bi);
corr_baseline = corr(loadings_bi(:), recon_bi(:));
fprintf('Video → MR: %.3f\n\n', corr_baseline);

% Trimodal combinations
fprintf('TRIMODAL RECONSTRUCTIONS\n');
trimodalData = [audioFeatures; thisMRWarp; thisVideoWarp];
[pca_tri, ~, loadings_tri] = doPCA(trimodalData);

% Test 1: Audio+Video → MR
partial1 = trimodalData;
partial1(modalityBoundaries(2)+1:modalityBoundaries(3), :) = 0;  % Zero MR
recon1 = reconstructFromPartial(partial1, pca_tri);
corr1 = corr(loadings_tri(:), recon1(:));
fprintf('Audio+Video → MR: %.3f', corr1);
if corr1 > corr_baseline
    fprintf(' ✓ (improves on baseline!)\n');
else
    fprintf('\n');
end

% Test 2: Audio+MR → Video
partial2 = trimodalData;
partial2(modalityBoundaries(3)+1:modalityBoundaries(4), :) = 0;  % Zero Video
recon2 = reconstructFromPartial(partial2, pca_tri);
corr2 = corr(loadings_tri(:), recon2(:));
fprintf('Audio+MR → Video: %.3f\n', corr2);

% Test 3: MR+Video → Audio
partial3 = trimodalData;
partial3(modalityBoundaries(1)+1:modalityBoundaries(2), :) = 0;  % Zero Audio
recon3 = reconstructFromPartial(partial3, pca_tri);
corr3 = corr(loadings_tri(:), recon3(:));
fprintf('MR+Video → Audio: %.3f\n\n', corr3);

%% 5. Single modality reconstructions (for reference)
fprintf('SINGLE MODALITY → OTHERS\n');

% Audio → MR+Video
partial4 = trimodalData;
partial4(modalityBoundaries(2)+1:end, :) = 0;  % Keep only audio
recon4 = reconstructFromPartial(partial4, pca_tri);
corr4 = corr(loadings_tri(:), recon4(:));
fprintf('Audio → MR+Video: %.3f\n', corr4);

% Video → Audio+MR  
partial5 = trimodalData;
partial5(1:modalityBoundaries(3), :) = 0;  % Zero audio and MR
partial5(modalityBoundaries(3)+1:end, :) = trimodalData(modalityBoundaries(3)+1:end, :);  % Keep video
recon5 = reconstructFromPartial(partial5, pca_tri);
corr5 = corr(loadings_tri(:), recon5(:));
fprintf('Video → Audio+MR: %.3f\n', corr5);

% MR → Audio+Video
partial6 = trimodalData;
partial6([1:modalityBoundaries(2), modalityBoundaries(3)+1:end], :) = 0;  % Zero audio and video
partial6(modalityBoundaries(2)+1:modalityBoundaries(3), :) = trimodalData(modalityBoundaries(2)+1:modalityBoundaries(3), :);  % Keep MR
recon6 = reconstructFromPartial(partial6, pca_tri);
corr6 = corr(loadings_tri(:), recon6(:));
fprintf('MR → Audio+Video: %.3f\n', corr6);

%% 6. Summary
fprintf('\n=== KEY FINDINGS ===\n');
fprintf('Bimodal baseline (Video→MR): %.3f\n', corr_baseline);
fprintf('Trimodal (Audio+Video→MR): %.3f\n', corr1);
if corr1 > corr_baseline
    improvement = ((corr1 - corr_baseline) / corr_baseline) * 100;
    fprintf('→ Audio improves MR reconstruction by %.1f%%!\n', improvement);
else
    fprintf('→ Audio does not improve MR reconstruction\n');
end

%% Helper Functions
function [prinComp, MorphMean, loadings] = doPCA(data)
    MorphMean = mean(data, 2);
    data = bsxfun(@minus, data, MorphMean);
    xxt = data' * data;
    [~, LSq, V] = svd(xxt);
    LInv = 1./sqrt(diag(LSq));
    prinComp = data * V * diag(LInv);
    loadings = (data') * prinComp;
end

function reconLoadings = reconstructFromPartial(partialData, prinComp)
    partialMorphMean = mean(partialData, 2);
    partial_centered = bsxfun(@minus, partialData, partialMorphMean);
    reconLoadings = partial_centered' * prinComp;
end