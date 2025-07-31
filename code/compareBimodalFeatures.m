%% Compare Bimodal Features - MFCCs vs Spectrograms

clear all; close all; clc;

%% Load and preprocess audio
audioFile = '/Users/jaker/Research-Project/data/audio/sub1_sen_252_1_svtimriMANUAL.wav';
[Y, FS] = audioread(audioFile);
[lpCleanAudio, lpFs] = processAudio(Y, FS);

%% Load MR and Video data
load('/Users/jaker/Research-Project/data/mrAndVideoData.mat', 'data');
actorIdx = 5;  % Actor 8 (index 5)
thisMRWarp = data(actorIdx).mr_warp2D;
thisVideoWarp = data(actorIdx).vid_warp2D;

%% Set common parameters
offset = 2;  
numFrames = min([size(thisMRWarp,2), size(thisVideoWarp,2)]);

%% BASELINE TEST (Video -> MR)
fprintf('BASELINE VALIDATION\n');
fprintf('------------------\n');
videoMRData = [thisMRWarp(:,1:numFrames); thisVideoWarp(:,1:numFrames)];
[pca_base, ~, loadings_base] = doPCA(videoMRData);
partial_base = videoMRData;
partial_base(1:size(thisMRWarp,1), :) = 0;  % Zero MR, keep video
recon_base = reconstructFromPartial(partial_base, pca_base);
corr_baseline = corr(loadings_base(:), recon_base(:));
fprintf('Video → MR: %.3f (validates our implementation)\n\n', corr_baseline);

%% Extract MFCCs / Spectrograms
% Extract MFCCs
mfccFeatures = extractMFCCs(lpCleanAudio, lpFs);

% Extract Spectrograms  
spectrogramFeatures = extractSpectrograms(lpCleanAudio, lpFs);

% Align to video/MR length
numFrames = min([numFrames, size(mfccFeatures,2), size(spectrogramFeatures,2)]);
mfccFeatures = mfccFeatures(:, 1:numFrames);
spectrogramFeatures = spectrogramFeatures(:, 1:numFrames);
thisMRWarp = thisMRWarp(:, 1:numFrames);
thisVideoWarp = thisVideoWarp(:, 1:numFrames);

% Apply offset
mfccFeatures_shifted = [zeros(size(mfccFeatures,1), offset), ...
                        mfccFeatures(:, 1:end-offset)];
spectrogramFeatures_shifted = [zeros(size(spectrogramFeatures,1), offset), ...
                               spectrogramFeatures(:, 1:end-offset)];

%% Test both features
results = struct();

% MFCC Tests
fprintf('MFCC RESULTS\n');

% MFCC -> MR
audioMRData = [mfccFeatures_shifted; thisMRWarp];
[pca1, ~, loadings1] = doPCA(audioMRData);
partial1 = audioMRData;
partial1(size(mfccFeatures_shifted,1)+1:end, :) = 0;
recon1 = reconstructFromPartial(partial1, pca1);
results.mfcc_mr = corr(loadings1(:), recon1(:));

% MFCC -> Video
audioVideoData = [mfccFeatures_shifted; thisVideoWarp];
[pca2, ~, loadings2] = doPCA(audioVideoData);
partial2 = audioVideoData;
partial2(size(mfccFeatures_shifted,1)+1:end, :) = 0;
recon2 = reconstructFromPartial(partial2, pca2);
results.mfcc_video = corr(loadings2(:), recon2(:));

fprintf('MFCC -> MR:    %.3f\n', results.mfcc_mr);
fprintf('MFCC -> Video: %.3f\n\n', results.mfcc_video);

% Spectrogram Tests
fprintf('SPECTROGRAM RESULTS\n');

% Spectrogram -> MR
audioMRData = [spectrogramFeatures_shifted; thisMRWarp];
[pca3, ~, loadings3] = doPCA(audioMRData);
partial3 = audioMRData;
partial3(size(spectrogramFeatures_shifted,1)+1:end, :) = 0;
recon3 = reconstructFromPartial(partial3, pca3);
results.spec_mr = corr(loadings3(:), recon3(:));

% Spectrogram -> Video
audioVideoData = [spectrogramFeatures_shifted; thisVideoWarp];
[pca4, ~, loadings4] = doPCA(audioVideoData);
partial4 = audioVideoData;
partial4(size(spectrogramFeatures_shifted,1)+1:end, :) = 0;
recon4 = reconstructFromPartial(partial4, pca4);
results.spec_video = corr(loadings4(:), recon4(:));

fprintf('Spectrogram -> MR:    %.3f\n', results.spec_mr);
fprintf('Spectrogram -> Video: %.3f\n\n', results.spec_video);



%% Figures 

% Figure 1: Spectrogram Visualization 
figure('Name', 'Spectrogram Visualization', 'Position', [100 100 1200 600]);

% Full spectrogram
subplot(2,2,[1 2]);
% Extract just static features for visualization
numStaticFeatures = size(spectrogramFeatures,1) / 3;
staticSpec = spectrogramFeatures(1:numStaticFeatures, :);
imagesc((0:numFrames-1)/16, (0:numStaticFeatures-1)*8000/numStaticFeatures, staticSpec);
axis xy;
colormap jet;
xlabel('Time (seconds)');
ylabel('Frequency (Hz)');
title('Spectrogram of Speech Signal (Pooled to 16 fps)');
colorbar;

% Show individual frames as "images"
subplot(2,2,3);
frame10 = reshape(staticSpec(:,10), [], 1);
imagesc(repmat(frame10, 1, 20));  % Make it wider for visibility
title('Frame 10 as 2D Image');
ylabel('Frequency Bin');
colorbar;

subplot(2,2,4);
frame20 = reshape(staticSpec(:,20), [], 1);
imagesc(repmat(frame20, 1, 20));
title('Frame 20 as 2D Image');
ylabel('Frequency Bin');
colorbar;

sgtitle('Spectrograms: From Audio Signal to PCA-ready Frames');

% Figure 2: Feature Comparison (essential for paper)
figure('Name', 'Feature Comparison', 'Position', [100 100 800 600]);

% Bar chart comparison
subplot(2,1,1);
methods = categorical({'MFCCs', 'Spectrograms'});
mr_results = [results.mfcc_mr, results.spec_mr];
video_results = [results.mfcc_video, results.spec_video];

b = bar(methods, [mr_results; video_results]');
b(1).FaceColor = [0.8 0.2 0.2];  % Red for MR
b(2).FaceColor = [0.2 0.2 0.8];  % Blue for Video
ylabel('Reconstruction Correlation');
title('Audio Feature Comparison');
legend('Audio→MR', 'Audio→Video', 'Location', 'northwest');
ylim([0 0.7]);
grid on;

% Add value labels
xtips1 = b(1).XEndPoints;
ytips1 = b(1).YEndPoints;
labels1 = string(round(b(1).YData, 3));
text(xtips1, ytips1, labels1, 'HorizontalAlignment', 'center', ...
    'VerticalAlignment', 'bottom');

xtips2 = b(2).XEndPoints;
ytips2 = b(2).YEndPoints;
labels2 = string(round(b(2).YData, 3));
text(xtips2, ytips2, labels2, 'HorizontalAlignment', 'center', ...
    'VerticalAlignment', 'bottom');



% Add labels
for i = 1:length(dims)
    text(i, dims(i)+20, num2str(dims(i)), 'HorizontalAlignment', 'center');
end

%% Save figures 
saveas(figure(1), 'spectrogram_visualization.png');
saveas(figure(2), 'feature_comparison.png');
fprintf('\nFigures saved: spectrogram_visualization.png, feature_comparison.png\n');

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


