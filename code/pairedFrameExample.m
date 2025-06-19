function pairedFrameExample
% pairedFrameExample
% SCRIPT FOR THE PAPER 'The inter-relationship between the face and vocal-tract configuration during audio-visual speech'
%   A PCA is performed on the hybrid MR/video, one modality (user-specified) is reconstructed, and the original and reconstructed loadings are displayed
%   The same process is then performed having paired up consecutive frames 
%   The single and paired loading correlations are then plotted against each other as in Figure 3b in the manuscript (if reconstructInd is set to 1 for MR)
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
% Date created:  9th Dec 2019
% Last modified: 22nd Sept 2019


% THINGS THAT YOU MAY WANT TO CHANGE ************************************************************************************************************
% Point to the location of the mat file 'mrAndVideoData.mat'
dataDir = '/Users/lpzcs/Documents/MATLAB/';

reconstructInd = 1; % 1 = MR, 2 = video
% ***********************************************************************************************************************************************


% Reset random seed
rng('default');

dataFile = 'mrAndVideoData.mat';

addpath(dataDir) % Add the user-defined data directory to the path 
load(dataFile,'data');

actors = [data.actor]; % Array of actor numbers
sentences = [data.sentence]; % Array of sentence numbers

%% PCA on hybrid facial video and vocal-tract MR images

for ii = 1:length(actors)
    ii
    
    % Select out data for this actor/sentence
    thisMRWarp = data(ii).mr_warp2D;
    thisVidWarp = data(ii).vid_warp2D;
    
    % Concatentate the MR and video data
    mixWarps = [thisMRWarp; thisVidWarp];
    
    % Perform a PCA on the hybrid data
    [origPCA,origMorphMean,origloadings] = doPCA(mixWarps);
    
    % Do the reconstruction ************************************************************************************************************************
    
    % Indexes of boundaries between MR and video sections
    warpSize = size(thisMRWarp);
    elementBoundaries = [0 warpSize(1) 2*warpSize(1)]; % this works because MR and video frames have the same number of pixels
    
    partial_data = mixWarps;
    partial_data(elementBoundaries(reconstructInd)+1:elementBoundaries(reconstructInd+1),:) = 0; % Set one modality to 0
    partialMorphMean = mean(partial_data, 2);
    partial_centered = bsxfun(@minus, partial_data, partialMorphMean);
    partial_loading = partial_centered'*origPCA;
    
    % Store the loadings for further processing
    [results(ii).loadingCorr,~] = corr(origloadings(:),partial_loading(:));
    
    % Pair up frames ************************************************************************************************************************
    
    % Frame 1 to N-1 atop Frame 2 to N
    pairedVidWarp = [thisVidWarp(:,1:end-1);thisVidWarp(:,2:end)];
    pairedMRWarp = [thisMRWarp(:,1:end-1);thisMRWarp(:,2:end)];
    
    pairedMixWarps = [pairedMRWarp; pairedVidWarp];
    
    % Perform a PCA on the paired hybrid data
    [pairedPCA,pairedMorphMean,pairedloadings] = doPCA(pairedMixWarps);
    
    % Indexes of boundaries between MR and video
    warpSize = size(pairedMRWarp);
    elementBoundaries = [0 warpSize(1) 2*warpSize(1)]; % this works because MR and video frames have the same number of pixels
    
    partial_data = pairedMixWarps;
    partial_data(elementBoundaries(reconstructInd)+1:elementBoundaries(reconstructInd+1),:) = 0; % Set one modality to 0
    partialMorphMean = mean(partial_data, 2);
    partial_centered = bsxfun(@minus, partial_data, partialMorphMean);
    paired_partial_loading = partial_centered'*pairedPCA;
    
    [results(ii).pairedLoadingCorr,~] = corr(pairedloadings(:),paired_partial_loading(:));
end

% Display ************************************************************************************************************************************

figure('defaultaxesfontsize',14);
uniqueActors = unique(actors);
plotCols = 'rgc';

for ii = 1:length(uniqueActors)
    thisActor = actors==uniqueActors(ii);
    
    % Original and reconstructed loadings
    plot([results(thisActor).loadingCorr],[results(thisActor).pairedLoadingCorr],'ko','markersize',10,'markerfacecolor',plotCols(ii));hold on
end

axis([.65 .93 .65 .93]);
hline=refline(1,0);
hline.Color = 'k';
xlabel('Single loading correlation');ylabel('Paired loading correlation');
grid on

end % end pairedFrameExample

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