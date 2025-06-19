function bubbleAnalysisExample
% bubbleAnalysisExample
% SCRIPT FOR THE PAPER 'The inter-relationship between the face and vocal-tract configuration during audio-visual speech'
%   Bubble masks are applied to one modality and the other modality is reconstructed
%   This process is repeated 10000 times with random bubble positions on each iteration
%   The sum of squared error between the original loadings (with no bubbles) and the reconstructed loadings for each iteration of bubbles is computed
%   A proportionPlane is created by dividing the sum of the masks that give the lowest 10% SSE with the sum of all masks
%   ProportionPlanes are displayed for the whole sequence and for individual frames for both facial video and MR images
%
%   Only one actor/sentence example is shown by default, as it takes some time to run this analysis
%   In addition, because the analysis takes so long, the images with the proportionPlane overlaid are saved into a .mat file in the current directory
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


% THINGS THAT YOU MAY WANT TO CHANGE *********************************************************************************************************
% Point to the location of the mat file 'mrAndVideoData.mat'
dataDir = '/Users/lpzcs/Documents/MATLAB/';

usePar = true; % set to false if parallel processing isn't required/working

% ********************************************************************************************************************************************

cwd = pwd; % current directory (where images will be saved)

% Bubble parameters
numBubb = 46;
numReps = 10000;
bubbSD = 5; % Because we end up using a Boolean mask, the disc diameter works out as ~12 pixels (FWHM = 5*2.3)
cutoffCrit = 0.1; % best N% of reconstructions based on loading SSE
cmapStr = 'jet';
set(0,'DefaultFigureColormap',feval(cmapStr)); % Set default colormap

% Reset random seed
rng('default');

dataFile = 'mrAndVideoData.mat';

addpath(dataDir) % Add the user-defined data directory to the path 
load(dataFile,'data');

actors = [data.actor]; % Array of actor numbers
sentences = [data.sentence]; % Array of sentence numbers

%% Create and store masks - WARNING, THIS CREATES HUGE ARRAYS (3 GB)
% Create random positions here, given hard-coded image dimensions for both MR and video
vidSize = [120,160];
mrSize = [160,120];

% Random X and Y co-ordinates for Gaussian centres
randLocs = rand(numReps,numBubb,2);

vidMask = NaN(numReps,vidSize(1),vidSize(2));
mrMask = NaN(numReps,mrSize(1),mrSize(2));
nCores = feature('numcores');

tic
if usePar && nCores>2
    poolOpen = gcp('nocreate');
    if isempty(poolOpen)
        pp = parpool(nCores-1);
    end
    
    parfor runI = 1:numReps
        tmpVid = gaussMask(vidSize,bubbSD,squeeze(randLocs(runI,:,:)));
        tmpMr = gaussMask(mrSize,bubbSD,squeeze(randLocs(runI,:,:)));
        vidMask(runI,:,:) = tmpVid;
        mrMask(runI,:,:) = tmpMr;
    end
else
    for runI = 1:numReps
        tmpVid = gaussMask(vidSize,bubbSD,squeeze(randLocs(runI,:,:)));
        tmpMr = gaussMask(mrSize,bubbSD,squeeze(randLocs(runI,:,:)));
        vidMask(runI,:,:) = tmpVid;
        mrMask(runI,:,:) = tmpMr;
    end
end
toc

%% PCA on hybrid facial video and vocal-tract MR images

for jj = 1%:length(actors)
    
    % Select out data for this actor/sentence
    thisMRWarp = data(jj).mr_warp2D;
    thisVidWarp = data(jj).vid_warp2D;
    
    % Concatentate the MR and video data
    mixWarps = [thisMRWarp; thisVidWarp];
    
    % Perform a PCA on the hybrid data
    [origPCA,origMorphMean,origloadings] = doPCA(mixWarps);
    
    % Indexes of boundaries between MR and video
    warpSize = size(thisMRWarp);
    elementBoundaries = [0 warpSize(1) 2*warpSize(1)]; % this works because MR and video frames have the same number of pixels
    nFrames = warpSize(2);
    
    %% Bubbles
    % To multiply the mask up we need to know how many times
    maskScaler = size(thisVidWarp,1) / prod(vidSize);
    
    for ii = 1:2 % First, reconstruct the MR, applying bubbles to the video; second, reconstruct the video, applying bubbles to the MR
        ii
        
        if ii==1
            allMask = vidMask;
            clear('vidMask');
            thisVid = data(jj).video_frames;
        else
            allMask = mrMask;
            clear('mrMask');
            thisVid = data(jj).mr_frames;
        end
        
        partial_data = mixWarps;
        partial_data(elementBoundaries(ii)+1:elementBoundaries(ii+1),:) = 0;
        partialMorphMean = mean(partial_data, 2);
        partial_centered = bsxfun(@minus, partial_data, partialMorphMean); % resizes partialMorphMean to make subtraction possible (could use matrix maths?)
        partial_loadings = partial_centered'*origPCA;
        
        % Reconstruction with no masks
        %         [origReconR,p] = corr(origloadings(:),partial_loadings(:));
        %         origReconS = polyfit(origloadings,partial_loadings,1);
        ll = length(origloadings);
        
        allLoad = NaN(numReps,ll,ll);
        tic
        if usePar && nCores>2
            parfor runI = 1:numReps
                
                mask = squeeze(allMask(runI,:,:));
                
                % Use discs rather than Gaussian blobs
                % For FWHM we just need to change >0.5 to 1 and <0.5 to 0
                mask(mask>=.5) = 1;
                mask(mask<.5) = 0;
                
                mask1D = mask(:);
                % To get mask1D up to size we'll use indexing (see http://www.vincentcheung.ca/research/matlabindexrepmat.html)
                mask2D = mask1D((1:size(mask1D,1))' * ones(1,maskScaler),(1:size(mask1D,2))' * ones(1,size(thisVidWarp,2)));
                if ii==1
                    maskedWarp = mask2D .* thisVidWarp;
                else
                    maskedWarp = mask2D .* thisMRWarp;
                end
                
                % Turn the RGB values back into integers
                sectionSize = warpSize/maskScaler;
                % This is just the pixel bit, not the warp bit
                maskedWarp((2*sectionSize+1):(5*sectionSize),:) = round(maskedWarp((2*sectionSize+1):(5*sectionSize),:));
                if maskScaler==10
                    maskedWarp((7*sectionSize+1):(10*sectionSize),:) = round(maskedWarp((2*sectionSize+1):(5*sectionSize),:));
                end
                
                if ii==1
                    mixBubbleWarps = [thisMRWarp; maskedWarp];
                else
                    mixBubbleWarps = [maskedWarp; thisVidWarp];
                end
                
                % Reconstruction of MR with bubbles
                partial_data = mixBubbleWarps;
                partial_data(elementBoundaries(ii)+1:elementBoundaries(ii+1),:) = 0;
                partialMorphMean = mean(partial_data, 2);
                partial_centered = bsxfun(@minus, partial_data, partialMorphMean); % resizes partialMorphMean to make subtraction possible (could use matrix maths?)
                partial_loading_bubbles = partial_centered'*origPCA;
                allLoad(runI,:,:) = partial_loading_bubbles;
            end
        else
            for runI = 1:numReps
                
                mask = squeeze(allMask(runI,:,:));
                
                % Use discs rather than Gaussian blobs
                % For FWHM we just need to change >0.5 to 1 and <0.5 to 0
                mask(mask>=.5) = 1;
                mask(mask<.5) = 0;
                
                mask1D = mask(:);
                % To get mask1D up to size we'll use indexing (see http://www.vincentcheung.ca/research/matlabindexrepmat.html)
                mask2D = mask1D((1:size(mask1D,1))' * ones(1,maskScaler),(1:size(mask1D,2))' * ones(1,size(thisVidWarp,2)));
                if ii==1
                    maskedWarp = mask2D .* thisVidWarp;
                else
                    maskedWarp = mask2D .* thisMRWarp;
                end
                
                % Turn the RGB values back into integers
                sectionSize = warpSize/maskScaler;
                % This is just the pixel bit, not the warp bit
                maskedWarp((2*sectionSize+1):(5*sectionSize),:) = round(maskedWarp((2*sectionSize+1):(5*sectionSize),:));
                if maskScaler==10
                    maskedWarp((7*sectionSize+1):(10*sectionSize),:) = round(maskedWarp((2*sectionSize+1):(5*sectionSize),:));
                end
                
                if ii==1
                    mixBubbleWarps = [thisMRWarp; maskedWarp];
                else
                    mixBubbleWarps = [maskedWarp; thisVidWarp];
                end
                
                % Reconstruction of MR with bubbles
                partial_data = mixBubbleWarps;
                partial_data(elementBoundaries(ii)+1:elementBoundaries(ii+1),:) = 0;
                partialMorphMean = mean(partial_data, 2);
                partial_centered = bsxfun(@minus, partial_data, partialMorphMean); % resizes partialMorphMean to make subtraction possible (could use matrix maths?)
                partial_loading_bubbles = partial_centered'*origPCA;
                allLoad(runI,:,:) = partial_loading_bubbles;
            end
        end
        toc
        
        % Bubbles across whole sequence *************************************************************************************************************
        
        % Compute SSE of loadings for each mask
        load2D = reshape(allLoad,[numReps ll*ll])';
        orig1D = origloadings(:);
        orig2D = orig1D(:,ones(1,numReps));
        SSE = sum((orig2D-load2D).^2);
        
        % Sort SSE and pick out best cutoffCrit% of reconstructions
        [sortCorr,sortI] = sort(SSE,'descend');
        selectInds = sortI((end-cutoffCrit*length(sortI)):end);
        
        % Put the proportion plane together
        % allMask is Gaussian - turn it into Boolean
        allMask(allMask>=.5) = 1;
        allMask(allMask<.5) = 0;
        
        totalPlane = squeeze(sum(allMask));
        correctPlane = squeeze(sum(allMask(selectInds,:,:)));
        propPlane = correctPlane./totalPlane;
        
        % Create and save mask overlaid on to first frame of MR/video sequence
        if ii==1
            imB = 2*rgb2gray(thisVid{1});
        else
            imB = 2*rgb2gray(thisVid{1});
        end
        imF = propPlane;
        [hF,hB] = imoverlay(imB,imF,[min(imF(:)) max(imF(:))],[min(imB(:)) max(imB(:))],'hot',.6);
        
        F=getframe;
        wholeSequencePropPlane{ii} = F.cdata;
        
        % Save the images to the current directory
        save('bubbleImages','wholeSequencePropPlane');
        disp(['Whole sequence images saved to ' cwd filesep 'bubbleImages.mat']);
        
        % Bubbles for each frame ********************************************************************************
        
        % SSE by frame
        perFrameSSE = NaN(size(allLoad,1),size(allLoad,2));
        tic
        for kk = 1:size(allLoad,1)
            perFrameSSE(kk,:) = sum((origloadings-squeeze(allLoad(kk,:,:))).^2,2);
        end
        toc
        
        % Picking stuff out based on just correlation/gradient
        allPropPlane = NaN([size(allLoad,2),size(totalPlane)]);
        % Select out nBoot values for this frame
        for kk = 1:size(allLoad,2)
            theseVals = perFrameSSE(:,kk);
            
            % Sort the values, retaining the indices
            [sortVals,sortI] = sort(theseVals,'descend');
            
            % Select the last N percent of indices (will be either highest or lowest)
            selectInds = sortI((end-cutoffCrit*length(sortI)):end);
            % Index into the stored masks to retrieve the N percent and add together
            correctPlane = squeeze(sum(allMask(selectInds,:,:)));
            
            % Divide by the total sum
            allPropPlane(kk,:,:) = correctPlane./totalPlane;
        end
        
        % Store frames with Bubble mask overlaid
        tmp = cell(1,size(allLoad,2));
        for kk = 1:size(allLoad,2)
            
            if ii==1
                imB = 2.*rgb2gray(thisVid{kk});
            else
                imB = 5.*rgb2gray(thisVid{kk});
            end
            
            imF = squeeze(allPropPlane(kk,:,:));
            [hF,hB] = imoverlay(imB,imF,[min(imF(:)) max(imF(:))],[min(imB(:)) max(imB(:))],cmapStr,.4);
            F=getframe;
            tmp{kk} = F.cdata;
            close all
        end
        eachFramePropPlane{ii} = tmp;
        
        % Save the images to the current directory
        save('bubbleImages','eachFramePropPlane','-append');
        disp(['Frame images saved to ' cwd filesep 'bubbleImages.mat']);
        
    end
    
    % Display ************************************************************************************************************************************
    
    % Whole sequence
    figure;
    subplot(2,2,1);
    imagesc(wholeSequencePropPlane{1});
    title('Whole face sequence');
    subplot(2,2,2);
    imagesc(wholeSequencePropPlane{2});
    title('Whole vocal-tract sequence');
    
    % Across frames
    figure;
    axAcr = 6;
    axDown = ceil(nFrames/axAcr);
    
    for ii = 1:nFrames
        subplot(axDown,axAcr,ii);
        imagesc([eachFramePropPlane{2}{ii} imresize(eachFramePropPlane{1}{ii},mrSize)])
        title(['Frame ' num2str(ii)]);
    end

end
end % end bubbleAnalysisExample


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


%% gaussMask
function [mask,numGauss] = gaussMask(imSize,sigma,numGauss)
% If numGauss is an integer then numGauss Gaussians are randomly assigned locations
% If numGauss is a list of positions then these positions are used to create the mask

xSize = imSize(2); % columns
ySize = imSize(1); % rows

X = 1:xSize;                           % X is a vector from 1 to imageSize
X0 = (X / xSize) - .5;                 % rescale X -> -.5 to .5
Y = 1:ySize;                           % X is a vector from 1 to imageSize
Y0 = (Y / ySize) - .5;                 % rescale Y -> -.5 to .5

[Xm, Ym] = meshgrid(X0, Y0);             % 2D matrices

% gaussian width as fraction of x and y image sizes
sX = sigma ./ xSize;
sY = sigma ./ ySize;

% Where do the Gaussians go? Random or user-specified
% Randomly create x,y posns
if length(numGauss)==1
    tmp = numGauss;
    numGauss = rand([2,tmp]); % random numbers between 0 and 1
end

xStart0 = -.5;
xOff = xStart0 + numGauss(:,1) .* (.5-xStart0);
yOff = numGauss(:,2) - .5;

% Make 2D gaussian blob
mask = zeros(imSize(1:2));
for ii = 1:length(xOff)
    xTerm = ( ( Xm - xOff(ii) ) .^2 )  ./ (2*sX^2);
    yTerm = ( ( Ym - yOff(ii) ) .^2 )  ./ (2*sY^2);
    gauss = exp( -(xTerm+yTerm)); % formula for 2D gaussian
    mask = max(mask,gauss);
end
end % end gaussMask


%% imoverlay - third-party function
function [hF,hB] = imoverlay(B,F,climF,climB,cmap,alpha,haxes)
% IMOVERLAY(B,F) displays the image F transparently over the image B.
%    If the image sizes are unequal, image F will be scaled to the aspect
%    ratio of B.
%
%    [hF,hB] = imoverlay(B,F,[low,high]) limits the displayed range of data
%    values in F. These values map to the full range of values in the
%    current colormap.
%
%    [hF,hB] = imoverlay(B,F,[],[low,high]) limits the displayed range of
%    data values in B.
%
%    [hF,hB] = imoverlay(B,F,[],[],map) applies the colormap to the figure.
%    This can be an array of color values or a preset MATLAB colormaps
%    (e.g. 'jet' or 'hot').
%
%    [hF,hB] = imoverlay(B,F,[],[],[],alpha) sets the transparency level to
%    alpha with the range 0.0 <= alpha <= 1.0, where 0.0 is fully
%    transparent and 1.0 is fully opaque.
%
%    [hF,hB] = imoverlay(B,F,[],[],[],[],ha) displays the overlay in the
%    axes with handle ha.
%
%    [hF,hB] = imoverlay(...) returns the handles to the front and back
%    images.
%
%
% Author: Matthew Smith / University of Wisconsin / Department of Radiology
% Date created:  February 6, 2013
% Last modified: Jan 2, 2015
%
%
%  Examples:
%
%     % Overlay one image transparently onto another
%     imB = phantom(256);                       % Background image
%     imF = rgb2gray(imread('ngc6543a.jpg'));   % Foreground image
%     [hf,hb] = imoverlay(imB,imF,[40,180],[0,0.6],'jet',0.6);
%     colormap('parula'); % figure colormap still applies
%
%
%     % Use the interface for flexibility
%     imoverlay_tool;
%
%
% See also IMOVERLAY_TOOL, IMAGESC, HOLD, CAXIS.
ALPHADEFAULT = 0.4; % Default transparency value
CMAPDEFAULT = 'parula';
if nargin == 0,
    try
        imoverlay_tool;
        return;
    catch
        errordlg('Cannot find imoverlay_tool.', 'Error');
    end
end
% Check image sizes
if size(B,3) > 1
    error('Back image has %d dimensions!\n',length(size(B)));
end
if size(F,3) > 1
    error('Front image has %d dimensions!\n',length(size(F)));
end
if ~isequal(size(B),size(F))
    fprintf('Warning! Image sizes unequal. Undesired scaling may occur.\n');
end
% Check arguments
if nargin < 7
    haxes = [];
end
if nargin < 6 || isempty(alpha)
    alpha = ALPHADEFAULT;
end
if nargin < 5 || isempty(cmap)
    cmap = CMAPDEFAULT;
end
if nargin < 4 || isempty(climB)
    climB = [min(B(:)), max(B(:))];
end
if nargin < 3 || isempty(climF)
    climF = [min(F(:)), max(F(:))];
end
if abs(alpha) > 1
    error('Alpha must be between 0.0 and 1.0!');
end
% Create a figure unless axes is provided
if isempty(haxes) || ~ishandle(haxes)
    f=figure('Visible','off',...
        'Units','pixels','Renderer','opengl');
    pos = get(f,'Position');
    set(f,'Position',[pos(1),pos(2),size(B,2),size(B,1)]);
    haxes = axes;
    set(haxes,'Position',[0,0,1,1]);
    movegui(f,'center');
    
    % ADDED BY SCHOLES 20/08/19
    % Assume that colormap is already set correctly
    %     % Create colormap
    %     cmapSize = 100; % default size of 60 shows visible discretization
    %     if ischar(cmap)
    %
    %         try
    %             cmap = eval([cmap '(' num2str(cmapSize) ');']);
    %         catch
    %             fprintf('Colormap ''%s'' is not supported. Using ''jet''.\n',cmapName);
    %             cmap = jet(cmapSize);
    %         end
    %     end
    %     colormap(cmap);
end
% To have a grayscale background, replicate image to 3-channels
B = repmat(mat2gray(double(B),double(climB)),[1,1,3]);
% Display the back image
axes(haxes);
hB = imagesc(B);axis image off;
% set(gca,'Position',[0,0,1,1]);
% Add the front image on top of the back image
hold on;
hF = imagesc(F,climF);
% If images are different sizes, map the front image to back coordinates
set(hF,'XData',get(hB,'XData'),...
    'YData',get(hB,'YData'))
% Make the foreground image transparent
alphadata = alpha.*(F >= climF(1));
set(hF,'AlphaData',alphadata);
if exist('f')
    set(f,'Visible','on');
end
% Novel colormaps
%
% JET2 is the same as jet but with black base
    function J = jet2(m)
        if nargin < 1
            m = size(get(gcf,'colormap'),1);
        end
        J = jet(m); J(1,:) = [0,0,0];
    end
% JET3 is the same as jet but with white base
    function J = jet3(m)
        if nargin < 1
            m = size(get(gcf,'colormap'),1);
        end
        J = jet(m); J(1,:) = [1,1,1];
    end
% PARULA2 is the same as parula but with black base
    function J = parula2(m)
        if nargin < 1
            m = size(get(gcf,'colormap'),1);
        end
        J = parula(m); J(1,:) = [0,0,0];
    end
% HSV2 is the same as HSV but with black base
    function map = hsv2(m)
        map =hsv;
        map(1,:) = [0,0,0];
    end
% HSV3 is the same as HSV but with white base
    function map = hsv3(m)
        map =hsv;
        map(1,:) = [1,1,1];
    end
% HSV4 a slight modification of hsv (Hue-saturation-value color map)
    function map = hsv4(m)
        if nargin < 1, m = size(get(gcf,'colormap'),1); end
        h = (0:m-1)'/max(m,1);
        if isempty(h)
            map = [];
        else
            map = hsv2rgb([h h ones(m,1)]);
        end
    end
end % end imoverlay
