% At the top
load(sprintf('sub%d_sen%d_cleaned.mat', subNum, senNum));

function pooledMFCCs = extractMFCCs(lpCleanAudio, lpFs)
% extractMFCCs - Extract and pool MFCCs to match MRI frame rate
%
% Input:
%   lpCleanAudio - cleaned and filtered audio signal
%   lpFs - sampling rate of the audio (typically 20000 Hz)
%
% Output:
%   pooledMFCCs - MFCCs pooled to 16fps and transposed to [13 × numFrames]

% Example MFCC extraction with standard audio settings (e.g., 25ms/10ms)
frameLength = round(0.025 * lpFs);     % 25 ms window
@@ -10,18 +16,15 @@
    'Window', hamming(frameLength, 'periodic'), ...
    'OverlapLength', frameOverlap);


actualFPS = lpFs / (frameLength - frameOverlap);
fprintf('Extracted MFCCs at approx. %.2f fps (%d frames, %d coeffs per frame)\n', ...
    actualFPS, size(coeffs,1), size(coeffs,2));

% Plot high-res MFCCs before pooling
figure;
imagesc(coeffs'); axis xy; colormap jet;
xlabel('Frame Index (~100fps)'); ylabel('MFCC Coefficient');
title('High-Resolution MFCCs (~100 fps)');


%% Plot high-res MFCCs before pooling 
% figure;
% imagesc(coeffs'); axis xy; colormap jet;
% xlabel('Frame Index (~100fps)'); ylabel('MFCC Coefficient');
% title('High-Resolution MFCCs (~100 fps)');

%% Pool MFCCs to match MRI 16fps
targetFPS = 16;
@@ -54,20 +57,17 @@
            pooledMFCCs(i,:) = zeros(1, size(coeffs,2));  % Zero padding for first frame
        end
    end

end

fprintf('Pooled MFCCs to %d frames at %d fps.\n', size(pooledMFCCs,1), targetFPS);

%% Plot pooled MFCCs - COMMENTED OUT
% figure;
% imagesc(pooledMFCCs'); axis xy; colormap jet;
% xlabel('Frame (16fps)'); ylabel('MFCC Coeff #');
% title('Pooled MFCCs aligned to MRI Frame Rate');

figure;
imagesc(pooledMFCCs'); axis xy; colormap jet;
xlabel('Frame (16fps)'); ylabel('MFCC Coeff #');
title('Pooled MFCCs aligned to MRI Frame Rate');


% Double check this is the right place to put it 
% Transpose to match PCA format [coeffs × frames]
pooledMFCCs = pooledMFCCs';  % Now [13 × numFrames] to match PCA format

% Figure out how to get this into the PCA now

end