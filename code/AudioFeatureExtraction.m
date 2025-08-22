%% === High-Dimensional Articulatory Audio Feature Extraction ===
% Extracts formants, MFCCs, prosody, and spectral features
% Dense sampling for high dimensionality while maintaining articulatory focus

clc; clear;

%% Load and clean audio
audioFilePath = 'C:\Users\jaker\Research-Project\data\audio\sub8_sen_256_6_svtimriMANUAL.wav';
[Y, FS] = audioread(audioFilePath);
cleanAudio = Y(:,2) - Y(:,1);

fprintf('[INFO] Audio sampling rate: %d Hz\n', FS);

%% Load MR/video frame count
load('C:\Users\jaker\Research-Project\data\mrAndVideoData.mat', 'data');
dataIdx = 9;  % sub8_sen_256
nFrames = size(data(dataIdx).mr_warp2D, 2);
fprintf('[INFO] Number of MRI/video frames: %d\n', nFrames);

%% Define time boundaries for each frame
totalDuration = length(cleanAudio) / FS;
frameEdges = linspace(0, totalDuration, nFrames + 1);

%% Parameters for dense feature extraction
% More windows = more features per frame
winDur_ms = 25;  % 25ms windows
hopDur_ms = 5;   % 5ms hop for dense sampling
winSamp = round(winDur_ms * FS / 1000);
hopSamp = round(hopDur_ms * FS / 1000);
hammingWin = hamming(winSamp, 'periodic');

%% Initialize feature storage
allFrameFeatures = [];

%% Process each frame
for frameIdx = 1:nFrames
    % Get audio segment for this frame
    t0 = frameEdges(frameIdx);
    t1 = frameEdges(frameIdx+1);
    segStart = max(1, floor(t0 * FS) + 1);
    segEnd = min(length(cleanAudio), floor(t1 * FS));
    frameAudio = cleanAudio(segStart:segEnd);
    
    % Ensure minimum length
    if length(frameAudio) < winSamp
        frameAudio = [frameAudio; zeros(winSamp - length(frameAudio), 1)];
    end
    
    frameFeatures = [];
    
    %% 1. MEL SPECTROGRAM (High-resolution, similar to your original)
    % This gives us spectral evolution within each frame
    numMelBands = 40;  % 40 mel bands
    freqRange = [50 8000];
    
    [melSpec, melFreqs, melTimes] = melSpectrogram(frameAudio, FS, ...
        'Window', hammingWin, ...
        'OverlapLength', winSamp - hopSamp, ...
        'NumBands', numMelBands, ...
        'FrequencyRange', freqRange);
    
    logMelSpec = log(melSpec + 1e-10);
    
    % Statistics across time for each mel band
    for band = 1:numMelBands
        bandData = logMelSpec(band, :);
        frameFeatures = [frameFeatures, ...
            mean(bandData), std(bandData), ...
            min(bandData), max(bandData), ...
            skewness(bandData), kurtosis(bandData), ...
            median(bandData), iqr(bandData)];
    end
    
    % DCT of mel bands (captures temporal patterns)
    if size(logMelSpec, 2) > 1
        melDCT = dct(logMelSpec, [], 2);  % DCT along time
        % Keep first 10 DCT coefficients per band
        nDCT = min(10, size(melDCT, 2));
        frameFeatures = [frameFeatures, reshape(melDCT(:, 1:nDCT), 1, [])];
    else
        frameFeatures = [frameFeatures, zeros(1, numMelBands * 10)];
    end
    
    %% 2. MFCCs WITH DELTAS (Dense extraction)
    % Extract MFCCs at multiple time points within the frame
    nWindows = max(1, floor((length(frameAudio) - winSamp) / hopSamp) + 1);
    mfccMatrix = [];
    
    for w = 1:nWindows
        wStart = (w-1) * hopSamp + 1;
        wEnd = min(wStart + winSamp - 1, length(frameAudio));
        segment = frameAudio(wStart:wEnd);
        
        if length(segment) < winSamp
            segment = [segment; zeros(winSamp - length(segment), 1)];
        end
        
        % Compute MFCCs using Window parameter (fixes the error)
        coeffs = mfcc(segment, FS, ...
            'NumCoeffs', 20, ...  % More coefficients for higher dimensionality
            'Window', hammingWin, ...
            'OverlapLength', 0);
        
        mfccMatrix = [mfccMatrix; coeffs];
    end
    
    % MFCC statistics
    for c = 1:20
        if size(mfccMatrix, 1) > 1
            mfccTrack = mfccMatrix(:, c);
            frameFeatures = [frameFeatures, ...
                mean(mfccTrack), std(mfccTrack), ...
                min(mfccTrack), max(mfccTrack), ...
                median(mfccTrack), prctile(mfccTrack, 25), prctile(mfccTrack, 75)];
            
            % Delta features
            mfccDelta = diff(mfccTrack);
            if ~isempty(mfccDelta)
                frameFeatures = [frameFeatures, mean(mfccDelta), std(mfccDelta)];
            else
                frameFeatures = [frameFeatures, 0, 0];
            end
        else
            frameFeatures = [frameFeatures, mfccMatrix(c), zeros(1, 8)];
        end
    end
    
    %% 3. FORMANT TRACKING (Multiple estimates per frame)
    % Extract formants at several points for trajectory information
    formantWinMs = 30;
    formantWinSamp = round(formantWinMs * FS / 1000);
    formantHop = round(10 * FS / 1000);  % 10ms hop
    nFormantWindows = max(1, floor((length(frameAudio) - formantWinSamp) / formantHop) + 1);
    
    formantMatrix = zeros(nFormantWindows, 5);  % Track 5 formants
    
    for w = 1:nFormantWindows
        wStart = (w-1) * formantHop + 1;
        wEnd = min(wStart + formantWinSamp - 1, length(frameAudio));
        segment = frameAudio(wStart:wEnd);
        
        if length(segment) >= formantWinSamp * 0.5  % Only process if segment is reasonable
            try
                % Pre-emphasis
                preEmph = filter([1 -0.97], 1, segment);
                
                % LPC analysis
                lpcOrder = round(FS/1000) + 4;
                a = lpc(preEmph .* hamming(length(preEmph)), lpcOrder);
                
                % Find formants
                rts = roots(a);
                rts = rts(imag(rts) > 0);
                angz = atan2(imag(rts), real(rts));
                freqs = sort(angz * (FS / (2*pi)));
                
                % Keep formants in speech range
                freqs = freqs(freqs > 90 & freqs < 8000);
                nFound = min(length(freqs), 5);
                formantMatrix(w, 1:nFound) = freqs(1:nFound);
            catch
                % Silent fail - zeros already in matrix
            end
        end
    end
    
    % Formant statistics
    for f = 1:5
        formantTrack = formantMatrix(:, f);
        formantTrack(formantTrack == 0) = NaN;
        
        if sum(~isnan(formantTrack)) > 0
            frameFeatures = [frameFeatures, ...
                nanmean(formantTrack), nanstd(formantTrack), ...
                nanmin(formantTrack), nanmax(formantTrack)];
        else
            frameFeatures = [frameFeatures, zeros(1, 4)];
        end
    end
    
    % Formant ratios and differences
    validF1 = formantMatrix(:,1); validF1(validF1==0) = NaN;
    validF2 = formantMatrix(:,2); validF2(validF2==0) = NaN;
    validF3 = formantMatrix(:,3); validF3(validF3==0) = NaN;
    
    if sum(~isnan(validF1)) > 0 && sum(~isnan(validF2)) > 0
        frameFeatures = [frameFeatures, ...
            nanmean(validF2./validF1), ...  % F2/F1 ratio
            nanmean(validF2 - validF1)];    % F2-F1 difference
    else
        frameFeatures = [frameFeatures, 0, 0];
    end
    
    if sum(~isnan(validF2)) > 0 && sum(~isnan(validF3)) > 0
        frameFeatures = [frameFeatures, ...
            nanmean(validF3./validF2), ...  % F3/F2 ratio
            nanmean(validF3 - validF1)];    % F3-F1 (formant dispersion)
    else
        frameFeatures = [frameFeatures, 0, 0];
    end
    
    %% 4. PROSODIC FEATURES (Dense extraction)
    % Pitch tracking
    [f0, f0_loc] = pitch(frameAudio, FS, ...
        'Method', 'NCF', ...
        'WindowLength', winSamp, ...  % Use scalar length, not vector
        'OverlapLength', winSamp - hopSamp, ...
        'Range', [50 400]);
    
    f0_voiced = f0(f0 > 0);
    
    if ~isempty(f0_voiced)
        frameFeatures = [frameFeatures, ...
            mean(f0_voiced), std(f0_voiced), ...
            min(f0_voiced), max(f0_voiced), ...
            median(f0_voiced), iqr(f0_voiced)];
        
        % F0 dynamics
        if length(f0_voiced) > 1
            f0Delta = diff(f0_voiced);
            frameFeatures = [frameFeatures, mean(f0Delta), std(f0Delta)];
        else
            frameFeatures = [frameFeatures, 0, 0];
        end
    else
        frameFeatures = [frameFeatures, zeros(1, 8)];
    end
    
    % Energy features at multiple scales
    % Short-time energy
    energyWin = round(10 * FS / 1000);  % 10ms
    energyHop = round(5 * FS / 1000);   % 5ms
    nEnergyWin = max(1, floor((length(frameAudio) - energyWin) / energyHop) + 1);
    
    energyTrack = zeros(nEnergyWin, 1);
    for w = 1:nEnergyWin
        wStart = (w-1) * energyHop + 1;
        wEnd = min(wStart + energyWin - 1, length(frameAudio));
        energyTrack(w) = sum(frameAudio(wStart:wEnd).^2);
    end
    
    logEnergy = log(energyTrack + 1e-10);
    frameFeatures = [frameFeatures, ...
        mean(logEnergy), std(logEnergy), ...
        min(logEnergy), max(logEnergy), ...
        skewness(logEnergy), kurtosis(logEnergy)];
    
    % RMS in frequency bands (more spectral detail)
    nBands = 10;
    for band = 1:nBands
        fLow = (band-1) * 4000/nBands;
        fHigh = band * 4000/nBands;
        if fLow < FS/2 && fHigh < FS/2 && fLow < fHigh
            try
                [b_band, a_band] = butter(3, [max(fLow,50) fHigh]/(FS/2), 'bandpass');
                bandSignal = filtfilt(b_band, a_band, frameAudio);
                bandRMS = rms(bandSignal);
                frameFeatures = [frameFeatures, log(bandRMS + 1e-10)];
            catch
                frameFeatures = [frameFeatures, 0];
            end
        else
            frameFeatures = [frameFeatures, 0];
        end
    end
    
    %% 5. SPECTRAL FEATURES (More comprehensive)
    % Multiple spectral shape descriptors
    spectralCentroidVals = spectralCentroid(frameAudio, FS, ...
        'Window', hammingWin, ...
        'OverlapLength', winSamp - hopSamp);
    
    spectralSpreadVals = spectralSpread(frameAudio, FS, ...
        'Window', hammingWin, ...
        'OverlapLength', winSamp - hopSamp);
    
    spectralFluxVals = spectralFlux(frameAudio, FS, ...
        'Window', hammingWin, ...
        'OverlapLength', winSamp - hopSamp);
    
    spectralRolloffVals = spectralRolloffPoint(frameAudio, FS, ...
        'Window', hammingWin, ...
        'OverlapLength', winSamp - hopSamp);
    
    spectralEntropyVals = spectralEntropy(frameAudio, FS, ...
        'Window', hammingWin, ...
        'OverlapLength', winSamp - hopSamp);
    
    % Add statistics for each spectral feature
    spectralFeatures = {spectralCentroidVals, spectralSpreadVals, ...
                       spectralFluxVals, spectralRolloffVals, spectralEntropyVals};
    
    for sf = 1:length(spectralFeatures)
        vals = spectralFeatures{sf};
        if ~isempty(vals)
            frameFeatures = [frameFeatures, ...
                mean(vals), std(vals), min(vals), max(vals), median(vals)];
        else
            frameFeatures = [frameFeatures, zeros(1, 5)];
        end
    end
    
    % Zero crossing rate
    zcr = sum(abs(diff(sign(frameAudio)))) / (2 * length(frameAudio));
    frameFeatures = [frameFeatures, zcr];
    
    %% 6. RAW SPECTRAL SNAPSHOTS (For higher dimensionality)
    % Take FFT at multiple points in the frame
    fftSize = 1024;
    nSnapshots = 10;  % 10 spectral snapshots per frame
    snapHop = max(1, floor(length(frameAudio) / nSnapshots));
    
    for snap = 1:nSnapshots
        snapStart = (snap-1) * snapHop + 1;
        snapEnd = min(snapStart + fftSize - 1, length(frameAudio));
        snapSegment = frameAudio(snapStart:snapEnd);
        
        if length(snapSegment) < fftSize
            snapSegment = [snapSegment; zeros(fftSize - length(snapSegment), 1)];
        end
        
        % Get magnitude spectrum (positive frequencies only)
        spectrum = abs(fft(snapSegment .* hamming(length(snapSegment))));
        spectrum = spectrum(1:fftSize/2);  % Keep positive frequencies
        
        % Downsample spectrum to 64 bins for manageability
        specBins = 64;
        binSize = length(spectrum) / specBins;
        for bin = 1:specBins
            binStart = floor((bin-1) * binSize) + 1;
            binEnd = min(floor(bin * binSize), length(spectrum));
            frameFeatures = [frameFeatures, mean(spectrum(binStart:binEnd))];
        end
    end
    
    %% Store features for this frame
    allFrameFeatures(frameIdx, :) = frameFeatures;
    
    % Progress update
    if mod(frameIdx, 10) == 0
        fprintf('[PROGRESS] Processed %d/%d frames\n', frameIdx, nFrames);
    end
end

%% Summary
featuresPerFrame = size(allFrameFeatures, 2);
fprintf('\n=== EXTRACTION COMPLETE ===\n');
fprintf('Total frames: %d\n', nFrames);
fprintf('Features per frame: %d\n', featuresPerFrame);
fprintf('\nFeature breakdown:\n');
fprintf('  - Mel spectrogram stats: %d features\n', 40*8 + 40*10);
fprintf('  - MFCCs (20 coeffs) with deltas: %d features\n', 20*9);
fprintf('  - Formants (5) with trajectories: %d features\n', 5*4 + 4);
fprintf('  - Prosody (F0, energy, band RMS): %d features\n', 8 + 6 + 10);
fprintf('  - Spectral shape features: %d features\n', 5*5 + 1);
fprintf('  - Raw spectral snapshots: %d features\n', 10*64);
fprintf('  TOTAL: %d features per frame\n', featuresPerFrame);

%% Save (NO normalization - that's for PCA script)
audioData(dataIdx).audioFeatures_articulatory = allFrameFeatures;

save('C:\Users\jaker\Research-Project\data\audioFeaturesData_articulatory.mat', 'audioData');
fprintf('\n[INFO] Saved raw features to: audioFeaturesData_articulatory.mat\n');
fprintf('[INFO] Features are NOT normalized - normalization should be done in PCA script\n');