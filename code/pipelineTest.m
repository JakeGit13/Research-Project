% Test script for audio preprocessing and MFCC extraction

audioFolder = '/Users/jaker/Research-Project/data/audio';

subNum = 8; % subject number
senNum = 252; % sentence number

cd(audioFolder);
filePattern = sprintf('sub%d_sen_%d*', subNum, senNum);
thisFile = dir(filePattern);

if length(thisFile) > 1
    error('File name is not unique - more than one match found.');
elseif isempty(thisFile)
    error('File not found: check the subject and sentence numbers.');
end

% Load audio
[Y, FS] = audioread(thisFile.name);
[~, filename, ~] = fileparts(thisFile.name);
fprintf('Processing audio: %s\n', filename);

% Step 1: Process audio
[lpCleanAudio, lpFs, cleanAudio] = processAudio(Y, FS);
fprintf('Audio cleaned and downsampled to %d Hz\n', lpFs);

% Step 2: Extract MFCCs
pooledMFCCs = extractMFCCs(lpCleanAudio, lpFs);
fprintf('MFCCs extracted: %d coefficients x %d frames\n', size(pooledMFCCs));

% Optional: Plot results
figure;
imagesc(pooledMFCCs); axis xy; colormap jet;
xlabel('Frame (16fps)'); ylabel('MFCC Coefficient');
title(sprintf('Pooled MFCCs for %s', filename));
colorbar;