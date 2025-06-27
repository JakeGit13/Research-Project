% How to deal with the audio files for the speech dataset featured in the
% PNAS paper

% Audiofile format:      sub1_sen_252_1_svtimriMANUAL.wav
% sub1_ = subject 1
% sen_252 = sentence 252 (can be x=252:261 or recoded as 1:10 [x-251])
% _1_ = repetition number (1:20)

clear

% If you wish to load the PNAS data file, put the directory here (otherwise
% just comment these lines out - you don't need the data to look at the audio
% alone) [CHANGE THIS]
% dataDir = '/Users/lpzcs/Library/CloudStorage/OneDrive-TheUniversityofNottingham/Documents/mrAndVideoSpeech/';
% load([dataDir 'mrAndVideoData.mat']);

% You can see what audio files we have in the audio folder, but to summarise:
% Sentence 1 (~252) for subjects 1, 8 and 14
% Sentence 1:10 (~252:261) for subject 8
subNum = 8; % which subject number for the audio
senNum = 252; % which sentence number for the audio

% Where are the audio files stored? [CHANGE THIS]
audioFolder = '/Users/jaker/Research-Project/data/audio';

% Navigate to the audio folder so we can use dir with a wildcard
cd(audioFolder)
thisFile = dir(['sub' num2str(subNum) '_sen_' num2str(senNum) '*']);

if length(thisFile)>1
    error('File name is not unique - more than one file has been found');
elseif length(thisFile)<1
    error('File not found: check the subject and sentence numbers')
end

% Pull in the audio data
[Y,FS] = audioread(thisFile.name);

% Looking at the audio
% First channel seems to contain the scanner noise which can be deleted,
% leading to a much cleaner audio
cleanAudio = Y(:,2)-Y(:,1);


%% Low-pass filter and downsample to 20kHz

targetFs = 20000;
lpCleanAudio = resample(cleanAudio, targetFs, FS);   % LPF + downsample
lpRawAudio = resample(Y(:,2), targetFs, FS);         % Raw audio also filtered

% Update sampling rate for visualisation
lpFs = targetFs;





%% Plotting
figure;
subplot(2,2,1);
plot(Y(:,2))
title('Audio before cleaning')
subplot(2,2,2);
plot(cleanAudio)
title('Audio after cleaning')



%% Create 2x2x2 comparison plot for filtered and unfiltered audio
figure('Name', 'Audio Comparison (Before/After Cleaning + Filtering)', ...
    'Position', [100, 100, 1400, 900]);

% Common spectrogram parameters
window = hamming(2048);
noverlap = 1024;
nfft = 4096;

% 1: Raw audio
subplot(3,2,1);
spectrogram(Y(:,2), window, noverlap, nfft, FS, 'yaxis');
title('Original (Raw)');
ylabel('Freq (Hz)');
xlabel('Time (s)');

% 2: Cleaned audio
subplot(3,2,2);
spectrogram(cleanAudio, window, noverlap, nfft, FS, 'yaxis');
title('After Cleaning');

% 3: LPF Original
subplot(3,2,3);
spectrogram(lpRawAudio, window, noverlap, nfft, lpFs, 'yaxis');
title('LPF + Downsampled (Original)');
ylabel('Freq (Hz)');
xlabel('Time (s)');

% 4: LPF Cleaned
subplot(3,2,4);
spectrogram(lpCleanAudio, window, noverlap, nfft, lpFs, 'yaxis');
title('LPF + Downsampled (Cleaned)');

% 5: Power Spectrum of original
subplot(3,2,5);
[Pxx_raw, F] = pwelch(Y(:,2), window, noverlap, nfft, FS);
plot(F/1000, 10*log10(Pxx_raw)); grid on;
title('Power Spectrum - Original'); xlabel('Freq (kHz)'); ylabel('Power (dB)');

% 6: Power Spectrum of LPF cleaned
subplot(3,2,6);
[Pxx_lpf, F2] = pwelch(lpCleanAudio, window, noverlap, nfft, lpFs);
plot(F2/1000, 10*log10(Pxx_lpf), 'r'); grid on;
title('Power Spectrum - LPF Cleaned'); xlabel('Freq (kHz)');




% Play original noisy audio
disp('Playing: Original noisy audio');
sound(Y(:,2), FS);      % Original
pause(length(Y)/FS + 1);

% Play after noise cancellation
disp('Playing: After noise cancellation');
sound(cleanAudio, FS);  % Cleaned
pause(length(cleanAudio)/FS + 1);

% Play after LPF + downsampling (original)
disp('Playing: LPF + downsampled (raw)');
sound(lpRawAudio, lpFs);  
pause(length(lpRawAudio)/lpFs + 1);

% Play after LPF + downsampling (cleaned)
disp('Playing: LPF + downsampled (cleaned)');
sound(lpCleanAudio, lpFs);  
pause(length(lpCleanAudio)/lpFs + 1);
