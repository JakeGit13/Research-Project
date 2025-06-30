function [lpCleanAudio, lpFs, cleanAudio] = processAudio(Y, FS)
    % First channel contains scanner noise, subtract from second channel
    cleanAudio = Y(:,2) - Y(:,1);

    % Low-pass filter and downsample to 20kHz
    targetFs = 20000;
    lpCleanAudio = resample(cleanAudio, targetFs, FS);
    lpFs = targetFs;
end

%% Leave commented out for now

% %% Example function call 
% 
% audioFolder = '/Users/jaker/Research-Project/data/audio';
% 
% subNum = 8; % subject number
% senNum = 252; % sentence number
% 
% cd(audioFolder);
% filePattern = sprintf('sub%d_sen_%d*', subNum, senNum);
% thisFile = dir(filePattern);
% 
% if length(thisFile) > 1
%     error('File name is not unique - more than one match found.');
% elseif isempty(thisFile)
%     error('File not found: check the subject and sentence numbers.');
% end
% 
% [Y, FS] = audioread(thisFile.name);
% [~, filename, ~] = fileparts(thisFile.name);
% fprintf('Processing audio: %s\n', filename);
% 
% % Compute LPF version of raw audio (for plotting)
% lpRawAudio = resample(Y(:,2), 20000, FS);
% 
% % Process audio using function
% [lpCleanAudio, lpFs, cleanAudio] = processAudio(Y, FS);
% 
% 
% %% Plotting
% 
% % Spectrogram and Power Spectrum Parameters
% window = hamming(2048);
% noverlap = 1024;
% nfft = 4096;
% 
% % Calculate power spectra
% [Pxx_raw, F_raw] = pwelch(Y(:,2), window, noverlap, nfft, FS);
% [Pxx_proc, F_proc] = pwelch(lpCleanAudio, window, noverlap, nfft, lpFs);
% 
% % Define a consistent max frequency for plotting (e.g. 0–20 kHz)
% fMax = 20000;
% 
% % Create the figure
% figure('Name', 'Audio Processing Comparison', 'Position', [100, 100, 1200, 800]);
% 
% % Spectrogram - Original
% subplot(2,2,1);
% spectrogram(Y(:,2), window, noverlap, nfft, FS, 'yaxis');
% title('Original Audio (Noisy) - Spectrogram');
% ylabel('Frequency (Hz)');
% xlabel('Time (s)');
% ylim([0 fMax/1000]);
% 
% % Power Spectrum - Original
% subplot(2,2,2);
% plot(F_raw, 10*log10(Pxx_raw), 'k');
% title('Original Audio (Noisy) - Power Spectrum');
% xlabel('Frequency (Hz)');
% ylabel('Power (dB)');
% xlim([0 fMax]);
% grid on;
% 
% % Spectrogram - Processed
% subplot(2,2,3);
% spectrogram(lpCleanAudio, window, noverlap, nfft, lpFs, 'yaxis');
% title('Processed Audio - Spectrogram');
% xlabel('Time (s)');
% ylabel('Frequency (Hz)');
% ylim([0 fMax/1000]);
% 
% % Power Spectrum - Processed
% subplot(2,2,4);
% plot(F_proc, 10*log10(Pxx_proc), 'r');
% title('Processed Audio - Power Spectrum');
% xlabel('Frequency (Hz)');
% ylabel('Power (dB)');
% xlim([0 fMax]);
% grid on;
% 
% sgtitle('Audio Signal: Before and After Processing');
% 
% 
% %% === Playback ===
% disp('Playing: Original noisy audio');
% sound(Y(:,2), FS);
% pause(length(Y)/FS + 1);
% 
% disp('Playing: After noise cancellation');
% sound(cleanAudio, FS);
% pause(length(cleanAudio)/FS + 1);
% 
% disp('Playing: LPF + downsampled (raw)');
% sound(lpRawAudio, lpFs);
% pause(length(lpRawAudio)/lpFs + 1);
% 
% disp('Playing: LPF + downsampled (cleaned)');
% sound(lpCleanAudio, lpFs);
% pause(length(lpCleanAudio)/lpFs + 1);
