%% Investigate the frequency content more carefully
clear; close all;

% Load your audio
subNum = 8; 
senNum = 252;
audioFolder = '/Users/jaker/Research-Project/data/audio';
cd(audioFolder)
thisFile = dir(['sub' num2str(subNum) '_sen_' num2str(senNum) '*']);
[Y,FS] = audioread(thisFile.name);
cleanAudio = Y(:,2) - Y(:,1);

%% Look at different frequency ranges
figure('Position', [100 100 1400 900]);

% Full frequency range
subplot(3,2,1);
spectrogram(Y(:,1), 2048, 1024, 2048, FS, 'yaxis');
title('Channel 1 - Full Range');
colorbar;

subplot(3,2,2);
spectrogram(cleanAudio, 2048, 1024, 2048, FS, 'yaxis');
title('Cleaned - Full Range');
colorbar;

% Speech frequencies (0-8kHz)
subplot(3,2,3);
spectrogram(Y(:,1), 2048, 1024, 2048, FS, 'yaxis');
ylim([0 8000]);
title('Channel 1 - Speech Range');
colorbar;

subplot(3,2,4);
spectrogram(cleanAudio, 2048, 1024, 2048, FS, 'yaxis');
ylim([0 8000]);
title('Cleaned - Speech Range');
colorbar;

% High frequencies (10-20kHz)
subplot(3,2,5);
spectrogram(Y(:,1), 2048, 1024, 2048, FS, 'yaxis');
ylim([10000 20000]);
title('Channel 1 - High Frequencies');
colorbar;

subplot(3,2,6);
spectrogram(cleanAudio, 2048, 1024, 2048, FS, 'yaxis');
ylim([10000 20000]);
title('Cleaned - High Frequencies');
colorbar;

%% Check the actual power of that 16kHz line
figure;
[Pxx_clean, F] = pwelch(cleanAudio, 2048, 1024, 4096, FS);
plot(F/1000, 10*log10(Pxx_clean));
xlabel('Frequency (kHz)');
ylabel('Power (dB)');
title('Power Spectrum of Cleaned Audio');
grid on;
xlim([0 20]);

% Mark 16kHz
%hold on;
%plot([16 16], ylim, 'r--', 'LineWidth', 2);
%text(16.2, mean(ylim), '16 kHz line', 'Color', 'red');