function [lpAudio, lpFs] = processAudio(Y, FS)
    % Just take a single channel
    audio = Y(:,2);  % Or Y(:,1) if channel 1 is better
    
    % Low-pass filter and downsample to 20kHz
    targetFs = 30000;
    lpAudio = resample(audio, targetFs, FS);
    lpFs = targetFs;
end