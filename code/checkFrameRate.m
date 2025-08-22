function checkActualFrameRate()
    % Load your data
    dataDir = '/Users/jaker/Research-Project/data';
    load(fullfile(dataDir, 'mrAndVideoData.mat'), 'data');
    audioFile = 'C:\Users\jaker\Research-Project\data\audio\sub8_sen_256_6_svtimriMANUAL.wav';
    
    % Load and clean audio
    [Y, fs] = audioread(audioFile);
    if size(Y, 2) == 2
        audio = Y(:,2) - Y(:,1);
    else
        audio = Y;
    end
    
    % Get actual durations
    dataIdx = 9;
    nFrames = length(data(dataIdx).video_frames);
    audioDuration = length(audio) / fs;
    
    fprintf('=== Data Analysis ===\n');
    fprintf('Number of frames: %d\n', nFrames);
    fprintf('Audio duration: %.3f seconds\n', audioDuration);
    fprintf('Audio sample rate: %d Hz\n', fs);
    fprintf('Total audio samples: %d\n', length(audio));
    
    % Calculate what frame rate would make them match
    actualFrameRate = nFrames / audioDuration;
    fprintf('\n=== Calculated Frame Rate ===\n');
    fprintf('Frame rate needed for full audio: %.2f fps\n', actualFrameRate);
    
    % Test standard frame rates
    fprintf('\n=== Standard Frame Rates ===\n');
    standardRates = [15, 20, 24, 25, 30, 50, 60];
    for rate = standardRates
        duration = nFrames / rate;
        fprintf('%2d fps: %.3f seconds (%.1f%% of audio)\n', ...
            rate, duration, (duration/audioDuration)*100);
    end
    
    % Check if maybe there are more frames in other fields
    fprintf('\n=== Checking for additional data ===\n');
    fields = fieldnames(data(dataIdx));
    for i = 1:length(fields)
        field = fields{i};
        value = data(dataIdx).(field);
        if iscell(value)
            fprintf('%s: %d cells\n', field, length(value));
        elseif isnumeric(value)
            fprintf('%s: size = [%s]\n', field, num2str(size(value)));
        end
    end
end

% Run this to see what's going on
checkActualFrameRate();