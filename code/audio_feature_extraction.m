%% File path of the audio folder
audioFilePath = '/Users/jaker/Research-Project/data/audio';

% Construct full path to a specific file
audioFile = fullfile(audioFilePath, 'sub1_sen_252_1_svtimriMANUAL.wav');

% Video frame rate
mrFPS = 16;
frameDuration = 1/mrFPS; % Duration of each frame in seconds

% Create the file reader
fileReader = dsp.AudioFileReader(audioFile);

% Calculate samples per video frame
samplesPerFrame = round(fileReader.SampleRate * frameDuration);

% Set the reader to output the correct number of samples per frame
fileReader.SamplesPerFrame = samplesPerFrame;

% Create audio device writer
deviceWriter = audioDeviceWriter("SampleRate", fileReader.SampleRate);

% Display info
fprintf('Audio sample rate: %d Hz\n', fileReader.SampleRate);
fprintf('Video frame rate: %d fps\n', mrFPS);
fprintf('Samples per video frame: %d\n', samplesPerFrame);
fprintf('Frame duration: %.3f seconds\n', frameDuration);

%% Play the audio frame by frame
frameCount = 0;
tic; % Start timer

while ~isDone(fileReader)
    % Read one frame worth of audio data
    audioData = fileReader();
    
    % Write audio data to speakers
    deviceWriter(audioData);
    
    frameCount = frameCount + 1;
    
    % Optional: Display progress
    if mod(frameCount, mrFPS) == 0 % Every second
        fprintf('Playing... Frame %d (%.1f seconds)\n', frameCount, frameCount/mrFPS);
    end
end

%% Display playback info
totalTime = toc;
fprintf('Playback complete. Total frames: %d, Total time: %.2f seconds\n', frameCount, totalTime);

%% Release resources
release(fileReader)
release(deviceWriter)