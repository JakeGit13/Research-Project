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
figure;
subplot(2,2,1);
plot(Y(:,2))
title('Audio before cleaning')
subplot(2,2,2);
plot(cleanAudio)
title('Audio after cleaning')

% If you'd like to listen to the recording
sound(cleanAudio,FS)
