% ADD INFO HERE 

clearvars; clc;

%% Paths =====
projectRoot = '/Users/jaker/Research-Project/data';     % This should be the only thing that the user needs to set up? e.g. path to research project?  
addpath(projectRoot);

% Load in MR and video data struct
mrAndVid = load(fullfile(projectRoot, 'mrAndVideoData.mat'), 'data');   % MR / video data struct 
mrAndVideoData = mrAndVid.data;

% Path to raw audio folder
audioFolderPath = fullfile(projectRoot, 'Audio', 'Raw Audio');     % Where raw audio files are located

resultsRoot = fullfile(projectRoot, 'results'); % Where H1 / H2 results save to 

%% CSV files for results 
h1CSV = fullfile(resultsRoot, 'h1_results.csv');
h2_bimodalCSV = fullfile(resultsRoot, 'h2_bimodal.csv');
h2_trimodalCSV = fullfile(resultsRoot, 'h2_trimodal.csv');
scholes_bimodalCSV = fullfile(resultsRoot, 'scholes_bimodal.csv');


%% Controls ====
nBoots = 100;    % Universal across all tests (1000 as default)
targetAudioShare = 0.15; % Call on all tests % subject to change

generateCsv = false;

% Independent switches to write CSVs 
writeToCsv = false;

doH1 = true;
doH2_bimodal = false;
doH2_trimodal = false;
doScholes_bimodal = false; 





% Corrected map between MR/Video and Audio files [dataIdx, filename]
manifest = {
     9,  'sub8_sen_256_6_svtimriMANUAL.wav';
     1,  'sub8_sen_252_18_svtimriMANUAL.wav';   % Swapped dataIdx with ...
     5,  'sub1_sen_252_1_svtimriMANUAL.wav';    % ... this dataIdx
     6,  'sub8_sen_253_18_svtimriMANUAL.wav';
     7,  'sub8_sen_254_15_svtimriMANUAL.wav';
     8,  'sub8_sen_255_17_svtimriMANUAL.wav';
    10,  'sub8_sen_257_15_svtimriMANUAL.wav';
    11,  'sub8_sen_258_8_svtimriMANUAL.wav';
    12,  'sub8_sen_259_18_svtimriMANUAL.wav';
    13,  'sub8_sen_260_1_svtimriMANUAL.wav';
    14,  'sub8_sen_261_15_svtimriMANUAL.wav';
    18,  'sub14_sen_252_14_svtimriMANUAL.wav'
};


manifestLength = size(manifest,1) / 12 ; % 1 sentences for testing purposes 


for i = 1:manifestLength        % Loop through all 12 sentences using manifest

    fprintf("Run %d/%d\n",i,manifestLength);

    % Get necessary attributes for sentence i of the manifest 
    dataIdx = manifest{i,1};
    wavName = manifest{i,2};

    % Get direct path to .wav file of sentence i by concatenating 
    wavPath = fullfile(audioFolderPath, wavName);

    nFrames = size(mrAndVideoData(dataIdx).mr_warp2D, 2); % Number of MR / video frames 

    actorID    = mrAndVideoData(dataIdx).actor;   
    sentenceID = mrAndVideoData(dataIdx).sentence;

    preProcessedAudioStruct = processAudio(wavPath, nFrames, VERBOSE = false);

    audioFeatures = extractAudioFeatures(preProcessedAudioStruct,VERBOSE = true,useNoiseAudio=false);





    if doH1
        fprintf("Starting H1\n");
    
       
        rH1 = trimodalPCA(mrAndVideoData, audioFeatures, dataIdx, ...
                       reconstructId=3, ...     % 3 = Audio
                       nBoots=nBoots, ...
                       VERBOSE=true, ...
                       ifNormalise=true, ...
                       targetAudioShare= targetAudioShare, ...      
                       includeAudio= true);

        if ~isfile(h1CSV), generateEmptyCSV(rH1, h1CSV); end
        if writeToCsv && isfile(h1CSV), appendToCSV(rH1, h1CSV); end
    
    end

    fprintf("H1 Done\n");
   
    if doH2_bimodal
        fprintf("Starting H2 bimodal\n");
    

        %% BIMODAL BASELINE TESTS (NO AUDIO)
        rH2_bimodalMr = trimodalPCA(mrAndVideoData, audioFeatures, dataIdx, ...
               reconstructId=1, ...     % 1 = MR
               nBoots=nBoots, ...
               VERBOSE=true, ...
               ifNormalise=true, ...
               targetAudioShare= targetAudioShare, ...      
               includeAudio= false);    % without audio
        
        
        if ~isfile(h2_bimodalCSV), generateEmptyCSV(rH2_bimodalMr, h2_bimodalCSV); end % Only generate if no CSV file is there
        if writeToCsv && isfile(h2_bimodalCSV); appendToCSV(rH2_bimodalMr,h2_bimodalCSV); end   % Append these results for this sentence to the CSV 


        rH2_bimodalVid = trimodalPCA(mrAndVideoData, audioFeatures, dataIdx, ...
               reconstructId=2, ...     % 2 = VID
               nBoots=nBoots, ...
               VERBOSE=true, ...
               ifNormalise=true, ...
               targetAudioShare= targetAudioShare, ...      
               includeAudio= false);    % without audio
        
        
        if ~isfile(h2_bimodalCSV), generateEmptyCSV(rH2_bimodalVid, h2_bimodalCSV); end % Only generate if no CSV file is there
        if writeToCsv && isfile(h2_bimodalCSV); appendToCSV(rH2_bimodalVid,h2_bimodalCSV); end   % Append these results for this sentence to the CSV 
        
        fprintf("H2 Bimodal Done\n");
    end
        
    if doH2_trimodal
        
        fprintf("Starting H2 trimodal\n")
        %% TRIMODAL TESTS (WITH AUDIO)
        rH2_trimodalMr = trimodalPCA(mrAndVideoData, audioFeatures, dataIdx, ...
               reconstructId=1, ...     % 1 = MR
               nBoots=nBoots, ...
               VERBOSE=true, ...
               ifNormalise=true, ...
               targetAudioShare= targetAudioShare, ...      
               includeAudio= true);     % with audio
        
        if ~isfile(h2_trimodalCSV), generateEmptyCSV(rH2_trimodalMr, h2_trimodalCSV); end % Only generate if no CSV file is there
        if writeToCsv && isfile(h2_trimodalCSV); appendToCSV(rH2_trimodalMr,h2_trimodalCSV); end   % Append these results for this sentence to the CSV 


        rH2_trimodalVid = trimodalPCA(mrAndVideoData, audioFeatures, dataIdx, ...
               reconstructId=2, ...     % 2 = VID
               nBoots=nBoots, ...
               VERBOSE=true, ...
               ifNormalise=true, ...
               targetAudioShare= targetAudioShare, ...      
               includeAudio= true);  % with audio
        
        if ~isfile(h2_trimodalCSV), generateEmptyCSV(rH2_trimodalVid, h2_trimodalCSV); end % Only generate if no CSV file is there
        if writeToCsv && isfile(h2_trimodalCSV); appendToCSV(rH2_trimodalVid,h2_trimodalCSV); end   % Append these results for this sentence to the CSV 

        fprintf("H2 Trimodal Done\n");
    end


    %% Original SCHOLES bimodal shuffling PCA to compare to bimodal baseline
    if doScholes_bimodal
        fprintf("Starting Scholes bimodal\n")
        
        rH2_scholesMR = pcaAndShufflingExample(mrAndVideoData, dataIdx, ...
            reconstructId=1, ...    % 1 = MR
            nBoots=100);            % 100 Nboots just for this test
    

        if ~isfile(scholes_bimodalCSV), generateEmptyCSV(rH2_scholesMR, scholes_bimodalCSV); end % Only generate if no CSV file is there
        if writeToCsv && isfile(scholes_bimodalCSV); appendToCSV(rH2_scholesMR,scholes_bimodalCSV); end   % Append these results for this sentence to the CSV 


    
        rH2_scholesVid = pcaAndShufflingExample(mrAndVideoData, dataIdx, ...
            reconstructId=2, ...    % 2 = VID
            nBoots=100);            % 100 Nboots just for this test

        if ~isfile(scholes_bimodalCSV), generateEmptyCSV(rH2_scholesVid, scholes_bimodalCSV); end % Only generate if no CSV file is there
        if writeToCsv && isfile(scholes_bimodalCSV); appendToCSV(rH2_scholesVid,scholes_bimodalCSV); end   % Append these results for this sentence to the CSV

        fprintf("Scholes Bimodal Done\n");

    end

    




end


function appendToCSV(resultStruct, csvPath)

    % Read header (column order)
    options = detectImportOptions(csvPath);
    headerVariables = options.VariableNames;   % cell array of variable names

    % Build one row in header order; flatten non-scalars
    row = cell(1, numel(headerVariables));
    for k = 1:numel(headerVariables)
        name = headerVariables{k};
        v = resultStruct.(name);   % will error naturally if field missing

        if (isnumeric(v) || islogical(v)) && isscalar(v)
            row{k} = v;
        elseif ischar(v) || (isstring(v) && isscalar(v))
            row{k} = v;
        else
            row{k} = string(jsonencode(v));  % flatten arrays/structs/cells
        end
    end

    % Append the row
    T = cell2table(row, 'VariableNames', headerVariables);
    writetable(T, csvPath, 'WriteMode', 'append');
end

function generateEmptyCSV(resultStruct, csvPathIn)
    % Get field Names of the results from trimodal H1 / H2
    fn = fieldnames(resultStruct);

    %  Create a 0-row table with those columns to create the header
    T = cell2table(cell(0, numel(fn)), 'VariableNames', fn);

    % Write header-only CSV (will overwrite if exists)
    writetable(T, csvPathIn);

    fprintf('Created empty CSV at %s with %d columns.\n', csvPathIn, numel(fn));
end







