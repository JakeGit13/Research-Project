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
h1_audioCSV = fullfile(resultsRoot, 'h1_audio_results.csv');
h2_bimodalCSV = fullfile(resultsRoot, 'h2_bimodal_results.csv');
h2_trimodalCSV = fullfile(resultsRoot, 'h2_trimodal_results.csv');
scholes_bimodalCSV = fullfile(resultsRoot, 'scholes_bimodal_results.csv');


%% Controls ====
nBoots = 500;    % Universal across all tests (1000 as default)
targetAudioShare = 0.15; % Call on all tests % subject to change

% Independent switches to write CSVs 
writeToCsv = false;

doH1_audio = false;
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


manifestLength = size(manifest,1); % 12


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

    preProcessedAudioStruct = processAudio(wavPath, nFrames, VERBOSE = false,genFigures=false);

    audioFeatures = extractAudioFeatures(preProcessedAudioStruct,VERBOSE = false,genFigures=false,useNoiseAudio=false);


    if doH1_audio
        fprintf("Starting H1 Audio\n");
    
        %% MR+VID -> AUD 
        H1_trimodal_AUD_MR_VID = trimodalPCA(mrAndVideoData, audioFeatures, dataIdx, ...
                       reconstructId=3, ...     % 3 = Audio
                       nBoots=nBoots, ...
                       VERBOSE=false, ...
                       ifNormalise=true, ...
                       targetAudioShare= targetAudioShare, ...      
                       includeAudio= true, ...
                       h1Source="MRVID");

        if ~isfile(h1_audioCSV)
            generateEmptyCSV(H1_trimodal_AUD_MR_VID, h1_audioCSV, "H1_trimodal_AUD_MR_VID"); 
        end

        if writeToCsv && isfile(h1_audioCSV), appendToCSV(H1_trimodal_AUD_MR_VID, h1_audioCSV, "H1_trimodal_AUD_MR_VID"); end

        %% MR-only -> AUD 
        H1_trimodal_AUD_MR = trimodalPCA(mrAndVideoData, audioFeatures, dataIdx, ...
               reconstructId=3, ...     % 3 = Audio
               nBoots=nBoots, ...
               VERBOSE=false, ...
               ifNormalise=true, ...
               targetAudioShare= targetAudioShare, ...      
               includeAudio= true, ...
               h1Source="MR");

        if ~isfile(h1_audioCSV), generateEmptyCSV(H1_trimodal_AUD_MR, h1_audioCSV, "H1_trimodal_AUD_MR"); end
        if writeToCsv && isfile(h1_audioCSV), appendToCSV(H1_trimodal_AUD_MR, h1_audioCSV, "H1_trimodal_AUD_MR"); end

        %% VID-only -> AUD
        H1_trimodal_AUD_VID = trimodalPCA(mrAndVideoData, audioFeatures, dataIdx, ...
                reconstructId=3, ...     % 3 = Audio
                nBoots=nBoots, ...
                VERBOSE=false, ...
                ifNormalise=true, ...
                targetAudioShare= targetAudioShare, ...      
                includeAudio= true, ...
                h1Source="VID");

        if ~isfile(h1_audioCSV), generateEmptyCSV(H1_trimodal_AUD_VID, h1_audioCSV, "H1_trimodal_AUD_VID"); end
        if writeToCsv && isfile(h1_audioCSV), appendToCSV(H1_trimodal_AUD_VID, h1_audioCSV, "H1_trimodal_AUD_VID"); end

        fprintf("H1 Audio Done\n");


    
    end


   
    if doH2_bimodal
        fprintf("Starting H2 bimodal\n");
    

        %% BIMODAL BASELINE TESTS (NO AUDIO)
        H2_bimodal_MR = trimodalPCA(mrAndVideoData, audioFeatures, dataIdx, ...
               reconstructId=1, ...     % 1 = MR
               nBoots=0, ...
               VERBOSE=false, ...
               ifNormalise=true, ...
               targetAudioShare= targetAudioShare, ...      
               includeAudio= false);    % without audio
        
        
        if ~isfile(h2_bimodalCSV), generateEmptyCSV(H2_bimodal_MR, h2_bimodalCSV, "H2_bimodal_MR"); end % Only generate if no CSV file is there
        if writeToCsv && isfile(h2_bimodalCSV); appendToCSV(H2_bimodal_MR,h2_bimodalCSV, "H2_bimodal_MR"); end   % Append these results for this sentence to the CSV 


        H2_bimodal_Vid = trimodalPCA(mrAndVideoData, audioFeatures, dataIdx, ...
               reconstructId=2, ...     % 2 = VID
               nBoots=0, ...
               VERBOSE=false, ...
               ifNormalise=true, ...
               targetAudioShare= targetAudioShare, ...      
               includeAudio= false);    % without audio
        
        
        if ~isfile(h2_bimodalCSV), generateEmptyCSV(H2_bimodal_Vid, h2_bimodalCSV, "H2_bimodal_Vid"); end % Only generate if no CSV file is there
        if writeToCsv && isfile(h2_bimodalCSV); appendToCSV(H2_bimodal_Vid,h2_bimodalCSV, "H2_bimodal_Vid"); end   % Append these results for this sentence to the CSV 
        
        fprintf("H2 Bimodal Done\n");
    end
        
    if doH2_trimodal
        
        fprintf("Starting H2 trimodal\n")
        %% TRIMODAL TESTS (WITH AUDIO)
        H2_trimodal_MR = trimodalPCA(mrAndVideoData, audioFeatures, dataIdx, ...
               reconstructId=1, ...     % 1 = MR
               nBoots=0, ...
               VERBOSE=false, ...
               ifNormalise=true, ...
               targetAudioShare= targetAudioShare, ...      
               includeAudio= true);     % with audio
        
        if ~isfile(h2_trimodalCSV), generateEmptyCSV(H2_trimodal_MR, h2_trimodalCSV, "H2_trimodal_MR"); end % Only generate if no CSV file is there
        if writeToCsv && isfile(h2_trimodalCSV); appendToCSV(H2_trimodal_MR,h2_trimodalCSV, "H2_trimodal_MR"); end   % Append these results for this sentence to the CSV 


        H2_trimodal_Vid = trimodalPCA(mrAndVideoData, audioFeatures, dataIdx, ...
               reconstructId=2, ...     % 2 = VID
               nBoots=0, ...
               VERBOSE=false, ...
               ifNormalise=true, ...
               targetAudioShare= targetAudioShare, ...      
               includeAudio= true);  % with audio
        
        if ~isfile(h2_trimodalCSV), generateEmptyCSV(H2_trimodal_Vid, h2_trimodalCSV, "H2_trimodal_Vid"); end % Only generate if no CSV file is there
        if writeToCsv && isfile(h2_trimodalCSV); appendToCSV(H2_trimodal_Vid,h2_trimodalCSV,"H2_trimodal_Vid"); end   % Append these results for this sentence to the CSV 

        fprintf("H2 Trimodal Done\n");
    end


    %% Original SCHOLES bimodal shuffling PCA to compare to bimodal baseline
    if doScholes_bimodal
        fprintf("Starting Scholes bimodal\n")
        
        scholes_MR = pcaAndShufflingExample(mrAndVideoData, dataIdx, ...
            reconstructId=1, ...    % 1 = MR
            nBoots=100);            % 100 Nboots just for this test
    

        if ~isfile(scholes_bimodalCSV), generateEmptyCSV(scholes_MR, scholes_bimodalCSV, "scholes_MR"); end % Only generate if no CSV file is there
        if writeToCsv && isfile(scholes_bimodalCSV); appendToCSV(scholes_MR,scholes_bimodalCSV, "scholes_MR"); end   % Append these results for this sentence to the CSV 


    
        scholes_vid = pcaAndShufflingExample(mrAndVideoData, dataIdx, ...
            reconstructId=2, ...    % 2 = VID
            nBoots=100);            % 100 Nboots just for this test

        if ~isfile(scholes_bimodalCSV), generateEmptyCSV(scholes_vid, scholes_bimodalCSV, "scholes_vid"); end % Only generate if no CSV file is there
        if writeToCsv && isfile(scholes_bimodalCSV); appendToCSV(scholes_vid,scholes_bimodalCSV, "scholes_vid"); end   % Append these results for this sentence to the CSV

        fprintf("Scholes Bimodal Done\n");

    end

    




end


function appendToCSV(resultStruct, csvPath, testName)

    % Inject/overwrite test_name into the struct 
    if nargin >= 3 && ~isempty(testName)
        resultStruct.test_name = string(testName);
    end

    % Read header (column order)
    opts = detectImportOptions(csvPath);
    headerVariables = opts.VariableNames;

    % Ensure 'test_name' exists in the file header and is first column
    if ~ismember('test_name', headerVariables)
        T = readtable(csvPath);
        T.test_name = strings(height(T),1);
        T = movevars(T, 'test_name', 'Before', 1);
        writetable(T, csvPath);               % rewrite once to add the column
        headerVariables = T.Properties.VariableNames;
    else
        % Move it to the front in our ordering (no file rewrite needed)
        headerVariables = [{'test_name'}, setdiff(headerVariables, {'test_name'}, 'stable')];
    end

    % Build one row in header order; flatten non-scalars
    row = cell(1, numel(headerVariables));
    for k = 1:numel(headerVariables)
        name = headerVariables{k};

        if isfield(resultStruct, name)
            v = resultStruct.(name);
        else
            v = [];  % missing field -> missing value
        end

        if (isnumeric(v) || islogical(v)) && isscalar(v)
            row{k} = v;
        elseif ischar(v) || (isstring(v) && isscalar(v))
            row{k} = string(v);
        elseif isempty(v)
            row{k} = missing;
        else
            row{k} = string(jsonencode(v));  % flatten arrays/structs/cells
        end
    end

    % Append the row
    Trow = cell2table(row, 'VariableNames', headerVariables);
    writetable(Trow, csvPath, 'WriteMode', 'append');
end



function generateEmptyCSV(resultStruct, csvPathIn, testName) %#ok<INUSD>
    fn = fieldnames(resultStruct).';
    % Prepend 'test_name' to the header (ignore the actual value here)
    varNames = ['test_name', fn];
    T = cell2table(cell(0, numel(varNames)), 'VariableNames', varNames);
    writetable(T, csvPathIn);
end








