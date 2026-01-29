% Program: Crop the individual animal face images from the collected video for on YOLO detection

% Aims:
% 1. Extracting frames of individual subjects from the raw input video by a defined time interval.
% 2. Saving extracted frames based on identity.

% The input video file (.mp4, .avi, etc.) contains face and identity data of one or multiple animal individuals.
% The output folder contains the extracted frames (.png, .jpg, etc.) of the animals, which may be differentiated by identities.
% Contributor: Jiayue Yang, 2025-01-14

% Read the video
dir = 'raw_animal_face_video'; % (to edit) collected face video directory
RawVid = VideoReader(dir);

% Frame rate (fps)
fps = 30; % (to edit) based on camera properties

% Calculate frame numbers for the time interval (in seconds)
start_frame = ceil(150 * fps); % (to edit) start time of the clip for frame extraction
end_frame = floor(194 * fps); % (to edit) end time of the clip for frame extraction

% Extract frames in the specified interval
interval = 6; % (to edit) an image is extracted every n frames between the start and end frame (n=interval)
for i = start_frame:interval:end_frame
    % Read the current frame
    frame = read(RawVid, i);
    
    % Save the frame as an image
    imwrite(frame, ['output_folder' int2str(i) '.jpg']); % (to edit) output folder of the extracted images, and the file format

end
