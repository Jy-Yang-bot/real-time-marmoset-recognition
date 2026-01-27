function output = whoTouch_read_result()
    % read whoTouch_read_result - read YOLOv8 real-time detection results
    % Contributor: Jiayue Yang, 2025-03-10
    % input:
    %   detection result JSON file
    % output: 
    %   one cell that contains {name, time}
    %            name: most common name
    %            time: current system time

    % persist change，storing the most recent detected results (labels)
    persistent recent_labels
    if isempty(recent_labels)
        recent_labels = {};  % initialize empty lists
    end

    % check if JSON exists
    if isfile('true_animal_detection_results.json')  % (to edit) customize JSON file path
        % read JSON file
        data = fileread('true_animal_detection_results.json');
        result = jsondecode(data);
        
        % retrieve weighted_detections data list
        weighted_detections = result;
        
        % put weighted_detections content into recent_labels list
        recent_labels_add = [recent_labels, weighted_detections];
        
        % if list over n, remove the oldest results (n is the most recent detected frame numbers --> to edit based on own data)
        if numel(recent_labels) > 45
            recent_labels_add(1:end-45) = [];  % keep the last n detection elements
        end
        
        % calculate the most common labels from the most recent n detection elements
        if ~isempty(recent_labels_add)
            counts = tabulate(recent_labels_add);
            [~, idx] = max(cell2mat(counts(:,2)));
            most_common_label = counts{idx,1}; % output the most commonly detected label from the n recent frames
        else
            most_common_label = 'no detection'; % if nothing being detected
        end
        
        % retrieve current system time
        current_time = datetime('now', 'Format', 'yyyy-MM-dd HH:mm:ss');

        % output results
        output = {most_common_label, char(current_time)};
        disp(['Most common label in lastest frames: ', most_common_label]);
        disp(['Time: ', char(current_time)]);
    else
        % if file not exists, output empty
        output = {[], []};
        disp('Waiting for detection results...');
    end

end
