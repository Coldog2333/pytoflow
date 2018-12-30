data_path = '/home/ftp/Coldog/Dataset/toflow/vimeo_septuplet/sequences';
noise_path = '/home/ftp/Coldog/Dataset/toflow/vimeo_septuplet/sequences_blur';
if ~exist(noise_path, 'dir')
  mkdir(noise_path);
end
videos = dir(data_path);
total_num = 91701;
count = 0;
disp('Start bluring...')
for i=1:length(videos)
    if strcmp(videos(i).name, '.') || strcmp(videos(i).name, '..')
        continue
    end
    seps = dir(fullfile(data_path, videos(i).name));
    for j=1:length(seps)
        if strcmp(seps(j).name, '.') || strcmp(seps(j).name, '..')
            continue
        end
        blur_input(fullfile(data_path, videos(i).name, seps(j).name), fullfile(noise_path, videos(i).name, seps(j).name))
        count = count + 1;
        if mod(count,500)==0
            disp(['  Processed ',num2str(count/total_num*100), '%'])
        end
    end
end
disp(['  Processed ',num2str(count/total_num*100), '%'])