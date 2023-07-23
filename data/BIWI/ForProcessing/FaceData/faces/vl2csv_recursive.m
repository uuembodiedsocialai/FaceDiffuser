rootdir = pwd;
filelist = dir(fullfile(rootdir, '**\*.*'));  %get list of files and folders in any subfolder
filelist = filelist(~[filelist.isdir]); %remove folders from list

% display(filelist(55).folder);
% path = filelist(55).folder;
% fname = filelist(55).name;
% outfname = '\' + string(fname(1:end-2)) + 'csv';
% outfilepath = path+outfname;
tic
for i=1:size(filelist,1)
    currfile = filelist(i);
    path = currfile.folder;
    fname = currfile.name;
    [~, ~, ext] = fileparts(fname);
    if isequal(ext, '.vl')
        outfname = '\' + string(fname(1:end-2)) + 'csv';
        outfilepath = path+outfname;
        disp(outfilepath);
        infilepath = path+"\"+fname;
        fid = fopen(infilepath); % load the .vl file
        n_vertices = fread(fid, 1, 'ulong'); 
        vertices = fread(fid, [3, n_vertices] , 'float');
        disp("Loaded the .vl file! No. of Vs: " + n_vertices);

        fclose(fid);
        header = {'x', 'y', 'z'};
        textHeader = strjoin(header, ',');
        %write header to file
        fid2 = fopen(outfilepath,'w'); 
        fprintf(fid2,'%s\n', textHeader);
        fclose(fid2);
        dlmwrite(outfilepath, vertices.', '-append');

    end
end
toc