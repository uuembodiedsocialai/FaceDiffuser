import maya.cmds as cm
import maya.mel as mel
import csv
import re
import os


# facial controls
def get_facial_controls():
    items = []
    fileWrite = open("RayControls.txt", 'r')
    ctl = fileWrite.readline()
    while ctl != '#EOF':
        items.append(ctl[:-1].split(':')[1])
        ctl = fileWrite.readline()
    fileWrite.close()
    return items


# facial controls transformations
def get_facial_controls_transformations():
    items = []
    fileRead = open("RayControls2.txt", 'r')

    for i in range(88):
        trans = fileRead.readline()
        items.append(trans[:-1])
    print(len(items))
    fileRead.close()

    return items


# select facial controls
def select_facial_controls():
    items = get_facial_controls()
    for item in items:
        cm.select(item, add=True)


# write control values to a csv file for multiple frames
""" 
start = first frame to write
end = last frame to write
filename = name of csv file
"""
def write_ctls_values_to_file(start, end, filename):
    ctls = get_facial_controls()
    trans = get_facial_controls_transformations()

    outfile = open('path/to/output/folder/' + str(filename), 'w')
    csvwriter = csv.writer(outfile)

    for frame in range(start, end):
        values = []
        for i in range(len(ctls)):
            t = trans[i].split()
            for j in range(len(t)):
                tx = cm.getAttr('{}.{}'.format(ctls[i], t[j]), time=frame)
                values.append(tx)
        frame += 1
        csvwriter.writerow(values)

    eof = ['#EOF']
    csvwriter.writerow(eof)
    outfile.close()


def sort_key(file):
    match = re.search(r"([0-9-]+/Session[0-9]+/Take[0-9]+)/head_[0-9]+/head_[0-9]+_([0-9]+)-([0-9]+)-v2\.csv$",
                      file, re.IGNORECASE)
    number = match.group(2)
    return int(number)


def get_data_files(path, data_type, actor):
    files = []
    for root, dirs, file in os.walk(path):
        for name in file:
            filepath = root + os.sep + name
            acting = 'head_' + str(actor)
            if filepath.endswith(data_type) and acting in filepath and \
                    '60fps' not in filepath and '-v2' in filepath:
                files.append(root + '/' + name)
    files.sort(key=sort_key)
    return files


def line_to_array(line):
    result = line.split(',')
    for i in range(len(result)):
        result[i] = float(result[i])
    return result


mouth_mask = [81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 118,
              119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134,
              135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 154, 155, 156, 157, 158, 159]
upper_face_mask = [0, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28,
                   29, 30, 31, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55,
                   56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80,
                   99, 100,
                   101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 147, 148, 149,
                   150,
                   151, 152, 153]
eyegaze_mask = [1, 2, 32, 33]
reduced_mask = [20, 21, 22, 24, 26, 30, 51, 52, 53, 54, 55, 57, 59, 61, 67, 76, 77, 79, 80, 81, 83, 111,
                112, 118, 119, 125, 128, 129, 131, 132, 133, 135, 163, 164, 166, 167, 170, 177, 184]

# Read animation from a take in the dataset
"""
path = path to the take
start_frame = the first frame to read of the take
length = the number of frames to read
mask = {
 None: read the whole face
 mouth_mask: read only the mouth
 upper_face_mask: read only the upper face
 eyegaze_mask: read only the eye gaze
}
"""


def read_real_animation(path, start_frame, length, mask):
    ctls = get_facial_controls()
    trans = get_facial_controls_transformations()

    ctls_trans = []
    for i in range(len(ctls)):
        t = trans[i].split()
        for j in range(len(t)):
            ctls_trans.append(str(ctls[i]) + '.' + str(t[j]))
    print(len(ctls_trans))

    # get the csv files for actor 1 in the take (in order)
    files = get_data_files(path, '.csv', 1)
    print(files)
    main_file = files.pop(0)
    reading = open(main_file, 'r')
    # get to the start frame
    for i in range(start_frame):
        line = reading.readline()
        if '#EOF' in line and len(files) > 0:
            print('EOF and go to next file: ', line)
            main_file = files.pop(0)
            reading.close()
            reading = open(main_file, 'r')
            line = reading.readline()
        if '#EOF' in line or line == '':
            print('EOF and none: ', line)
            return None

    # get the next frames of the given length
    for i in range(length):
        line = reading.readline()
        if '#EOF' in line and len(files) > 0:
            print('EOF and go to next file: ', line)
            main_file = files.pop(0)
            reading.close()
            reading = open(main_file, 'r')
            line = reading.readline()
        if '#EOF' in line or line == '':
            print('EOF and none: ', line)
            return None
        line = line_to_array(line)
        for j in range(len(line)):
            number = line[j]
            if mask is None:
                cm.setAttr(ctls_trans[j], float(number))
                cm.setKeyframe(ctls_trans[j], t=i)
            else:
                if j in mask:
                    cm.setAttr(ctls_trans[j], float(number))
                    cm.setKeyframe(ctls_trans[j], t=i)
    reading.close()


# read control values for multiple frames
""" 
filename = name of the file to be read
amount = the number of frames to read
"""
def read_ctls_animation(filename, amount=0):
    ctls = get_facial_controls()
    trans = get_facial_controls_transformations()

    ctls_trans = []
    for i in range(len(ctls)):
        t = trans[i].split()
        for j in range(len(t)):
            ctls_trans.append(str(ctls[i]) + '.' + str(t[j]))
    print(len(ctls_trans))

    fileRead = open(str(filename), 'r')

    csv_reader = csv.reader(fileRead)
    i = 0
    write = True
    for row in csv_reader:
        i += 1
        for j in range(len(row)):
            number = row[j]
            cm.setAttr(ctls_trans[j], float(number))
            cm.setKeyframe(ctls_trans[j], t=i)
            # write = True
        if amount > 0 and i > amount:
            break
    return i
    fileRead.close()


def render_seq(startframe=1, endframe=10, renderfn=cm.render, renderfn_args=None):
    '''render out a sequence of frames as per global settings

    defaults to using maya.cmds.render for frames 1-10'''

    # save state
    now = cm.currentTime(q=True)
    for x in range(startframe, endframe):
        cm.currentTime(x)
        renderfn(cm.ls(type='camera')[0], renderfn_args)

    # restore state
    cm.currentTime(now)


path_to_folder = ''

to_render = ['test']  # sequences to render
for sequence in to_render:
    # read generated animation
    num_frames = read_ctls_animation(f"{path_to_folder}/{sequence}_Damm.csv")
    cm.setAttr('defaultRenderGlobals.imageFilePrefix', sequence, type='string')
    print('Rendering Sequence: ', sequence, num_frames)
    render_seq(1, num_frames)

print('read generated')
# read real data
# read_real_animation('path/to/take/in/dataset/', 48136, 361, eyegaze_mask)
