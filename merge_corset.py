import pickle
import pandas as pd
import sys
import atpc

# output_dir_name coreset_size_min coreset_size_max coreset_size_step time_min time_max uniform

if len(sys.argv) > 1:
    output_dir = sys.argv[1]
else:
    output_dir = 'output_data'

if len(sys.argv) > 2:
    coreset_size_min = int(sys.argv[2])
else:
    coreset_size_min = 100

if len(sys.argv) > 3:
    coreset_size_max = int(sys.argv[3])
else:
    coreset_size_max = 1000

if len(sys.argv) > 4:
    coreset_size_step = int(sys.argv[4])
else:
    coreset_size_step = 100

if len(sys.argv) > 5:
    time_min = int(sys.argv[5])
else:
    time_min = 0

if len(sys.argv) > 6:
    time_max = int(sys.argv[6])
else:
    time_max = 10

if len(sys.argv) > 7:
    uniform_sampling = bool(int(sys.argv[7]))
else:
    uniform_sampling = False

if len(sys.argv) > 8:
    init_file_name = sys.argv[8]
else:
    init_file_name = 'database_conf/example.json'

merge_corset = {}
for k in range(coreset_size_min, coreset_size_max + 1, coreset_size_step):
    merge_corset[k] = []
    for times in range(time_min, time_max + 1):
        f = open(output_dir + '/' + init_file_name.split('/')[-1].split('.')[0] + '_' + ('uniform' if uniform_sampling else 'coreset') + '_' + str(k) + '_' + str(times) + '.data', 'rb')
        merge_corset[k].append(pickle.load(f))

f = open(output_dir + '/' + init_file_name.split('/')[-1].split('.')[0] + '_' + ('uniform' if uniform_sampling else 'coreset') + '.data', 'wb')

# pickle.dump({'total_weight': total_weight, 'coreset': coreset}, f, -1)
pickle.dump(merge_corset, f, -1)

f.close()
