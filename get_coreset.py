import pickle
import pandas as pd
import sys
import atpc

if len(sys.argv) > 1:
    k = int(sys.argv[1])
else:
    k = 100

if len(sys.argv) > 2:
    times = int(sys.argv[2])
else:
    times = 0

if len(sys.argv) > 3:
    init_file_name = sys.argv[3]
else:
    init_file_name = 'database_conf/example.json'

if len(sys.argv) > 4:
    output_dir = sys.argv[4]
else:
    output_dir = 'output_data'

if len(sys.argv) > 5:
    uniform_sampling = bool(int(sys.argv[5]))
else:
    uniform_sampling = False

do_check = False
database = atpc.database(init_file_name)

f = open(output_dir + '/' + init_file_name.split('/')[-1].split('.')
         [0] + '_' + ('uniform' if uniform_sampling else 'coreset') + '_' + str(k) + '_' + str(times) + '.data', 'wb')

total_weight = database.getPointNum([], [], 0)

if uniform_sampling:
    coreset = database.sampleFromCenter([], [], 0, k)
else:
    coreset = database.getCoreSet(k, 0, False)

# pickle.dump({'total_weight': total_weight, 'coreset': coreset}, f, -1)
pickle.dump(coreset, f, -1)

f.close()
