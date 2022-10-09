import json
import warnings
from sqlalchemy import create_engine
import sqlalchemy
import pandas as pd
import numpy as np
import cupy as cp
import random
import multiprocessing
try:
    cp.array([])
except:
    cp = np


class database():
    class table():
        count_column: str = ""  # 用来统计的
        target_table: str = ""  # count column 统计后综合到哪张表
        count_table_name: str = ""
        sample_flag: bool = False

        def __init__(self, table_name: str, columns: 'list[str]') -> None:
            self.table_name: str = table_name
            self.columns: 'list[str]' = list(columns)
            self.center_column: 'list[str]' = list(columns)  # 用来约束的
            self.count_table_names: 'dict[tuple,str]' = {}
            self.weight: 'map[str,str]' = {}  # weight: (weight_table_name_from: column_name)
            self.connect_index = []  # 记录连接的column在connects中的下标
            return

        def resetCountTableNames(self, engine: sqlalchemy.engine.base.Engine) -> None:
            for center in self.count_table_names:
                engine.execute('drop table if exists ' + self.count_table_names[center] + ';')
            self.count_table_names = {}
            return

        def getTmpTableName(self, center) -> bool:
            if tuple(center) in self.count_table_names:
                self.count_table_name = self.count_table_names[tuple(center)]
                return True
            self.count_table_name = self.table_name + '_tmp_' + str(len(self.count_table_names))
            self.count_table_names[tuple(center)] = self.count_table_name
            return False

        def createInitSQL(self) -> str:
            select_str = "select "
            for i in self.center_column:
                select_str += '"{0}"."{1}" "{0}__{1}", '.format(self.table_name, i)
            if not self.center_column:
                select_str += '* '
            select_str = select_str[:-2]
            select_str += ' from "{0}"'.format(self.table_name)
            return select_str

        def createCountSQL(self, center: 'dict[str, float]', r_2: float, tables: 'dict[str,database.table]') -> str:
            if self.getTmpTableName(center.values()) and self.count_column != "":
                return ""
            if self.weight:
                tmp_str_from = self.table_name
                tmp_str_sum = ""
                tmp_str_where = ""
                for child_table_name in self.weight:
                    tmp_str_from += ', ' + tables[child_table_name].count_table_name
                    tmp_str_sum += tables[child_table_name].count_table_name + '.__weight * '
                    tmp_str_where += '"{0}"."{1}"="{2}"."{3}" and '.format(
                        self.table_name, self.weight[child_table_name], tables[child_table_name].count_table_name, tables[child_table_name].count_column)
                select_str = 'create temporary table {2} as (select "{5}"."{0}" "{0}", sum({3}) __weight from {1}  where {4}'.format(
                    self.count_column, tmp_str_from, self.count_table_name, tmp_str_sum[:-3], tmp_str_where, self.table_name)
            else:
                select_str = 'create temporary table {2} as (select "{0}" "{0}", count(*) __weight from {1} where '.format(
                    self.count_column, self.table_name, self.count_table_name)

            if center:
                for i in center:
                    select_str += 'power("{2}"."{0}"-({1}),2) + '.format(i, center[i], self.table_name)
                select_str = select_str[:-2]
                select_str += '< ' + str(r_2)
            elif not self.weight:
                select_str = select_str[:-7]
            else:
                select_str = select_str[:-5]

            if self.count_column != "":
                select_str += ' group by "{0}"."{1}");'.format(self.table_name, self.count_column)
            else:
                select_str = select_str[len(select_str.split('(')[0]) + 1:]
                select_str = select_str[:6] + select_str[len(select_str.split(',')[0]) + 1:]
                select_str += ';'
            return select_str

        def createSampleSQL(self, center: 'dict[str, float]', r_2: float, columns_count_use: 'list[str]', tables: 'dict[str,database.table]') -> str:
            sample_str: str = ""
            tmp_str_from = self.table_name
            tmp_str_weight = "1"
            tmp_str_where = ""
            if len(columns_count_use) > 0 and self.count_column != "":
                tmp_str_where += '"{0}"."{1}" in {2} and '.format(self.table_name,  self.count_column, tuple(
                    columns_count_use) if len(columns_count_use) != 1 else '(' + str(columns_count_use[0]) + ')')
            for child_table_name in self.weight:
                tmp_str_from += ', ' + tables[child_table_name].count_table_name
                tmp_str_weight += ' * ' + tables[child_table_name].count_table_name + '.__weight'
                tmp_str_where += '"{0}"."{1}"="{2}"."{3}" and '.format(
                    self.table_name, self.weight[child_table_name], tables[child_table_name].count_table_name, tables[child_table_name].count_column)
            tmp_str_column = '({0}) __weight'.format(tmp_str_weight)
            for i in self.columns:
                tmp_str_column += ', "{0}"."{1}" "{0}__{1}"'.format(self.table_name, i)

            sample_str += 'select {0} from {1} where {2}'.format(tmp_str_column, tmp_str_from, tmp_str_where)
            if center:
                for i in center:
                    sample_str += 'power("{2}"."{0}"-({1}),2) + '.format(i, center[i], self.table_name)
                sample_str = sample_str[:-2]
                sample_str += '< ' + str(r_2)
            elif not self.weight and len(columns_count_use) == 0:
                sample_str = sample_str[:-7]
            else:
                sample_str = sample_str[:-5]
            sample_str += ';'
            return sample_str

        def sampleN(self, centers: 'dict[dict[str, float]]', r_2: float, n: 'pd.Series|int', tables: 'dict[str,database.table]', engine) -> pd.DataFrame:
            self.sample_flag = True
            center = centers[self.table_name]
            if type(n) == int:
                ans_df = pd.DataFrame(index=range(n))
                weight_sql_str = tables[self.table_name].createSampleSQL(center, r_2, [], tables)
                df_sample = pd.read_sql_query(weight_sql_str, engine)  # 返回表头：weight
                df_sample = df_sample.sample(
                    n=n, replace=True, weights=df_sample['__weight']).drop(columns=['__weight'])
                ans_df = pd.concat([ans_df, df_sample.reset_index().drop(columns=['index'])], axis=1)
            else:
                count_n = n.value_counts()
                weight_sql_str = self.createSampleSQL(center, r_2, count_n.index, tables)
                df_sample = pd.read_sql_query(weight_sql_str, engine)
                ans_df = pd.DataFrame(columns=df_sample.columns, index=range(len(n)))
                for i in count_n.index:
                    tmp_df = df_sample[df_sample[self.table_name + '__' + self.count_column] == i]
                    ans_df.values[n == i] = tmp_df.sample(count_n[i], replace=True, weights=tmp_df['__weight']).values
                ans_df = ans_df.drop(columns=['__weight'])
                if self.count_column not in self.center_column:
                    ans_df = ans_df.drop(columns=[self.table_name + '__' + self.count_column])
            for child_table_name in self.weight:
                column_name = self.weight[child_table_name]
                ans_df = pd.concat([ans_df, tables[child_table_name].sampleN(
                    centers, r_2, ans_df[self.table_name + '__' + column_name], tables, engine)], axis=1)
            for child_table_name in self.weight:
                column_name = self.weight[child_table_name]
                if tables[child_table_name].count_column in tables[child_table_name].center_column:
                    ans_df = ans_df.drop(columns=[self.table_name + '__' + column_name])
            return ans_df

    multi_tmp = {}

    def __init__(self, file_name: str = None, conf: 'dict' = None) -> None:
        if not conf:
            f = open(file_name, 'r')
            self.conf = json.load(f)
        else:
            self.conf = conf
        try:
            self.engine = create_engine('postgresql+psycopg2://' + self.conf['user_name'] + ':' + self.conf['password'] +
                                        '@' + self.conf['ip'] + ':' + str(self.conf['port']) + '/' + self.conf['database'])
            if 'temp_buffers' in self.conf:
                self.engine.execute('set temp_buffers to "' + self.conf['temp_buffers'] + '";')
            if 'work_mem' in self.conf:
                self.engine.execute('set work_mem to "' + self.conf['work_mem'] + '";')
        except:
            warnings.warn('There are something wrong!')
        if 'cpu_num' in self.conf:
            self.cpu_num = self.conf['cpu_num']
        else:
            self.cpu_num = int(multiprocessing.cpu_count() * 0.6)
        self.center_history: 'dict[str,dict[str,float]]' = {}
        self.tables: 'dict[str,database.table]' = {}
        self.connects: 'list[list[list[str]]]' = []
        self.table_sort: 'list[str]' = []
        if 'tables' in self.conf:
            for table_name in self.conf['tables']:
                self.addTable(table_name, self.conf['tables'][table_name])
        if 'connects' in self.conf:
            for connect in self.conf['connects']:
                self.addConnect(connect[0].copy(), connect[1])
        if 'target' in self.conf:
            self.target = tuple(self.conf['target'])
        else:
            self.target = None
        self.sortTables()
        return

    def __getstate__(self):
        # multiprocessing 执行时会对self进行序列化，engine不能被dump
        # state = {'init_func': self.__dict__['init_func']}
        return self.conf

    def __setstate__(self, state: dict):
        # 重写反序列化，multi_tmp 存下每个进程的中间状态，第一次由初始化函数生成
        if not self.multi_tmp:
            # self.__dict__.update(state['init_func']().__dict__)
            self.__init__(conf=state)
        else:
            self.__dict__.update(self.multi_tmp)

    def addTable(self, table_name: str, columns: 'list[str]'):
        new_table = self.table(table_name, columns)
        self.tables[table_name] = new_table
        self.center_history[table_name] = {'reset': {}}
        return

    def addConnect(self, connect: 'list[list[str]]', center_by_table: str) -> None:
        if self.table_sort:
            warnings.warn("Don't add connect after sort table!")
            return
        if len(connect) <= 1:
            warnings.warn("Connect must have 2 *** at least!")
            return
        for i in connect:
            table_name, column, index = i[0], i[1], len(self.connects)
            self.tables[table_name].connect_index.append(index)
            if table_name != center_by_table:
                self.tables[table_name].center_column.remove(column)
        self.connects.append(connect)
        return

    def sortTables(self) -> None:
        # 计算表间动态规划顺序
        tmp_connects = [[] for _ in range(len(self.connects))]
        flag = 1
        while flag == 1:
            flag, tmp_table = 0, []
            for table_name in self.tables:
                if flag == 0 and len(self.tables[table_name].connect_index) > 0:
                    flag = 2
                if len(self.tables[table_name].connect_index) == 1:
                    flag = 1
                    index = self.tables[table_name].connect_index[0]
                    for connect in self.connects[index]:
                        if connect[0] == table_name:
                            self.connects[index].remove(connect)
                            tmp_connects[index].append(connect)
                            break
                    self.tables[table_name].connect_index.remove(index)
                    self.table_sort.append(table_name)
                    if len(self.connects[index]) == 1:
                        father_t_name, father_column = self.connects[index][0][0], self.connects[index][0][1]
                        for connect in tmp_connects[index]:
                            t_name, column = connect[0], connect[1]
                            self.tables[father_t_name].weight[t_name] = father_column
                            self.tables[t_name].target_table = father_t_name
                            self.tables[t_name].count_column = column
                        tmp_table.append((father_t_name, index))
            for t_name, index in tmp_table:
                try:
                    self.tables[t_name].connect_index.remove(index)
                except:
                    pass
        if flag == 2:
            warnings.warn('Connects hava circle!')
            return
        for table_name in self.tables:
            if table_name not in self.table_sort:
                self.table_sort.append(table_name)
        return

    def getPointNum(self, center_column: 'list[str]', center: 'list[float]', r_2: float) -> int:
        ans = 1
        tidy_center = {table_name: {} for table_name in self.tables}
        for i in range(len(center_column)):
            t_name, col_name = center_column[i].split('__')
            tidy_center[t_name][col_name] = center[i]
        for table_name in self.table_sort:
            if tidy_center[table_name] == self.center_history[table_name] and self.tables[table_name].count_column != "":
                continue
            else:
                self.center_history[table_name] = tidy_center[table_name]
                father_table = self.tables[table_name].target_table
                # self.tables[table_name].resetCountTableNames(self.engine)
                if father_table:
                    self.center_history[father_table] = {'reset': {}}  # 仅更改父表flag
                    self.tables[father_table].resetCountTableNames(self.engine)
            sql_str = self.tables[table_name].createCountSQL(tidy_center[table_name], r_2, self.tables)
            if not sql_str:
                continue
            if self.tables[table_name].count_column != "":
                self.engine.execute(sql_str)
            else:
                df_count = pd.read_sql_query(sql_str, self.engine)
                if df_count["__weight"][0] is None:
                    return 0
                ans *= df_count["__weight"][0]
        return int(ans)

    def createCheckSQL(self, centers: 'dict[str,dict[str,float]]', r_2: float) -> str:
        tmp_str_from = ''
        tmp_str_where = ''
        for table_name in self.tables:
            tmp_str_from += table_name + ', '
            for child_table_name in self.tables[table_name].weight:
                tmp_str_where += '"{0}"."{1}"="{2}"."{3}" and '.format(
                    table_name, self.tables[table_name].weight[child_table_name], self.tables[child_table_name].table_name, self.tables[child_table_name].count_column)
            center = centers[table_name] if table_name in centers else {}
            if center:
                for i in center:
                    tmp_str_where += 'power("{2}"."{0}"-({1}),2) + '.format(i, center[i], table_name)
                tmp_str_where = tmp_str_where[:-2]
                tmp_str_where += '< ' + str(r_2) + ' and '

        check_str: str = "select 1 __have from {0} where {1}".format(tmp_str_from[:-2], tmp_str_where)
        if tmp_str_where == '':
            check_str = check_str[:-7]
        else:
            check_str = check_str[:-5]
        check_str += ' limit 1;'
        return check_str

    def sampleFromCenter(self, center_column: 'list[str]', center: 'list[float]', r_2: float, n: int, flag: bool = True) -> pd.DataFrame:
        # 先做getPointNum生成权重矩阵
        if not flag:
            self.getPointNum(center_column, center, r_2)
        for table_name in self.tables:
            self.tables[table_name].sample_flag = False
        tidy_center = {table_name: {} for table_name in self.tables}
        for i in range(len(center_column)):
            t_name, col_name = center_column[i].split('__')
            tidy_center[t_name][col_name] = center[i]
        ans_df = pd.DataFrame(index=range(n))
        for table_name in self.table_sort[::-1]:
            if not self.tables[table_name].sample_flag:
                ans_df = pd.concat([ans_df, self.tables[table_name].sampleN(
                    tidy_center, r_2, n, self.tables, self.engine)], axis=1)
        return ans_df

    class pseudoCube():
        r_2: int = 0
        size: int = 0
        columns: 'list[str]'
        centers: np.ndarray
        distance: np.ndarray

        def __init__(self, k: int, columns: list) -> None:
            self.size = 0
            self.columns = columns
            self.distance = np.zeros((k, k))
            self.centers = np.zeros((k, len(columns)))

        def addCenter(self, point: np.ndarray, distance2center: np.ndarray) -> None:
            if self.size >= self.centers.shape[0]:
                # error
                return
            try:
                self.centers[self.size, :] = cp.asnumpy(point)
            except:
                self.centers[self.size, :] = point
            if(len(distance2center)) == self.size:
                self.distance[:self.size, self.size] = distance2center
                self.distance[self.size, :self.size] = distance2center
            self.size += 1
            return

        def getDistance(self) -> None:
            for i in range(self.centers.shape[0]):
                self.distance[i, i] = 0
                for j in range(i):
                    self.distance[i, j] = self.distance[j, i] = np.sum((self.centers[i] - self.centers[j])**2)
            return

        def coverPoint(self, point: pd.DataFrame, center_index: np.ndarray) -> bool:
            def inBalls(centers: np.ndarray, points: np.ndarray, r_2):
                return (np.sum((np.expand_dims(centers, axis=0).repeat(points.shape[0], axis=0)-np.expand_dims(points, axis=1).repeat(centers.shape[0], axis=1))**2, axis=2)-r_2) <= 0
            centers_use = pd.DataFrame(self.centers[center_index], columns=self.columns)
            table_names = set(map(lambda x: x.split('__')[0], self.columns))
            table_column = {table_name: [column for column in self.columns if column.split(
                '__')[0] == table_name] for table_name in table_names}
            ans = np.full((point.shape[0], len(center_index)), True)
            for table_name in table_names:
                ans = ans & inBalls(centers_use[table_column[table_name]].values,
                                    point[table_column[table_name]].values, self.r_2)
            return np.sum(ans, axis=1) > 0

        def sortCube(self, sort: 'list[str]') -> np.ndarray:
            # 内部排序使得count操作时尽可能多的复用中间结果
            sort_dict = {table_name: -1 for table_name in sort}
            for i in range(len(self.columns)):
                table_name = self.columns[i].split('__')[0]
                if table_name in sort_dict and sort_dict[table_name] == -1:
                    sort_dict[table_name] = i
            index = np.array(range(self.centers.shape[0]))
            for table_name in sort[::-1]:
                if sort_dict[table_name] != -1:
                    tmp_index = np.argsort(self.centers[:, sort_dict[table_name]])
                    index = index[tmp_index]
                    self.centers = self.centers[tmp_index]
            self.distance = self.distance[index][:, index]
            return index

    def check_new(self, column: 'list[str]', center: 'list[float]', r_2: float) -> bool:
        tidy_center = {table_name: {} for table_name in self.tables}
        for i in range(len(column)):
            t_name, col_name = column[i].split('__')
            tidy_center[t_name][col_name] = center[i]
        sql_str = self.createCheckSQL(tidy_center, r_2)
        df_count = pd.read_sql_query(sql_str, self.engine)
        if len(df_count["__have"]) == 0:
            return False
        return True

    def getInitPoints(self, table_name: str):
        sql_str = self.tables[table_name].createInitSQL()
        df_data = pd.read_sql_query(sql_str, self.engine)
        return df_data

    def getInitCube(self, table_name: str, k: int, ignore: int = 0) -> pseudoCube:
        # ignore: 贪心算法前几个点可能是离群点，忽略掉
        sql_str = self.tables[table_name].createInitSQL()
        df_data = pd.read_sql_query(sql_str, self.engine)
        data = cp.array(df_data.values)
        data_size = data.shape[0]
        if data_size < k + ignore:
            k = data_size - ignore
        ans = self.pseudoCube(k, list(df_data.columns))
        del df_data
        max_index = random.randint(0, data_size - 1)
        ans.addCenter(data[max_index], [])
        tmp_distance = cp.sum((data - cp.expand_dims(data[max_index], axis=0).repeat(data_size, axis=0))**2, axis=1)
        distance_to_center = tmp_distance
        k -= 1
        while ignore > 0:
            max_index = distance_to_center.argmax()
            tmp_distance = cp.sum((data - cp.expand_dims(data[max_index], axis=0).repeat(data_size, axis=0))**2, axis=1)
            distance_to_center = cp.minimum(tmp_distance, distance_to_center)
            ignore -= 1
        while k > 0:
            max_index = distance_to_center.argmax()
            ans.addCenter(data[max_index], [])
            tmp_distance = cp.sum((data - cp.expand_dims(data[max_index], axis=0).repeat(data_size, axis=0))**2, axis=1)
            distance_to_center = cp.minimum(tmp_distance, distance_to_center)
            k -= 1
        ans.r_2 = distance_to_center.max()
        ans.getDistance()
        ans.sortCube(self.table_sort)
        return ans

    def mergeCube(self, c1: pseudoCube, c2: pseudoCube, k: int, ignore: int = 0, do_check: bool = True) -> pseudoCube:
        if c1.size == 0 or c2.size == 0 or k <= 0:
            return
        ans_columns = c1.columns + c2.columns
        point_num = c1.size * c2.size
        distance_to_center: np.ndarray = np.full((c1.size, c2.size), np.finfo(np.float32).max)
        distance_history = []
        r_2 = max(c1.r_2, c2.r_2)
        if do_check:
            for i in range(c1.size):
                for j in range(c2.size):
                    if not self.check_new(ans_columns, np.concatenate((c1.centers[i], c2.centers[j])), r_2):
                        # todo: 确认r
                        point_num -= 1
                        distance_to_center[i, j] = 0
        if k > point_num + ignore:
            k = point_num - ignore
        ans = self.pseudoCube(k, ans_columns)
        max_c1_index, max_c2_index = random.randint(0, c1.size - 1), random.randint(0, c2.size - 1)
        while distance_to_center[max_c1_index, max_c2_index] == 0:
            max_c1_index, max_c2_index = random.randint(0, c1.size - 1), random.randint(0, c2.size - 1)
        ans.addCenter(np.concatenate((c1.centers[max_c1_index], c2.centers[max_c2_index])), np.array(
            [distance_history[i][max_c1_index, max_c2_index] for i in range(len(distance_history))]))
        tmp_distance = np.expand_dims(c1.distance[:, max_c1_index], 1).repeat(
            c2.size, axis=1) + np.expand_dims(c2.distance[max_c2_index, :], 0).repeat(c1.size, axis=0)
        distance_history.append(tmp_distance)
        distance_to_center = np.minimum(tmp_distance, distance_to_center)
        k -= 1
        while ignore > 0:
            tmp_distance = np.expand_dims(c1.distance[:, max_c1_index], 1).repeat(
                c2.size, axis=1) + np.expand_dims(c2.distance[max_c2_index, :], 0).repeat(c1.size, axis=0)
            distance_history.append(tmp_distance)
            distance_to_center = np.minimum(tmp_distance, distance_to_center)
            ignore -= 1
        while k > 0:
            max_c1_index, max_c2_index = np.unravel_index(distance_to_center.argmax(), distance_to_center.shape)
            ans.addCenter(np.concatenate((c1.centers[max_c1_index], c2.centers[max_c2_index])), np.array(
                [distance_history[i][max_c1_index, max_c2_index] for i in range(len(distance_history))]))
            tmp_distance = np.expand_dims(c1.distance[:, max_c1_index], 1).repeat(
                c2.size, axis=1) + np.expand_dims(c2.distance[max_c2_index, :], 0).repeat(c1.size, axis=0)
            distance_history.append(tmp_distance)
            distance_to_center = np.minimum(tmp_distance, distance_to_center)
            k -= 1
        ans.r_2 = distance_to_center.max()
        # todo: 按算法扩大r
        ans.sortCube(self.table_sort)
        return ans

    def mutiGetPointNum(self, center_column: 'list[str]', center: 'list[float]', r_2: float) -> int:
        point_num = self.getPointNum(center_column, center, r_2)
        self.multi_tmp.update(self.__dict__)
        return point_num

    def getTarget(self, c: pd.DataFrame) -> list[float]:
        if not self.target:
            return [0 for _ in range(c.shape[0])]
        table_name, target_column = self.target
        target = []
        select_col = {}
        for column_name in self.tables[table_name].center_column:
            select_col[table_name + '__' + column_name] = column_name
        # if self.tables[table_name].target_table:
        #     select_col[self.tables[table_name].target_table + '__' +
        #                self.tables[self.tables[table_name].target_table].weight[table_name]] = self.tables[table_name].count_column
        base_query = "select {1} from {0} where ".format(table_name, target_column)
        for _, row in c.iterrows():
            append_query = ''
            for j in select_col:
                append_query += "{0}.{1}={2} and ".format(table_name, select_col[j], str(row[j]))
            append_query = append_query[:-5] + ";"
            query = base_query + append_query
            target.append(pd.read_sql_query(query, self.engine).values[0])
        return target

    def getCoreSet(self, k: int, ignore: int = 0, do_check: bool = False) -> pd.DataFrame:
        if len(self.table_sort) != len(self.tables):
            warnings.warn("Please do table sort first!")
            return
        tmp_Cube_queue: 'list[database.pseudoCube]' = []
        for table_name in self.tables:
            tmp_Cube_queue.append(self.getInitCube(table_name, k, ignore))
        while len(tmp_Cube_queue) > 1:
            tmp_Cube_queue.append(self.mergeCube(tmp_Cube_queue[0], tmp_Cube_queue[1], k, ignore, do_check))
            tmp_Cube_queue = tmp_Cube_queue[2:]
        pool = multiprocessing.Pool(self.cpu_num)
        final_Cube = tmp_Cube_queue[-1]
        weight = pool.starmap_async(self.mutiGetPointNum, [(final_Cube.columns, final_Cube.centers[i], final_Cube.r_2) for i in range(
            final_Cube.centers.shape[0])], final_Cube.size // self.cpu_num if final_Cube.size // self.cpu_num != 0 else 1).get()
        pool.close()
        pool.join()
        coreset = pd.DataFrame(columns=final_Cube.columns, data=final_Cube.centers)
        coreset['weight'] = weight
        if self.target:
            target = self.getTarget(coreset)
            coreset['target'] = target
        return coreset
