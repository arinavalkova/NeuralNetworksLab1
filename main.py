import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import chardet
import seaborn as sns

with open('content/ID_data_mass_18122012.csv', 'rb') as f:
    result = chardet.detect(f.read())
df = pd.read_csv('content/ID_data_mass_18122012.csv', sep=';', encoding=result['encoding'])
df = df.apply(lambda x: x.str.replace(',', '.'))

headers = df.values[0, 2:].tolist()
headers[headers.index('Pлин')] = 'Рлин'
d = {}
for i in range(len(headers)):
    if (d.get(headers[i]) != None):
        d[headers[i]] += 1
        headers[i] = headers[i] + '_' + str(d[headers[i]])
    else:
        d[headers[i]] = 1
df.drop([0, 1], axis=0, inplace=True)
df.drop([df.columns[0], df.columns[1]], axis=1, inplace=True)
df.columns = headers
df = df.reset_index(drop=True)
for i in df.columns:
    df[i] = pd.to_numeric(df[i], errors='coerce')
for i in range(len(df)):
    if ((pd.isnull(df['КГФ'][i])) & (pd.notnull(df['КГФ_2'][i]))):
        df['КГФ'][i] = df['КГФ_2'][i] * 1000
df.drop('КГФ_2', axis=1, inplace=True)
df.dropna(how="all", subset=['КГФ', 'G_total'], inplace=True)
df = df.reset_index(drop=True)

df.to_csv('content/withoutMess.csv', sep=";", encoding='cp1251')


def dist_properties(df):
    index = df.columns
    d = {}
    d['Кол-во'] = []
    d['% пропусков'] = []
    d['Минимум'] = []
    d['Максимум'] = []
    d['Среднее'] = []
    d['Мощность'] = []
    d['% уникальных'] = []
    d['Первый квартиль(0.25)'] = []
    d['Медиана'] = []
    d['Второй квартиль(0.75)'] = []
    d['Стандартное отклонение'] = []
    for h in df.columns:
        d['Кол-во'].append(df[h].count())
        d['% пропусков'].append(df[h].isna().sum() / len(df) * 100)
        d['Минимум'].append(df[h].min())
        d['Максимум'].append(df[h].max())
        d['Среднее'].append(df[h].mean())
        d['Мощность'].append(df[h].nunique())
        d['% уникальных'].append(df[h].nunique() / df[h].count() * 100)
        d['Первый квартиль(0.25)'].append(df[h].quantile(0.25))
        d['Медиана'].append(df[h].median())
        d['Второй квартиль(0.75)'].append(df[h].quantile(0.75))
        d['Стандартное отклонение'].append(df[h].std())
    return pd.DataFrame(d, index)


tab = dist_properties(df)

tab.to_csv('content/parametersTable.csv', sep=";", encoding='cp1251')

removed = []
cat_index = []
cont_index = []
for i in tab.index:
    if tab['% пропусков'][i] > 60 and i != 'G_total':
        removed.append(i)
        continue
    if tab['Мощность'][i] == 1:
        removed.append[i]
        continue
    if tab['Мощность'][i] < 25:
        cat_index.append(i)
    else:
        cont_index.append(i)
df.drop(removed, axis=1, inplace=True)

df.to_csv('content/without60.csv', sep=";", encoding='cp1251')


def cat_dist_properties(df):
    d = {}
    d['Кол-во'] = []
    d['% пропусков'] = []
    d['Мощность'] = []
    for j in (0, 1):
        d['Мода' + str(j + 1)] = []
        d['Частота моды' + str(j + 1)] = []
        d['% моды' + str(j + 1)] = []

    for i in cat_index:
        d['Кол-во'].append(df[i].count())
        d['% пропусков'].append(df[i].isna().sum() / len(df) * 100)
        d['Мощность'].append(df[i].nunique())
        vc = df[
            i].value_counts()  # сколько раз встречается каждое число из столбца и сортирует по убыванию встречаемости
        for j in (0, 1):
            m = vc.index[j]  # берется 2 самых встречаемых значения
            m_count = vc[m]  # сколько раз встретились 2 самых популярных значения
            m_p = m_count / d['Кол-во'][
                cat_index.index(i)] * 100  # процент таких двух самых популярных значений среди всех в столбце
            d['Мода' + str(j + 1)].append(m)
            d['Частота моды' + str(j + 1)].append(m_count)
            d['% моды' + str(j + 1)].append(m_p)
    return pd.DataFrame(d, cat_index)


tab2 = cat_dist_properties(df)

tab2.to_csv('content/catProperties.csv', sep=";", encoding='cp1251')

df.rename(columns={cat_index[i]: cat_index[i] + '_катериг' for i in range(len(cat_index))}).hist(bins=70,
                                                                                                 figsize=(20, 20),
                                                                                                 color='r')
plt.savefig('content/hist.png')
plt.close()

normal_dist = ['Руст', 'Рзаб', 'Рлин', 'Рлин_2', 'Дебит кон нестабильный']
for i in cont_index:
    if i in normal_dist:
        bot = tab['Среднее'][i] - 2 * tab['Стандартное отклонение'][i]
        top = tab['Среднее'][i] + 2 * tab['Стандартное отклонение'][i]
    else:
        x025 = tab['Первый квартиль(0.25)'][i]
        x075 = tab['Второй квартиль(0.75)'][i]
        bot = x025 - 1.5 * (x075 - x025)
        top = x075 + 1.5 * (x075 - x025)
    for j, row in df.iterrows():
        if df[i][j] < bot or df[i][j] > top:
            if i == 'КГФ':
                df.drop(index=j, inplace=True)
            else:
                df[i][j] = float('nan')
                if tab['% пропусков'][i] < 30:
                    df[i][j] = tab['Медиана'][i]

df = df.reset_index(drop=True)
df.hist(bins=70, figsize=(20, 20), color='r')

plt.savefig('content/hist2.png')
plt.close()

N = df.shape[0]
n = int(np.log2(N)) + 1

ct = pd.DataFrame(index=df.index, columns=df.columns)
for column in ct:
    min = df[column].min()
    max = df[column].max()
    step = (max - min) / n
    for i in range(N):
        if not np.isnan(df[column][i]):
            interval = int((df[column][i] - min) / step)
            if interval == n:
                interval -= 1
            ct[column][i] = interval
        else:
            ct[column][i] = -1

ct.astype('int32')
freq_T = np.zeros((n + 1, n), dtype=int)
for i in range(N):
    freq_T[ct['G_total'][i] + 1, ct['КГФ'][i]] += 1

info_T = 0  # Оценка среднего количества информации для определения класса примера из T
for i in range(n + 1):
    for j in range(n):
        ft = freq_T[i, j]
        if ft != 0:
            info_T -= (ft / N) * np.log2(ft / N)

gain_ratio = {}
for column in ct.columns:
    if column != 'КГФ' and column != 'G_total':
        info_x_T = 0
        split_info_x = 0
        for i in range(n):
            Ni = 0
            freq_x_T = np.zeros_like(freq_T)
            for j in range(N):
                x = ct[column][j]
                if x == i:
                    Ni += 1
                    freq_x_T[ct['G_total'][j] + 1, ct['КГФ'][j]] += 1
            info_Ti = 0
            if Ni != 0:
                for i in range(n + 1):
                    for j in range(n):
                        if freq_x_T[i, j] != 0:
                            info_Ti -= (freq_x_T[i, j] / Ni) * np.log2(freq_x_T[i, j] / Ni)
                info_x_T += (Ni / N) * info_Ti
                split_info_x -= (Ni / N) * np.log2((Ni / N))
        gain_ratio[column] = (info_T - info_x_T) / split_info_x

        vals = list(gain_ratio.values())
        length = len(vals)
        keys = list(gain_ratio.keys())
        fig, ax = plt.subplots(figsize=(7, 7))
        plt.barh(keys, vals, align='edge', color="green")
        for i in range(length):
            plt.annotate("%.2f" % vals[i], xy=(vals[i], keys[i]), va='center', color="red")

plt.savefig('content/hist3.png')

ax = plt.subplots(figsize=(30, 30))
heatmap = sns.heatmap(df.corr(), cmap='coolwarm', annot=True, vmin=-1, vmax=1, center=0)

plt.savefig('content/corrMatrix.png')

corr_matrix = df.corr().to_numpy()

dropped = []
for i in range(len(df.columns)):
    col1 = df.columns[i]
    if col1 != 'КГФ' and col1 != 'G_total':
        for j in range(i):
            col2 = df.columns[j]
            if col2 in dropped:
                continue
            if col2 != 'КГФ' and col2 != 'G_total':
                if corr_matrix[i, j] > 0.9:
                    drop_f = True
                    for k in range(len(df.columns)):
                        col3 = df.columns[k]
                        if col3 in dropped:
                            continue
                        if abs(corr_matrix[i, k] - corr_matrix[j, k]) > 0.25:
                            print(df.columns[i], df.columns[k], df.columns[j])
                            drop_f = False
                    if drop_f:
                        if gain_ratio[col1] > gain_ratio[col2]:
                            dropped.append(col2)
                        else:
                            dropped.append(col1)

df.drop(columns=dropped, inplace=True)
df.drop(['Рпл. Тек (послед точка на КВД)','Рпл. Тек (Расчет по КВД)'],axis=1, inplace=True)
df.to_csv('content/withCorel.csv', sep=";", encoding='cp1251')
