from itertools import combinations
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import LabelEncoder


df = pd.read_csv('game.csv')

print("Первые 5 строк датасета:")
print(df.head())
print("\nОбщая информация о данных:")
df.info()

rows, cols = df.shape
print(f"\nКоличество строк: {rows}")
print(f"Количество столбцов: {cols}")

# Разделяем столбцы на числовые и категориальные
num_cols = df.select_dtypes(include='number').columns
cat_cols = df.select_dtypes(exclude='number').columns

# Анализ числовых переменных
num_stats = pd.DataFrame({
    'Доля пропусков': df[num_cols].isna().mean(),
    'Минимум': df[num_cols].min(),
    'Максимум': df[num_cols].max(),
    'Среднее': df[num_cols].mean(),
    'Медиана': df[num_cols].median(),
    'Дисперсия': df[num_cols].var(),
    'Квантиль 0.1': df[num_cols].quantile(0.1),
    'Квантиль 0.9': df[num_cols].quantile(0.9),
    'Квартиль 1 (0.25)': df[num_cols].quantile(0.25),
    'Квартиль 3 (0.75)': df[num_cols].quantile(0.75),
})

print("\n\n==== Статистика по числовым переменным ====")
print(num_stats.round(4).to_string())

# Анализ категориальных переменных
cat_stats_list = []
for col in cat_cols:
    modes = df[col].mode().tolist()
    mode_str = ', '.join(map(str, modes[:3])) if modes else None

    cat_stats_list.append({
        'Переменная': col,
        'Доля пропусков': df[col].isna().mean(),
        'Уникальных значений': df[col].nunique(dropna=True),
        'Мода': mode_str
    })

cat_stats = pd.DataFrame(cat_stats_list).set_index('Переменная')

print("\n\n==== Статистика по категориальным переменным ====")
print(cat_stats.to_string())


# Формулировка и проверка статистических гипотез
alpha = 0.05

print("\n\n==== Информация о переменной season_type ====")
season_counts = df['season_type'].value_counts(dropna=False)
print("Частоты категорий:")
print(season_counts)

# берем только категории с достаточным числом наблюдений (>= 30)
season_types = season_counts[season_counts >= 30].index.tolist()
print(f"\nВыбранные категории: {season_types}")

print("\n==== гипотеза 1: Влияние типа сезона (season_type) на очки домашней команды (pts_home) ====")

# Формируем выборки динамически на основе найденных категорий
groups = [df.loc[df['season_type'] == s, 'pts_home'].dropna() for s in season_types]

# Проверка предпосылок (для N > 5000 используем тест пирсона вместо шапиро)
print("\nОписательные статистики и тест нормальности по группам:")
for s, g in zip(season_types, groups):
    stat_nt, p_nt = stats.normaltest(g)
    print(f"  {s:16s} (n={len(g):5d}): stat = {stat_nt:.4f}, p = {p_nt:.6e}, медиана = {g.median():.2f}")

stat_lev, p_lev = stats.levene(*groups)
print(f"\nТест Левена (равенство дисперсий): F = {stat_lev:.4f}, p-value = {p_lev:.6e}")

stat_kw, p_kw = stats.kruskal(*groups)
print(f"\nКритерий Краскела–Уоллиса: H = {stat_kw:.4f}, p-value = {p_kw:.6e}")

if p_kw < alpha:
    print("Отвергаем H0. Распределения pts_home статистически значимо различаются между типами сезона.")

    # попарный U-критерий Манна–Уитни с поправкой Бонферрони
    pairs = list(combinations(range(len(season_types)), 2))
    m = len(pairs)
    print(f"\nPost-hoc анализ (Манна–Уитни, m = {m}):")

    for i, j in pairs:
        u_stat, p_mw = stats.mannwhitneyu(groups[i], groups[j], alternative='two-sided')
        p_adj = min(p_mw * m, 1.0)
        signif = "Значимо" if p_adj < alpha else "Не значимо"
        print(f"  {season_types[i]:16s} vs {season_types[j]:16s}: U = {u_stat:.1f}, "
              f"p_raw = {p_mw:.6e}, p_adj = {p_adj:.6e} ({signif})")
else:
    print("Нет оснований отвергнуть H0.")


print("\n==== гипотеза 2: Взаимосвязь точности бросков (fg_pct_home) и разницы очков (plus_minus_home) ====")

df_clean = df[['fg_pct_home', 'plus_minus_home']].dropna()

# Проверка нормальности обеих переменных
_, p_nt_fg = stats.normaltest(df_clean['fg_pct_home'])
_, p_nt_pm = stats.normaltest(df_clean['plus_minus_home'])
print(f"Тест Д'Агостино-Пирсона: p_fg_pct = {p_nt_fg:.6e}, p_plus_minus = {p_nt_pm:.6e}")

# Корреляция Спирмена
rho, p_sp = stats.spearmanr(df_clean['fg_pct_home'], df_clean['plus_minus_home'])
print(f"\nКоэффициент корреляции Спирмена: rho = {rho:.4f}, p-value = {p_sp:.6e}")

if p_sp < alpha:
    direction = "положительная (прямая)" if rho > 0 else "отрицательная (обратная)"
    print(f"Вывод: Отвергаем H0. Выявлена статистически значимая {direction} связь.")
else:
    print("Вывод: Нет оснований отвергнуть H0.")


# Кодирование категорий и таблица корреляции
print("\n\n==== Кодирование и корреляция с целевым столбцом wl_home ====")

# убираем строки без целевого значения wl_home
df_encoded = df.dropna(subset=['wl_home']).copy()

# удаляем технические ID, текстовые описания (названия, даты, матчапы) и дубликат цели wl_away
cols_to_drop = [
    'season_id', 'team_id_home', 'team_abbreviation_home', 'team_name_home',
    'game_id', 'game_date', 'matchup_home',
    'team_id_away', 'team_abbreviation_away', 'team_name_away',
    'matchup_away', 'wl_away'
]
df_encoded = df_encoded.drop(columns=[c for c in cols_to_drop if c in df_encoded.columns])

# Применить LabelEncoding к бинарной целевой переменной wl_home (L -> 0, W -> 1)
le = LabelEncoder()
df_encoded['wl_home'] = le.fit_transform(df_encoded['wl_home'])

# Применить One-Hot Encoding к номинальному признаку season_type
df_encoded = pd.get_dummies(df_encoded, columns=['season_type'], dtype=int)

print("Первые 5 строк после кодирования:")
print(df_encoded.head())

# корреляция Спирмена т.к. нелинейность и ненорм распределений
corr_matrix = df_encoded.corr(method='spearman')

# Отдельная таблица корреляции признаков с целевым столбцом wl_home
target_corr = (
    corr_matrix[['wl_home']]
    .drop(index='wl_home')
    .sort_values(by='wl_home', ascending=False)
)

print("\nКорреляция признаков с целевым столбцом 'wl_home' (Спирмен):")
print(target_corr.round(4).to_string())

# для тепловой карты берем цель и топ-12 наиболее скоррелированных признаков (чтобы график не сливался)
top_features = target_corr.abs().sort_values(by='wl_home', ascending=False).head(12).index.tolist()
top_cols = ['wl_home'] + top_features

plt.figure(figsize=(12, 9))
sns.heatmap(corr_matrix.loc[top_cols, top_cols], annot=True, fmt='.2f', cmap='coolwarm', center=0, linewidths=0.5)
plt.title('Матрица корреляций топ-12 признаков и целевого столбца wl_home (Спирмен)')
plt.tight_layout()
plt.show()