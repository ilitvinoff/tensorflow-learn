import pandas as pd

data = pd.read_csv('data/titanic.csv', index_col='PassengerId')

# data.columns ['Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']

men_women = " ".join([str(i) for i in data['Sex'].value_counts()])
print("men, women: ", men_women)


def get_percent(df_column, query_param=None):
    if query_param is not None:
        return (sum([x for x in df_column if x == query_param]) / df_column.count()) * 100

    return (df_column.sum() / df_column.count()) * 100


survived = data['Survived']

print("Survived: %.2f" % (get_percent(survived)))

first_class = data['Pclass']
print("1st class passengers: %.2f" % (get_percent(first_class, 1)))

age = data['Age']

print(f"Average: {age.mean()}, median: {age.median()}")

print(f"Correlation: {data['SibSp'].corr(other=data['Parch'], method='pearson')}")

print('##########################################')


def get_names(full_names):
    ret = []
    for full_name in full_names:
        if 'Miss.' not in full_name and 'Mrs.' not in full_name:
            continue
        name = full_name.split('Miss.')[1] if 'Miss.' in full_name else full_name.split('Mrs.')[1]
        name = name.strip()

        if '(' in name:
            name = name.split('(')[-1].strip(')')

        name = name.split()[0]
        ret.append(name)

    return ret


names = get_names([row['Name'] for _, row in data.iterrows() if row['Sex'] == 'female'])
most_popular = max(set(names), key=names.count)
print(f"{names.count(most_popular)} females with name {most_popular}")

new_df = pd.DataFrame({'Name': names})
print(new_df.Name.mode().values[0])