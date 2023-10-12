# Title: Data-driven pre-determination of Cu oxidation state in copper nanoparticles: application to the synthesis by laser ablation in liquid
# Author: Runpeng Miao, Michael Bissoli, Andrea Basagni, Ester Marotta, Stefano Corni, Vincenzo Amendola,*
# Correspondence: runpeng.miao@phd.unipd.it, vincenzo.amendola@unipd.it

import pandas as pd
import random

def extract_rows(data):
    # Randomly select a number that is not in the excluded_numbers list
    excluded_numbers = [4, 24, 26, 43, 45, 50, 51, 52, 66, 72, 74, 85, 87, 94, 101, 105]
    random_number = random.choice([num for num in range(1, 105) if num not in excluded_numbers])
    print("Random number:", random_number)

    df = pd.DataFrame(data)

    # Extract initial row
    initial_row = df[df['Article ID '] == random_number]

    # Get initial row index and name
    initial_row_index = initial_row.index.tolist()
    initial_row_name = df.loc[initial_row_index, 'Article ID ']

    extracted_rows = []
    extracted_rows.extend(initial_row_index)

    List_1 = data.copy()

    List_2 = df.loc[extracted_rows]

    remaining_rows = [i for i in range(len(List_1)) if i not in extracted_rows]
    List_3 = df.loc[remaining_rows]

    # Step 2: Calculate the maximum allowable size of List_2 based on List_1
    max_size = len(List_1) * 0.2

    # Create a set to store used indices
    used_indices = set()

    # Step 3: Continuously extract rows from List_3 until the size of List_2 reaches the maximum allowable size
    while len(List_2) <= max_size and len(List_3) > 0:
        List_3 = List_3.reset_index(drop=True)
        # Generate a new random index that hasn't been used before
        random_index = random.randint(0, len(List_3) - 1)
        while random_index in used_indices:
            random_index = random.randint(0, len(List_3) - 1)
        print("Random index:", random_index)

    # Add the new index to the set of used indices
        used_indices.update(initial_row_index)
        used_indices.add(random_index)

    # Extract all rows with the same Article ID from List_3
        article_id = List_3.loc[random_index, 'Article ID ']
        extracted_rows = List_3[List_3['Article ID '] == article_id]

    # Delete the extracted rows from List_3
        List_3 = List_3.drop(extracted_rows.index)
        List_3 = List_3.reset_index(drop=True)
    # Append the extracted rows to List_2
        List_2 = List_2.append(extracted_rows)

    row_number = len(List_2) + len(List_3)
    print("Total number of rows:", row_number)
    print('rows_number_20_percent',len(List_2))
    print('rows_number_80_percent',len(List_3))
    df_20_percent = List_2.copy()
    df_80_percent = List_3.copy()

    # Check if there are common names in 'Article ID' column between df_80_percent and df_20_percent
    common_names = set(df_80_percent['Article ID']) & set(df_20_percent['Article ID'])
    if len(common_names) > 0:
       print("There are common names between df_80_percent and df_20_percent:")
       print(common_names)
    else:
       print("No common names found between df_80_percent and df_20_percent.")

    # delete unwanted columns
    X_test = df_20_percent.drop(columns=['Article ID ', 'ID experiment'], axis=1)
    X_train = df_80_percent.drop(columns=['Article ID ', 'ID experiment'], axis=1)

    X_train = X_train[['Pulse duration (s)', 'Repetition rate (Hz)',
                             'Pulse energy (J/pulse) [1]', 'Lens focal (cm) [1]',
                             'Duration of synthesis (min) [1]',
                             'Solvent number of atoms', '% of O+Cl+CN+S in solvent',
                             '% of O+Cl+CN+S in solute', 'Mass fraction of solute in solvent']]
    X_test = X_test[['Pulse duration (s)', 'Repetition rate (Hz)',
                            'Pulse energy (J/pulse) [1]', 'Lens focal (cm) [1]',
                            'Duration of synthesis (min) [1]',
                            'Solvent number of atoms', '% of O+Cl+CN+S in solvent',
                            '% of O+Cl+CN+S in solute', 'Mass fraction of solute in solvent']]

    new_column_order = ['pulse duration [s]', 'Repetition rate (r) [Hz]', 'Pulse energy (J/pulse)',
                        'Lens focal [cm]', 'Duration of synthesis (min)', 'Solvent Number of atoms',
                        '%O+Cl+CN+S in solvent', '%O+Cl+CN+S of solute', 'mass fraction solute']

    X_train.columns = new_column_order

    X_test.columns = new_column_order
    
    import re
    regex = re.compile(r"\[|\]|<", re.IGNORECASE)
    X_train.columns = [regex.sub("_", col) if any(x in str(col) for x in set(('[', ']', '<'))) else col for col in X_train.columns.values]
    X_test.columns = [regex.sub("_", col) if any(x in str(col) for x in set(('[', ']', '<'))) else col for col in X_test.columns.values]

    y_train = df_80_percent['Output 1: Mean oxidation state of products']
    y_test = df_20_percent['Output 1: Mean oxidation state of products']
    return X_train, y_train, X_test, y_test, df_80_percent,df_20_percent,random_number
