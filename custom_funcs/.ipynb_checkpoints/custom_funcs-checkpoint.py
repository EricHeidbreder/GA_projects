def category_to_bool_cols(dataframe, column):
    dummy_split = pd.get_dummies(dataframe[column], column) # Creates dummy columns with the name {column}_{value_in_row} per get_dummies documentation
    for dummy_key in dummy_split: # Iterates through dummy_key in dummy_split
        dataframe[dummy_key] = dummy_split[dummy_key] # adds new columns named {dummy_key} to original dataframe