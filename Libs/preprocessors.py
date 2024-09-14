from locale import normalize
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import OneHotEncoder


from sklearn.base import BaseEstimator, TransformerMixin



class TemporalVariableTransformer(BaseEstimator, TransformerMixin):
	# Temporal elapsed time transformer

    def __init__(self, variables):
        
        if not isinstance(variables, list):
            raise ValueError('variables should be a list')
        
        self.variables = variables

    def fit(self, X, y=None):
        # we need this step to fit the sklearn pipeline
        return self

    def transform(self, X):

    	# so that we do not over-write the original dataframe
        X = X.copy()
        
        for feature in self.variables:
            X[feature] = datetime.now().year - X[feature]

        return X



# categorical missing value imputer
class Mapper(BaseEstimator, TransformerMixin):

    def __init__(self, variables, mappings):

        if not isinstance(variables, list):
            raise ValueError('variables should be a list')

        self.variables = variables
        self.mappings = mappings

    def fit(self, X, y=None):
        # we need the fit statement to accomodate the sklearn pipeline
        return self

    def transform(self, X):
        X = X.copy()
        for feature in self.variables:
            X[feature] = X[feature].map(self.mappings)

        return X

class RareLabelCategoricalEncoder(BaseEstimator, TransformerMixin):
    """Groups infrequent categories into a single string"""

    def __init__(self, tol=0.05, variables=None):
            if not isinstance(variables, list):
                raise ValueError('variables should be a list')
        
            self.tol = tol
            self.variables = variables

    def fit(self, X, y=None):
        # persist frequent labels in dictionary
        self.encoder_dict_ = {}

        for var in self.variables:
            # the encoder will learn the most frequent categories
            t = pd.Series(X[var].value_counts(normalize))
            # frequent labels:
            self.encoder_dict_[var] = list(t[t >= self.tol].index)

        return self

    def transform(self, X):
        X = X.copy()
        for feature in self.variables:
            X[feature] = np.where(
                X[feature].isin(self.encoder_dict_[feature]),
                                X[feature], "Rare")

        return X


class CategoricalEncoder(BaseEstimator, TransformerMixin):
    """String to numbers categorical encoder."""

    def __init__(self, variables):

        if not isinstance(variables, list):
            raise ValueError('variables should be a list')
        
        self.variables = variables

    def fit(self, X, y):
        temp = pd.concat([X, y], axis=1)
        temp.columns = list(X.columns) + ["target"]

        # persist transforming dictionary
        self.encoder_dict_ = {}

        for var in self.variables:
            t = temp.groupby([var])["target"].mean().sort_values(ascending=True).index
            self.encoder_dict_[var] = {k: i for i, k in enumerate(t, 0)}

        return self

    def transform(self, X):
        # encode labels
        X = X.copy()
        for feature in self.variables:
            X[feature] = X[feature].map(self.encoder_dict_[feature])

        return X
    
class CategoricalOneHotEncoder(BaseEstimator, TransformerMixin):
    """Aplica OneHotEncoder a las variables categóricas especificadas."""

    def __init__(self, variables):
        # Asegurarse de que las variables sean una lista
        if not isinstance(variables, list):
            raise ValueError('variables debe ser una lista')
        self.variables = variables
        self.encoder = None  # Inicializamos el encoder

    def fit(self, X, y=None):
        """Ajusta el OneHotEncoder a los datos."""
        # Creamos un nuevo OneHotEncoder
        self.encoder = OneHotEncoder(sparse_output=False, drop='first')
        
        # Ajustamos el encoder solo a las variables categóricas seleccionadas
        self.encoder.fit(X[self.variables])
        return self

    def transform(self, X):
        """Aplica la transformación OneHotEncoder a los datos."""
        # Verificamos si ya hemos ajustado el encoder
        if self.encoder is None:
            raise ValueError('La instancia no ha sido ajustada todavía. Usa el método fit primero.')

        # Copiamos el DataFrame original
        X_copy = X.copy()

        # Codificamos las variables seleccionadas
        df_encoded = pd.DataFrame(self.encoder.transform(X_copy[self.variables]),
                                  columns=self.encoder.get_feature_names_out(self.variables),
                                  index=X_copy.index)

        # Eliminamos las columnas originales y concatenamos las codificadas
        X_copy.drop(columns=self.variables, inplace=True)
        X_copy = pd.concat([X_copy, df_encoded], axis=1)

        return X_copy
    
class ColumnSelector(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        columnas=[]
        for columna in self.columns:
            if columna in X.columns:
                columnas.append(columna)
        return X[columnas]
    
class MotorTypeTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy() 
        # Aplica las transformaciones
        X['motor_type_gas'] = X.apply(lambda x: 1 if x['motor_type_petrol and gas'] == 1 else x['motor_type_gas'], axis=1)
        X['motor_type_petrol'] = X.apply(lambda x: 1 if x['motor_type_petrol and gas'] == 1 else x['motor_type_petrol'], axis=1)
        
        # Elimina la columna 'motor_type_petrol and gas'
        X = X.drop(['motor_type_petrol and gas'], axis=1)
        
        return X
