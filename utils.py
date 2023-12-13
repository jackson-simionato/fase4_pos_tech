from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

class DropFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, feature_to_drop='ID_Cliente') -> None:
        self.feature_to_drop = feature_to_drop

    def fit(self, df):
        return self
    
    def transform(self, df):
        if self.feature_to_drop in df.columns:
            drop_df = df.drop(columns=[self.feature_to_drop])
            return drop_df
        else:
            print(f'Variável {self.feature_to_drop} não encontrada no dataframe')
            return df
        
class MinMax(BaseEstimator, TransformerMixin):
    def __init__(self, min_max_scaler=['Idade', 'Anos_empregado', 'Tamanho_familia','Rendimento_anual']):
        self.min_max_scaler = min_max_scaler

    def fit(self, df):
        return self
    
    def transform(self, df):
        if set(self.min_max_scaler).issubset(df.columns):
            scaler = MinMaxScaler()
            df[self.min_max_scaler] = scaler.fit_transform(df[self.min_max_scaler])

            return df
        else:
            print(f'Colunas {[col for col in self.min_max_scaler if col not in df.columns]} não encontradas')
            return df
        
class OneHotEncoding(BaseEstimator, TransformerMixin):
    def __init__(self, one_hot_encoder=['Categoria_de_renda','Estado_civil','Moradia','Ocupacao']):
        self.one_hot_encoder = one_hot_encoder

    def fit(self, df):
        return self
    
    def transform(self, df):
        if set(self.one_hot_encoder).issubset(df.columns):
            def one_hot_enc(df, one_hot_encoder):
                one_hot_enc = OneHotEncoder()
                one_hot_enc.fit(df[one_hot_encoder])
                feature_names = one_hot_enc.get_feature_names_out(one_hot_encoder)
                df = pd.DataFrame(one_hot_enc.transform(df[self.one_hot_encoder]).toarray(),
                                columns=feature_names, index=df.index)
                df[feature_names] = df[feature_names].astype(int)
            
                return df
            
            def concat_result(df, one_hot_enc_df, one_hot_encoder):
                other_features = [column for column in df.columns if column not in one_hot_encoder]
                df_concat = pd.concat([df[other_features], one_hot_enc_df], axis=1)

                return df_concat
            
            df_OneHotEncoding = one_hot_enc(df, self.one_hot_encoder)
            df_final = concat_result(df, df_OneHotEncoding, self.one_hot_encoder)

            return df_final
            
        else:
            print(f'Colunas {[col for col in self.one_hot_encoder if col not in df.columns]} não encontradas')
            return df

class OrdinalFeature(BaseEstimator, TransformerMixin):
    def __init__(self, ordinal_feature=['Grau_escolaridade']):
        self.ordinal_feature = ordinal_feature

    def fit(self, df):
        return self
    
    def transform(self, df):
        if self.ordinal_feature[0] in df.columns:
            ordinal_encoder = OrdinalEncoder(dtype=int)
            df[self.ordinal_feature] = ordinal_encoder.fit_transform(df[self.ordinal_feature])

            return df
        else:
            print(f'Variável {self.ordinal_feature} não encontrada no dataframe!')
            return df
        
class OverSample(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, df):
        return self
    
    def transform(self, df):
        oversampler = SMOTE(sampling_strategy='minority')
        x_bal, y_bal = oversampler.fit_resample(df.drop(columns=['Risco_de_credito']), df['Risco_de_credito'])
        df_bal = pd.concat([pd.DataFrame(x_bal), pd.DataFrame(y_bal)], axis=1)

        return df_bal
    

    def data_split(df, test_size):
        SEED = 1561651
        df_train, df_test = train_test_split(df, test_size, random_state=SEED)
        df_train.reset_index(drop=True, inplace=True)
        df_test.reset_index(drop=True, inplace=True)

        return df_train, df_test