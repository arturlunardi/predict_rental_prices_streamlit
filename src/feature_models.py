from sklearn.base import BaseEstimator, TransformerMixin
from keras.models import Sequential
from keras.layers import Dense, Dropout


def create_model(optimizer='adam', dropout=0.2, activation='relu', kernel_initializer='normal'):
    model = Sequential()
    model.add(Dense(units = 15, activation = activation, input_dim = 15, kernel_initializer=kernel_initializer))
    model.add(Dropout(dropout))
    model.add(Dense(units = 11, activation = activation))
    model.add(Dropout(dropout))
    model.add(Dense(units = 1, activation = activation))

    model.compile(optimizer = optimizer, loss = 'mean_squared_error', metrics = ['accuracy'])

    return model


class FeatureCreation(BaseEstimator, TransformerMixin):
    def __init__(self):
        # print('FeatureCreation initialized')
        return None
    
    # For the fit method, we will pass the parameter x. This is our independent variables. 
    # This fit method will be called when we fit the pipeline.
    def fit(self, x, y=None):
        # print('Fit FeatureCreation called')
        return self
    
    # Here, we will perform all of our transformations. For creating features automatic, we could create parameters in the class and pass the column names to them.
    # But in this case, since it's for this dataset specific, we will perform transformations in the column names directly into the fit method.
    # The transform method is called when we fit and when we predict using the Pipeline. And that's make sense, since we need to create our feature when we will train and when we will predict.
    def transform(self, x, y=None):
        # print('Transform FeatureCreation called')
        # creating a copy to avoid changes to the original dataset
        x_ = x.copy()
        # print(f'Before Transformation: {x_.shape}')
        # and now, we create everyone of our features.
        # Area power of two
        x_['area2'] = x_['area'] ** 2
        # The ratio between area and rooms
        x_['area/room'] = x_['area'] / x_['rooms']
        # The ratio between area and bathroom
        x_['area/bathroom'] = x_['area'] / x_['bathroom']
        # the sum of rooms and bathrooms
        x_['rooms+bathroom'] = x_['rooms'] + x_['bathroom']
        # the product between rooms and bathrooms
        x_['rooms*bathroom'] = x_['rooms'] * x_['bathroom']
        # the ratio between rooms and bathrooms
        x_['rooms/bathroom'] = x_['rooms'] / x_['bathroom']
        # the product between hoa and property tax
        x_['hoa*property tax'] = x_['hoa (R$)'] * x_['property tax (R$)']
        # print(f'After Transformation: {x_.shape}')
        return x_