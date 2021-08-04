# %%
import pandas as pd
import streamlit as st
import eda
import numpy as np
import os
# os.environ['HDF5_DISABLE_VERSION_CHECK']='2'
from feature_models import create_model, FeatureCreation
import pickle
from keras.wrappers.scikit_learn import KerasRegressor
from keras.models import load_model
import functools
from sklearn.model_selection import train_test_split
import graphs
# %%
st.set_page_config(
    page_title="Predicting Real Estate Prices in Brazil",
    page_icon="🧙‍♂️",
    layout="wide",
    initial_sidebar_state="expanded",
)

"""
# Predicting Rental Prices in Brazil
[![Star](https://img.shields.io/github/stars/arturlunardi/predict_rental_prices_streamlit?style=social)](https://github.com/arturlunardi/predict_rental_prices_streamlit)
&nbsp[![Follow](https://img.shields.io/badge/medium-arturlunardi-follow?style=social&logo=medium)](https://arturlunardi.medium.com/)
&nbsp[![Follow](https://img.shields.io/badge/Connect-follow?style=social&logo=linkedin)](https://www.linkedin.com/in/artur-lunardi-di-fante-393611194/)
"""

# ----------- Data -------------------


@st.cache
def get_raw_data():
    """
    This function return a pandas DataFrame with the raw data.
    """

    raw_df = pd.read_csv(os.path.join(os.path.abspath(''), 'data', 'houses_to_rent_v2.csv'))
    return raw_df


@st.cache
def get_cleaned_data():
    """
    This function return a pandas DataFrame with the cleaned data.
    """

    clean_data = pd.read_csv(os.path.join(os.path.abspath(''), 'data', 'houses_to_rent_v2_fteng.csv'))
    return clean_data


@st.cache
def get_raw_eval_df():
    """
    This function return a pandas DataFrame with the dataframe and the machine learning models along with it's metrics.
    """

    raw_eval_df = pd.read_csv(os.path.join(os.path.abspath(''), 'data', 'model_evaluation.csv'))
    return raw_eval_df


@st.cache(hash_funcs={pd.DataFrame: lambda x: x})
def load_models_df(dataframe):
    df_evaluated = dataframe.copy()
    models_list = os.listdir(os.path.join(os.path.abspath(''), 'models'))
    rep = {"pipe": "model", "pickle": "h5"}
    for index, row in df_evaluated.iterrows():
        # check if the file_name is in our models directory
        if row['pipe_file_name'] in models_list:
            # now, load the model.
            with open(os.path.join(os.path.abspath(''), 'models', row['pipe_file_name']), 'rb') as fid:
                model_trained = pickle.load(fid)
            
            # for the keras model, we have to load the model separately and add into the pipeline or transformed target object.
            if row['name'] == 'NeuralNetwork':
                model_keras = load_model(os.path.join(os.path.abspath(''), 'models', functools.reduce(lambda a, kv: a.replace(*kv), rep.items(), row['pipe_file_name'])))
                # check if the target transformer it is active
                if row['custom_target']:
                    # reconstruct the model inside a kerasregressor and add inside the transformed target object
                    model_trained.regressor.set_params(model = KerasRegressor(build_fn=create_model, verbose=0))
                    # add the keras model inside the pipeline object
                    model_trained.regressor_.named_steps['model'].model = model_keras
                else:
                    model_trained.named_steps['model'].model = model_keras

            df_evaluated.loc[index, 'model_trained'] = model_trained

    # we have to transform our score column to bring it back to a python list
    df_evaluated['all_scores_cv'] = df_evaluated['all_scores_cv'].apply(lambda x: [float(i) for i in x.strip('[]').split()])
    
    return df_evaluated.sort_values(by='rmse_cv').reset_index(drop=True)


@st.cache
def split(dataframe):
    df = dataframe.copy()
    x = df.drop(columns=['rent amount (R$)'], axis=1)
    y = df['rent amount (R$)']
    # check if the random state it is equal to when it was trained, this is very important.
    x_train, x_test, y_train, y_test = train_test_split(x,
                                                        y,
                                                        test_size=0.25,
                                                        random_state=0)

    return x, y, x_train, x_test, y_train, y_test

raw_df = get_raw_data()
clean_df = get_cleaned_data()
raw_eval_df = get_raw_eval_df()
eval_df = load_models_df(raw_eval_df)
x, y, x_train, x_test, y_train, y_test = split(clean_df)

# ----------- Global Sidebar ---------------

condition = st.sidebar.selectbox(
    "Select the visualization",
    ("Introduction", "EDA", "Model Prediction", "Model Evaluation")
)

# ------------- Introduction ------------------------

if condition == 'Introduction':
    st.image(os.path.join(os.path.abspath(''), 'data', 'dataset-cover.jpg'))
    st.subheader('About')
    
    ## FALTA O CHECK ON GITHUB
    st.write("""
    This application provides an overview of the brazilian_houses_to_rent dataset from Kaggle. It is a dataset that provides rent prices for real estate properties in Brazil.

    The data were provided from this [source](https://www.kaggle.com/rubenssjr/brasilian-houses-to-rent). 

    You can check on the sidebar:
    - EDA (Exploratory Data Analysis)
    - Model Prediction
    - Model Evaluation

    The prediction are made regarding to the rent amount utilizing pre trained machine learning models.

    All the operations in the dataset were already done and stored as csv files inside the data directory. If you want to check the code, go through the notebook directory in the [github repository](https://github.com/arturlunardi/predict_rental_prices_streamlit).
    """)

    st.subheader('Model Definition')

    st.write("""
    The structure of the training it is to wrap the process around a scikit-learn Pipeline. There were 4 possible combinations and 5 models, resulting in 20 trained models.

    The combinations are regarding to perform Feature Creation and/or Target Transformations in the dataset.

    Models:
    - Random Forest
    - XGB
    - Ridge
    - LGBM
    - Neural Network

    Our main accuracy metric is RMSE. To enhance our model definition, we utilized Cross Validation and Random Search for hyperparameter tuning.
    """)

# ------------- EDA ------------------------

elif condition == 'EDA':
    type_of_data = st.radio(
        "Type of Data",
        ('Raw Data', 'Cleaned Data'),
        help='Data source that will be displayed in the charts'
    )

    if type_of_data == 'Raw Data':
        data = raw_df.copy()
    else:
        data = clean_df.copy()

    with st.beta_container():
        st.header('Descriptive Statistics\n')
        col1, col2 = st.beta_columns([1, 3])
        col1.dataframe(eda.summary_table(data))
        col2.dataframe(data.describe())

    st.header('Data Visualization')

    height, width, margin = 450, 1500, 10

    st.subheader('Rent Amount Distribution')

    select_city_eda = st.selectbox(
        'Select the City',
        ['All'] + [i for i in data['city'].unique()]
    )

    if select_city_eda == 'All':
        fig = graphs.plot_histogram(data=data, x="rent amount (R$)", nbins=50, height=height, width=width, margin=margin)
    else:
        fig = graphs.plot_histogram(
            data = data.loc[data['city'] == select_city_eda], x="rent amount (R$)", nbins=50, height=height, width=width, margin=margin)
                      
    st.plotly_chart(fig)

    st.subheader('Scatterplot')

    select_numerical = st.selectbox(
        'Select the Numerical Variable',
        ['area', 'hoa (R$)', 'property tax (R$)', 'fire insurance (R$)']
    )

    fig = graphs.plot_scatter(data=data, x=select_numerical, y="rent amount (R$)", height=height, width=width, margin=margin)

    st.plotly_chart(fig)

    st.subheader('Categorical Graphs')

    select_graph = st.radio(
        'Select the Type of Graph',
        ('Boxplot', 'Countplot')
    )

    select_variable = st.selectbox(
        'Select the Variable',
        [i for i in data.columns if data[i].dtype == object and i != 'floor']
    )

    if select_graph == 'Boxplot':
        fig = graphs.plot_boxplot(data=data, x=select_variable, y="rent amount (R$)", color=select_variable, height=height, width=width, margin=margin)
    elif select_graph == 'Countplot':
        fig = graphs.plot_countplot(data=data, x=select_variable, height=height, width=width, margin=margin)

    st.plotly_chart(fig)

    st.subheader('Rent amount mean per Variable')

    option = st.selectbox(
        'Select the Column',
        ('rooms', 'bathroom', 'parking spaces'),
    )

    fig = graphs.plot_bar(data=data.groupby(option).mean().reset_index(), x=option, y='rent amount (R$)', height=height, width=width, margin=margin)

    st.plotly_chart(fig)

    st.subheader('Correlation Matrix')

    corr_matrix = data.corr()

    fig = graphs.plot_heatmap(corr_matrix=corr_matrix, height=height, margin=margin)

    st.plotly_chart(fig)


# -------------------------------------------

elif condition == 'Model Prediction':

    select_model_mpredict = st.sidebar.selectbox(
        'Select the Model',
        [i for i in eval_df['name'].unique()]  
    )

    select_custom_features_mpredict = st.sidebar.select_slider(
        'Create Custom Features?',
        [False, True],
        help='Feature Creation according to the FeatureCreation class in the load_models module'
    )

    select_custom_target_mpredict = st.sidebar.select_slider(
        'Perform Target Transformation?',
        [False, True],
        help='Perform a logarithm transformation in the target variable'
    )

    select_city = st.sidebar.selectbox(
        'Select the City',
        clean_df['city'].value_counts().index
    )

    select_area = st.sidebar.number_input(
        'Select the value of Area',
        help='The value must be in square meters (m²)',
        min_value=1,
    )

    select_rooms = st.sidebar.number_input(
        'Select the number of Rooms',
        min_value=1,
    )

    select_bathrooms = st.sidebar.number_input(
        'Select the number of Bathrooms',
        min_value=1,
    )

    select_parking_spaces = st.sidebar.number_input(
        'Select the number of Parking Spaces',
        min_value=0,
    )

    select_animal = st.sidebar.select_slider(
        'Accept Animals?',
        ['acept', 'not acept']
    )

    select_furniture = st.sidebar.select_slider(
        'It is furnished',
        ['furnished', 'not furnished']
    )

    select_hoa = st.sidebar.number_input(
        'Select the value of Hoa',
        help='The values must be in Reais (R$)',
        min_value=0,
    )

    select_property_tax = st.sidebar.number_input(
        'Select the value of Property Tax',
        help='The values must be in Reais (R$)',
        min_value=0,
    )

    select_fire_insurance = st.sidebar.number_input(
        'Select the value of Fire Insurance',
        help='The values must be in Reais (R$)',
        min_value=0,
    )

    predict_array = [select_city, select_area, select_rooms, select_bathrooms, select_parking_spaces, select_animal, select_furniture, select_hoa, select_property_tax, select_fire_insurance]

    model_trained_mpredict = eval_df.loc[(eval_df['name'] == select_model_mpredict) & (eval_df['custom_features'] == select_custom_features_mpredict) & (eval_df['custom_target'] == select_custom_target_mpredict)]['model_trained'].iloc[0]

    value_to_predict = pd.DataFrame(
        [predict_array], columns=clean_df.drop(columns='rent amount (R$)').columns
    )

    st.subheader('Available Models')

    st.dataframe(eval_df.drop(columns=['all_scores_cv', 'pipe_file_name', 'model_trained']))

    if st.button('Predict', help='Be certain to check the parameters on the sidebar'):
        predicted_value = model_trained_mpredict.predict(value_to_predict)
        st.success(f'The predicted value is R$ {round(predicted_value[0], 2)}')

        with st.beta_expander("Model Parameters"):
            st.write(f"The model chosen was {select_model_mpredict}. \n\n Parameters:", eval(eval_df.loc[(eval_df['name'] == select_model_mpredict) & (eval_df['custom_features'] == select_custom_features_mpredict) & (eval_df['custom_target'] == select_custom_target_mpredict)]['params'].iloc[0])[0])


# -------------------------------------------

elif condition == 'Model Evaluation':
    st.subheader('Available Models')

    st.dataframe(eval_df.drop(columns=['all_scores_cv', 'pipe_file_name', 'model_trained']))

    select_model_meval = st.sidebar.selectbox(
        'Select the Model',
        [i for i in eval_df['name'].unique()]  
    )

    select_custom_features_meval = st.sidebar.select_slider(
        'Create Custom Features?',
        [False, True]
    )

    select_custom_target_meval = st.sidebar.select_slider(
        'Perform Target Transformation?',
        [False, True]
    )

    model_trained_meval = eval_df.loc[(eval_df['name'] == select_model_meval) & (eval_df['custom_features'] == select_custom_features_meval) & (eval_df['custom_target'] == select_custom_target_meval)]['model_trained'].iloc[0]

# -------------- figs -----------------

    height, width, margin = 450, 1500, 30

    st.subheader('Distribution of the Target Variable')

    fig = graphs.plot_distplot(
        y_real=y_test, 
        y_predict=model_trained_meval.predict(x_test),
        height=height, 
        width=width, 
        margin=margin,
        title_text='Predicted and Real Value'
    )

    st.plotly_chart(fig)

    st.subheader('Distribution of the Residuals')

    # predict the values of the entire data
    prediction = model_trained_meval.predict(x)
    # calculate the residual
    resid = prediction - y

    # create a copy to not alter the original data
    df_plot = clean_df.copy()
    # create a column to identify the data regarding to train or test
    df_plot['split'] = 'train'
    df_plot.loc[x_test.index, 'split'] = 'test'
    df_plot['prediction'] = prediction
    df_plot['resid'] = resid

    # plot the residual plot with the histograms
    fig = graphs.plot_scatter(data=df_plot, x='prediction', y='resid', residual=True, height=height, width=width, margin=margin, title_text='Residuals per Split')
    
    st.plotly_chart(fig)

    st.subheader('Boxplot of RMSE in Cross Validation')

    fig = graphs.plot_boxplot(data=eval_df, x=None, y=None, model_name=select_model_meval, custom_feature=select_custom_features_meval, custom_target=select_custom_target_meval, single_box=True, title_text='Cross Validation with 5 Folds', height=height, width=width, margin=margin)

    st.plotly_chart(fig)

